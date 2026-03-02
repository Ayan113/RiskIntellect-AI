"""
RAG query API endpoints.

Provides regulatory document search and question answering
with guardrails against prompt injection.
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_hybrid_retriever, get_rag_guardrails, get_reasoning_engine
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/rag", tags=["RAG - Regulatory Search"])


class RAGQueryRequest(BaseModel):
    """Schema for RAG query input."""

    query: str = Field(..., min_length=3, max_length=2000, description="Regulatory question")
    top_k: int = Field(5, ge=1, le=20, description="Number of results to return")
    use_reranker: bool = Field(True, description="Whether to apply cross-encoder reranking")


class RAGSearchResult(BaseModel):
    """Schema for a single search result."""

    text: str
    source: str
    score: float
    retrieval_method: str


class RAGQueryResponse(BaseModel):
    """Schema for RAG query response."""

    query: str
    results: List[RAGSearchResult]
    num_results: int


class RAGAnswerRequest(BaseModel):
    """Schema for RAG question-answering input."""

    query: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)


class RAGAnswerResponse(BaseModel):
    """Schema for RAG question-answering output."""

    query: str
    answer: str
    grounded: bool
    guardrail_warnings: List[str]
    sources: List[RAGSearchResult]


@router.post("/search", response_model=RAGQueryResponse)
async def search_regulatory_docs(request: RAGQueryRequest):
    """
    Search regulatory documents using hybrid retrieval.

    Combines BM25 keyword search with dense vector similarity,
    then reranks results using a cross-encoder for precision.
    """
    # Validate query through guardrails
    guardrails = get_rag_guardrails()
    is_safe, sanitized_query, rejection_reason = guardrails.validate_query(request.query)

    if not is_safe:
        raise HTTPException(
            status_code=400,
            detail=f"Query rejected: {rejection_reason}",
        )

    try:
        retriever = get_hybrid_retriever()
        results = retriever.retrieve(
            query=sanitized_query,
            use_reranker=request.use_reranker,
        )

        search_results = [
            RAGSearchResult(
                text=r.text,
                source=r.metadata.get("source", "unknown"),
                score=r.score,
                retrieval_method=r.retrieval_method,
            )
            for r in results[: request.top_k]
        ]

        return RAGQueryResponse(
            query=sanitized_query,
            results=search_results,
            num_results=len(search_results),
        )
    except Exception as e:
        logger.error(f"RAG search failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/answer", response_model=RAGAnswerResponse)
async def answer_regulatory_question(request: RAGAnswerRequest):
    """
    Answer a regulatory question using RAG.

    Retrieves relevant documents and generates a grounded answer
    using the LLM reasoning layer with hallucination guardrails.
    """
    # Validate query
    guardrails = get_rag_guardrails()
    is_safe, sanitized_query, rejection_reason = guardrails.validate_query(request.query)

    if not is_safe:
        raise HTTPException(
            status_code=400,
            detail=f"Query rejected: {rejection_reason}",
        )

    try:
        # Retrieve context
        retriever = get_hybrid_retriever()
        retrieval_results = retriever.retrieve(
            query=sanitized_query,
            use_reranker=True,
        )

        contexts = [r.text for r in retrieval_results[: request.top_k]]

        # Filter contexts through guardrails
        safe_contexts = guardrails.validate_retrieved_context(contexts)

        # Generate answer
        engine = get_reasoning_engine()
        answer_result = engine.answer_regulatory_query(
            query=sanitized_query,
            contexts=safe_contexts,
        )

        sources = [
            RAGSearchResult(
                text=r.text[:200] + "...",
                source=r.metadata.get("source", "unknown"),
                score=r.score,
                retrieval_method=r.retrieval_method,
            )
            for r in retrieval_results[: request.top_k]
        ]

        return RAGAnswerResponse(
            query=sanitized_query,
            answer=answer_result["answer"],
            grounded=answer_result["grounded"],
            guardrail_warnings=answer_result["guardrail_warnings"],
            sources=sources,
        )
    except Exception as e:
        logger.error(f"RAG answer failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
