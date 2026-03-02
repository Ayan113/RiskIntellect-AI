"""
Hybrid retrieval with reciprocal rank fusion and cross-encoder reranking.

Combines BM25 (sparse) and FAISS (dense) retrieval results,
then reranks using a cross-encoder for maximum precision.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from rag_engine.bm25_index import BM25Index
from rag_engine.embeddings import EmbeddingGenerator
from rag_engine.vector_store import VectorStore
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)
config = Config()


class RetrievalResult:
    """Structured result from the retrieval pipeline."""

    def __init__(
        self,
        text: str,
        metadata: Dict[str, str],
        score: float,
        retrieval_method: str,
    ) -> None:
        self.text = text
        self.metadata = metadata
        self.score = score
        self.retrieval_method = retrieval_method

    def to_dict(self) -> Dict:
        return {
            "text": self.text,
            "metadata": self.metadata,
            "score": round(self.score, 4),
            "retrieval_method": self.retrieval_method,
        }


class HybridRetriever:
    """
    Hybrid retrieval combining BM25 + FAISS with cross-encoder reranking.

    Pipeline:
    1. Query is sent to both BM25 (sparse) and FAISS (dense) indices.
    2. Results are merged using Reciprocal Rank Fusion (RRF).
    3. Top candidates are reranked by a cross-encoder for precision.

    Design decision: RRF over learned fusion
    - RRF: Simple, parameter-free, robust across domains.
      Score = Σ 1/(k + rank_i) for each result across retrieval methods.
    - Learned fusion: Better in theory, but requires training data.
      Overkill for regulatory document retrieval.

    Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2
    - Adds ~200ms latency per query but dramatically improves precision.
    - Only runs on top-20 candidates, not the entire corpus.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        bm25_index: Optional[BM25Index] = None,
        embedding_generator: Optional[EmbeddingGenerator] = None,
    ) -> None:
        self.vector_store = vector_store or VectorStore()
        self.bm25_index = bm25_index or BM25Index()
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

        rag_config = config.get_section("rag_engine")
        retrieval_config = rag_config.get("retrieval", {})

        self.top_k_vector: int = retrieval_config.get("top_k_vector", 10)
        self.top_k_bm25: int = retrieval_config.get("top_k_bm25", 10)
        self.top_k_final: int = retrieval_config.get("top_k_final", 5)
        self.reranker_model_name: str = retrieval_config.get(
            "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.fusion_weight_vector: float = retrieval_config.get(
            "fusion_weight_vector", 0.6
        )
        self.fusion_weight_bm25: float = retrieval_config.get(
            "fusion_weight_bm25", 0.4
        )

        self._reranker = None

    def retrieve(self, query: str, use_reranker: bool = True) -> List[RetrievalResult]:
        """
        Execute hybrid retrieval pipeline.

        Args:
            query: User query string.
            use_reranker: Whether to apply cross-encoder reranking.

        Returns:
            List of RetrievalResult objects sorted by relevance.
        """
        logger.info(f"Hybrid retrieval for query: '{query[:100]}...'")

        # Stage 1: Dense retrieval (FAISS)
        query_embedding = self.embedding_generator.embed_query(query)
        vector_results = self.vector_store.search(
            query_embedding, top_k=self.top_k_vector
        )

        # Stage 2: Sparse retrieval (BM25)
        bm25_results = self.bm25_index.search(query, top_k=self.top_k_bm25)

        # Stage 3: Reciprocal Rank Fusion
        fused = self._reciprocal_rank_fusion(vector_results, bm25_results)

        logger.info(f"RRF produced {len(fused)} candidates")

        # Stage 4: Cross-encoder reranking (optional)
        if use_reranker and fused:
            fused = self._rerank(query, fused)
            logger.info(f"Reranked to {len(fused)} results")

        # Return top-k final results
        return fused[: self.top_k_final]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Tuple[str, Dict, float]],
        bm25_results: List[Tuple[str, Dict, float]],
        k: int = 60,
    ) -> List[RetrievalResult]:
        """
        Merge results from multiple retrieval methods using RRF.

        RRF Score = Σ weight_i / (k + rank_i)

        Args:
            vector_results: Results from FAISS.
            bm25_results: Results from BM25.
            k: RRF constant (default 60, as per original paper).

        Returns:
            Merged and scored list of RetrievalResult.
        """
        # Text → aggregated RRF score
        score_map: Dict[str, float] = {}
        text_map: Dict[str, Tuple[str, Dict]] = {}

        # Score vector results
        for rank, (text, meta, _) in enumerate(vector_results):
            text_key = text[:200]  # Use prefix as key for dedup
            rrf_score = self.fusion_weight_vector / (k + rank + 1)
            score_map[text_key] = score_map.get(text_key, 0) + rrf_score
            text_map[text_key] = (text, meta)

        # Score BM25 results
        for rank, (text, meta, _) in enumerate(bm25_results):
            text_key = text[:200]
            rrf_score = self.fusion_weight_bm25 / (k + rank + 1)
            score_map[text_key] = score_map.get(text_key, 0) + rrf_score
            if text_key not in text_map:
                text_map[text_key] = (text, meta)

        # Sort by RRF score
        sorted_keys = sorted(score_map.keys(), key=lambda x: score_map[x], reverse=True)

        results: List[RetrievalResult] = []
        for key in sorted_keys:
            text, meta = text_map[key]
            results.append(
                RetrievalResult(
                    text=text,
                    metadata=meta,
                    score=score_map[key],
                    retrieval_method="hybrid_rrf",
                )
            )

        return results

    def _rerank(
        self, query: str, candidates: List[RetrievalResult]
    ) -> List[RetrievalResult]:
        """
        Rerank candidates using a cross-encoder model.

        Args:
            query: Original query.
            candidates: List of retrieval results to rerank.

        Returns:
            Reranked list of RetrievalResult.
        """
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading reranker: {self.reranker_model_name}")
            self._reranker = CrossEncoder(self.reranker_model_name)

        # Create query-document pairs
        pairs = [(query, c.text) for c in candidates]
        scores = self._reranker.predict(pairs)

        # Update scores and sort
        for candidate, score in zip(candidates, scores):
            candidate.score = float(score)
            candidate.retrieval_method = "hybrid_rrf_reranked"

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates

    def load_indices(self) -> None:
        """Load both FAISS and BM25 indices from disk."""
        try:
            self.vector_store.load()
        except FileNotFoundError:
            logger.warning("FAISS index not found. Run indexing first.")
        try:
            self.bm25_index.load()
        except FileNotFoundError:
            logger.warning("BM25 index not found. Run indexing first.")
