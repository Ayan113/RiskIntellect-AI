"""
Health check endpoints.
"""

from fastapi import APIRouter

from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Health"])


@router.get("/health")
async def health_check():
    """
    System health check endpoint.

    Returns component availability status for monitoring
    and load balancer health probes.
    """
    components = {}

    # Check ML model
    try:
        from api.dependencies import get_fraud_predictor

        predictor = get_fraud_predictor()
        components["ml_engine"] = "ready" if getattr(predictor, '_is_loaded', False) else "not_loaded"
    except Exception as e:
        components["ml_engine"] = f"not_loaded"

    # Check RAG indices
    try:
        from api.dependencies import get_hybrid_retriever

        retriever = get_hybrid_retriever()
        vs_size = getattr(getattr(retriever, 'vector_store', None), 'size', 0)
        bm25_size = getattr(getattr(retriever, 'bm25_index', None), 'size', 0)
        components["rag_vector_store"] = "ready" if vs_size > 0 else "empty"
        components["rag_bm25"] = "ready" if bm25_size > 0 else "empty"
    except Exception as e:
        components["rag_engine"] = "not_loaded"

    # Check LLM
    try:
        from api.dependencies import get_reasoning_engine

        engine = get_reasoning_engine()
        components["llm_layer"] = "initialized"
    except Exception as e:
        components["llm_layer"] = "not_configured"

    all_ready = all(
        v in ("ready", "initialized") for v in components.values()
    )

    return {
        "status": "healthy" if all_ready else "degraded",
        "components": components,
    }


@router.get("/")
async def root():
    """API root — returns system info."""
    return {
        "system": "Financial Risk Intelligence Copilot",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }
