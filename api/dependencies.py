"""
Dependency injection for FastAPI.

Provides singleton instances of ML model, RAG retriever, and LLM
reasoning engine via FastAPI's dependency injection system.
"""

from functools import lru_cache
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Module-level singletons
_fraud_predictor = None
_fraud_explainer = None
_hybrid_retriever = None
_reasoning_engine = None
_rag_guardrails = None


def get_fraud_predictor():
    """Get or create the fraud prediction pipeline."""
    global _fraud_predictor
    if _fraud_predictor is None:
        from ml_engine.predictor import FraudPredictor

        _fraud_predictor = FraudPredictor()
        try:
            _fraud_predictor.load()
            logger.info("Fraud predictor loaded successfully")
        except FileNotFoundError:
            logger.warning(
                "Fraud model not found. Train the model first. "
                "Predictor will attempt lazy loading on first request."
            )
    return _fraud_predictor


def get_fraud_explainer():
    """Get or create the SHAP explainer."""
    global _fraud_explainer
    if _fraud_explainer is None:
        from ml_engine.explainer import FraudExplainer

        _fraud_explainer = FraudExplainer()
        try:
            _fraud_explainer.load()
            logger.info("SHAP explainer loaded successfully")
        except FileNotFoundError:
            logger.warning(
                "Fraud model not found for SHAP. Train the model first."
            )
    return _fraud_explainer


def get_hybrid_retriever():
    """Get or create the hybrid retriever."""
    global _hybrid_retriever
    if _hybrid_retriever is None:
        from rag_engine.retriever import HybridRetriever

        _hybrid_retriever = HybridRetriever()
        try:
            _hybrid_retriever.load_indices()
            logger.info("Hybrid retriever loaded successfully")
        except FileNotFoundError:
            logger.warning(
                "RAG indices not found. Run document ingestion first."
            )
    return _hybrid_retriever


def get_reasoning_engine():
    """Get or create the LLM reasoning engine."""
    global _reasoning_engine
    if _reasoning_engine is None:
        from llm_layer.reasoning import ReasoningEngine

        _reasoning_engine = ReasoningEngine()
        logger.info("Reasoning engine initialized")
    return _reasoning_engine


def get_rag_guardrails():
    """Get or create the RAG guardrails."""
    global _rag_guardrails
    if _rag_guardrails is None:
        from rag_engine.guardrails import RAGGuardrails

        _rag_guardrails = RAGGuardrails()
        logger.info("RAG guardrails initialized")
    return _rag_guardrails
