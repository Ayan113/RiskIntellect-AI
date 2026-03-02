"""
Full copilot reasoning API endpoint.

Orchestrates the complete pipeline: ML scoring → SHAP explanation →
RAG retrieval → LLM reasoning → Structured risk assessment.
"""

from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import (
    get_fraud_explainer,
    get_fraud_predictor,
    get_hybrid_retriever,
    get_rag_guardrails,
    get_reasoning_engine,
)
from utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/copilot", tags=["Copilot - Full Pipeline"])


class CopilotRequest(BaseModel):
    """Schema for the full copilot assessment request."""

    transaction_id: Optional[str] = Field(None, description="Transaction identifier")
    features: Dict[str, float] = Field(..., description="Transaction features")
    regulatory_query: Optional[str] = Field(
        None,
        description="Optional specific regulatory question about this transaction",
    )
    include_explanation: bool = Field(
        True, description="Include SHAP feature explanations"
    )
    include_regulatory_context: bool = Field(
        True, description="Include RAG regulatory context"
    )


class CopilotResponse(BaseModel):
    """Schema for the full copilot assessment response."""

    transaction_id: Optional[str]
    fraud_probability: float
    risk_tier: str
    risk_assessment: Dict
    explanation: Optional[Dict] = None
    regulatory_sources: Optional[List[Dict]] = None
    pipeline_metadata: Dict


@router.post("/assess", response_model=CopilotResponse)
async def full_risk_assessment(request: CopilotRequest):
    """
    Execute the full Financial Risk Intelligence Copilot pipeline.

    Pipeline stages:
    1. **ML Scoring**: XGBoost fraud probability + risk tier
    2. **SHAP Explanation**: Feature-level contribution analysis
    3. **RAG Retrieval**: Relevant regulatory document search
    4. **LLM Reasoning**: Grounded risk assessment with recommendations

    Returns a comprehensive risk assessment combining all components.
    """
    pipeline_metadata = {"stages_completed": []}

    try:
        # ── Stage 1: ML Fraud Scoring ──
        predictor = get_fraud_predictor()
        prediction = predictor.predict(
            features=request.features,
            transaction_id=request.transaction_id,
        )
        pipeline_metadata["stages_completed"].append("ml_scoring")

        # ── Stage 2: SHAP Explanation (optional) ──
        explanation_dict = None
        feature_contributions = None

        if request.include_explanation:
            try:
                explainer = get_fraud_explainer()
                shap_result = explainer.explain(
                    features=request.features,
                    transaction_id=request.transaction_id,
                )
                explanation_dict = {
                    "base_value": shap_result.base_value,
                    "feature_contributions": shap_result.feature_contributions,
                    "top_positive_features": shap_result.top_positive_features,
                    "top_negative_features": shap_result.top_negative_features,
                }
                feature_contributions = shap_result.feature_contributions
                pipeline_metadata["stages_completed"].append("shap_explanation")
            except Exception as e:
                logger.warning(f"SHAP explanation skipped: {e}")
                pipeline_metadata["shap_error"] = str(e)

        # ── Stage 3: RAG Retrieval (optional) ──
        regulatory_contexts = []
        regulatory_sources = None

        if request.include_regulatory_context:
            try:
                # Build a context-aware query
                query = request.regulatory_query or (
                    f"Fraud detection regulatory guidelines for "
                    f"{prediction.risk_tier} risk transactions"
                )

                # Validate query
                guardrails = get_rag_guardrails()
                is_safe, sanitized_query, _ = guardrails.validate_query(query)

                if is_safe:
                    retriever = get_hybrid_retriever()
                    results = retriever.retrieve(sanitized_query, use_reranker=True)

                    regulatory_contexts = [r.text for r in results]
                    # Filter through guardrails
                    regulatory_contexts = guardrails.validate_retrieved_context(
                        regulatory_contexts
                    )

                    regulatory_sources = [
                        {
                            "text": r.text[:300] + "..." if len(r.text) > 300 else r.text,
                            "source": r.metadata.get("source", "unknown"),
                            "score": r.score,
                        }
                        for r in results
                    ]
                    pipeline_metadata["stages_completed"].append("rag_retrieval")
                    pipeline_metadata["contexts_retrieved"] = len(regulatory_contexts)

            except Exception as e:
                logger.warning(f"RAG retrieval skipped: {e}")
                pipeline_metadata["rag_error"] = str(e)

        # ── Stage 4: LLM Reasoning ──
        try:
            engine = get_reasoning_engine()
            assessment = engine.assess_transaction(
                transaction_details=request.features,
                fraud_probability=prediction.fraud_probability,
                risk_tier=prediction.risk_tier,
                feature_explanations=feature_contributions,
                regulatory_context=regulatory_contexts if regulatory_contexts else None,
                query=request.regulatory_query,
            )
            risk_assessment_dict = assessment.to_dict()
            pipeline_metadata["stages_completed"].append("llm_reasoning")

        except Exception as e:
            logger.error(f"LLM reasoning failed: {e}", exc_info=True)
            # Fallback: return ML-only assessment
            risk_assessment_dict = {
                "risk_level": prediction.risk_tier,
                "confidence": prediction.fraud_probability,
                "explanation": f"ML model scored this transaction at {prediction.fraud_probability:.4f} fraud probability.",
                "regulatory_basis": "LLM reasoning unavailable — showing ML-only assessment.",
                "recommended_action": (
                    "Manual review recommended"
                    if prediction.risk_tier in ("HIGH", "CRITICAL")
                    else "Standard processing"
                ),
            }
            pipeline_metadata["llm_error"] = str(e)
            pipeline_metadata["stages_completed"].append("ml_fallback")

        return CopilotResponse(
            transaction_id=request.transaction_id,
            fraud_probability=prediction.fraud_probability,
            risk_tier=prediction.risk_tier,
            risk_assessment=risk_assessment_dict,
            explanation=explanation_dict,
            regulatory_sources=regulatory_sources,
            pipeline_metadata=pipeline_metadata,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Copilot pipeline failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}",
        )
