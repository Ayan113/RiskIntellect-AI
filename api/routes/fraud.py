"""
Fraud detection API endpoints.

Provides transaction scoring, batch prediction,
and SHAP-based model explanation endpoints.
"""

from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from api.dependencies import get_fraud_explainer, get_fraud_predictor
from ml_engine.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    PredictionResult,
    SHAPExplanation,
    TransactionInput,
)
from utils.logger import get_logger
from utils.security import validate_transaction_input

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1/fraud", tags=["Fraud Detection"])


@router.post("/predict", response_model=PredictionResult)
async def predict_fraud(transaction: TransactionInput):
    """
    Score a single transaction for fraud probability.

    Accepts transaction features and returns:
    - Fraud probability (0–1)
    - Binary fraud label
    - Risk tier (LOW/MEDIUM/HIGH/CRITICAL)
    """
    # Validate input
    is_valid, error = validate_transaction_input(transaction.features)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    try:
        predictor = get_fraud_predictor()
        result = predictor.predict(
            features=transaction.features,
            transaction_id=transaction.transaction_id,
        )
        return result
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Fraud model not loaded. Train the model first.",
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_fraud_batch(request: BatchPredictionRequest):
    """
    Score a batch of transactions for fraud probability.

    Accepts up to 1000 transactions per request.
    """
    try:
        predictor = get_fraud_predictor()
        features_list = [t.features for t in request.transactions]
        ids_list = [t.transaction_id for t in request.transactions]

        results = predictor.predict_batch(features_list, ids_list)

        high_risk_count = sum(
            1 for r in results if r.risk_tier in ("HIGH", "CRITICAL")
        )

        return BatchPredictionResponse(
            results=results,
            total_processed=len(results),
            high_risk_count=high_risk_count,
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Fraud model not loaded. Train the model first.",
        )
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/explain", response_model=SHAPExplanation)
async def explain_prediction(transaction: TransactionInput):
    """
    Explain a fraud prediction using SHAP values.

    Returns per-feature contributions showing which features
    drove the model's prediction toward or away from fraud.
    """
    is_valid, error = validate_transaction_input(transaction.features)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)

    try:
        explainer = get_fraud_explainer()
        explanation = explainer.explain(
            features=transaction.features,
            transaction_id=transaction.transaction_id,
        )
        return explanation
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded for explanation. Train the model first.",
        )
    except Exception as e:
        logger.error(f"Explanation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
