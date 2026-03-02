"""
Pydantic schemas for the ML engine.

Defines data contracts for transaction input, prediction output,
and SHAP explanation responses.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    """Schema for a single transaction to be scored."""

    transaction_id: Optional[str] = Field(
        None, description="Unique transaction identifier"
    )
    features: Dict[str, float] = Field(
        ..., description="Dictionary of feature name → value"
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "transaction_id": "txn_001",
                "features": {
                    "V1": -1.35,
                    "V2": -0.07,
                    "V3": 2.53,
                    "Amount": 149.62,
                    "Time": 0.0,
                },
            }
        ]
    }}


class PredictionResult(BaseModel):
    """Schema for fraud prediction output."""

    transaction_id: Optional[str] = None
    fraud_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of fraud (0–1)"
    )
    is_fraud: bool = Field(
        ..., description="Binary fraud label based on threshold"
    )
    threshold: float = Field(
        ..., description="Decision threshold used"
    )
    risk_tier: str = Field(
        ..., description="Risk classification: LOW, MEDIUM, HIGH, CRITICAL"
    )

    @staticmethod
    def compute_risk_tier(probability: float) -> str:
        """Map probability to risk tier."""
        if probability < 0.3:
            return "LOW"
        elif probability < 0.6:
            return "MEDIUM"
        elif probability < 0.85:
            return "HIGH"
        else:
            return "CRITICAL"


class SHAPExplanation(BaseModel):
    """Schema for SHAP-based model explanation."""

    transaction_id: Optional[str] = None
    base_value: float = Field(
        ..., description="SHAP base value (expected model output)"
    )
    feature_contributions: Dict[str, float] = Field(
        ..., description="Feature name → SHAP value contribution"
    )
    top_positive_features: List[str] = Field(
        ..., description="Top features pushing toward fraud"
    )
    top_negative_features: List[str] = Field(
        ..., description="Top features pushing away from fraud"
    )


class BatchPredictionRequest(BaseModel):
    """Schema for batch prediction input."""

    transactions: List[TransactionInput] = Field(
        ..., min_length=1, max_length=1000,
        description="List of transactions to score"
    )


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction output."""

    results: List[PredictionResult]
    total_processed: int
    high_risk_count: int
