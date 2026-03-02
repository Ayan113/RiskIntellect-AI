"""
Inference pipeline for fraud detection.

Loads saved model and feature pipeline, runs predictions
on new transactions, and returns structured results.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ml_engine.feature_engineering import FeatureEngineer
from ml_engine.schemas import PredictionResult
from ml_engine.trainer import FraudModelTrainer
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)
config = Config()


class FraudPredictor:
    """
    Production inference pipeline for fraud detection.

    Loads the serialized model and feature engineering pipeline,
    applies feature transforms, and returns structured predictions
    with risk tiering.
    """

    def __init__(self) -> None:
        self._trainer = FraudModelTrainer()
        self._feature_engineer = FeatureEngineer()
        self._threshold: float = config.get("ml_engine.model.threshold", 0.5)
        self._is_loaded: bool = False

    def load(self) -> None:
        """Load model and feature pipeline from disk."""
        self._trainer.load_model()
        self._feature_engineer.load()
        self._is_loaded = True
        logger.info("Fraud predictor loaded and ready for inference")

    def predict(self, features: Dict[str, float], transaction_id: Optional[str] = None) -> PredictionResult:
        """
        Score a single transaction.

        Args:
            features: Dictionary of feature name → value.
            transaction_id: Optional transaction identifier.

        Returns:
            PredictionResult with fraud probability and risk tier.
        """
        if not self._is_loaded:
            self.load()

        # Convert to DataFrame
        df = pd.DataFrame([features])

        # Apply feature engineering
        df_transformed = self._feature_engineer.transform(df)

        # Get probability
        proba = self._trainer.model.predict_proba(df_transformed)[:, 1][0]
        is_fraud = proba >= self._threshold

        result = PredictionResult(
            transaction_id=transaction_id,
            fraud_probability=round(float(proba), 6),
            is_fraud=bool(is_fraud),
            threshold=self._threshold,
            risk_tier=PredictionResult.compute_risk_tier(float(proba)),
        )

        logger.info(
            f"Prediction for {transaction_id}: prob={proba:.4f}, "
            f"fraud={is_fraud}, tier={result.risk_tier}"
        )
        return result

    def predict_batch(
        self, transactions: List[Dict[str, float]], transaction_ids: Optional[List[str]] = None
    ) -> List[PredictionResult]:
        """
        Score a batch of transactions.

        Args:
            transactions: List of feature dictionaries.
            transaction_ids: Optional list of transaction identifiers.

        Returns:
            List of PredictionResult objects.
        """
        if not self._is_loaded:
            self.load()

        if transaction_ids is None:
            transaction_ids = [None] * len(transactions)  # type: ignore

        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        df_transformed = self._feature_engineer.transform(df)

        # Batch prediction
        probas = self._trainer.model.predict_proba(df_transformed)[:, 1]

        results = []
        for i, proba in enumerate(probas):
            results.append(
                PredictionResult(
                    transaction_id=transaction_ids[i],
                    fraud_probability=round(float(proba), 6),
                    is_fraud=bool(proba >= self._threshold),
                    threshold=self._threshold,
                    risk_tier=PredictionResult.compute_risk_tier(float(proba)),
                )
            )

        high_risk = sum(1 for r in results if r.risk_tier in ("HIGH", "CRITICAL"))
        logger.info(
            f"Batch prediction: {len(results)} transactions, {high_risk} high-risk"
        )
        return results

    def get_raw_probability(self, features: Dict[str, float]) -> float:
        """
        Get raw fraud probability without structured response.

        Args:
            features: Transaction features.

        Returns:
            Fraud probability float.
        """
        if not self._is_loaded:
            self.load()

        df = pd.DataFrame([features])
        df_transformed = self._feature_engineer.transform(df)
        return float(self._trainer.model.predict_proba(df_transformed)[:, 1][0])
