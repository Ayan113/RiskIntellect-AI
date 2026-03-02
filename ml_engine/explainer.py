"""
SHAP-based model explainability for fraud detection.

Provides per-transaction explanations using SHAP TreeExplainer,
identifying which features drove each prediction.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import shap

from ml_engine.feature_engineering import FeatureEngineer
from ml_engine.schemas import SHAPExplanation
from ml_engine.trainer import FraudModelTrainer
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)
config = Config()


class FraudExplainer:
    """
    SHAP explainer for the fraud detection model.

    Updated for HistGradientBoostingClassifier compatibility.
    """

    def __init__(self) -> None:
        self._trainer = FraudModelTrainer()
        self._feature_engineer = FeatureEngineer()
        self._explainer: Optional[shap.Explainer] = None
        self._is_loaded: bool = False

    def load(self) -> None:
        """Load model and initialize SHAP explainer."""
        self._trainer.load_model()
        self._feature_engineer.load()
        
        # HistGradientBoostingClassifier compatibility
        # We use a small sample for the masker
        data_sample = np.zeros((1, len(self._feature_engineer.feature_names)))
        self._explainer = shap.Explainer(
            self._trainer.model, 
            masker=shap.maskers.Independent(data=data_sample)
        )
        self._is_loaded = True
        logger.info("SHAP explainer initialized with HistGradientBoostingClassifier")

    def explain(
        self,
        features: Dict[str, float],
        transaction_id: Optional[str] = None,
        top_k: int = 5,
    ) -> SHAPExplanation:
        """
        Generate SHAP explanation for a single transaction.
        """
        if not self._is_loaded:
            self.load()

        assert self._explainer is not None

        # Transform features
        df = pd.DataFrame([features])
        df_transformed = self._feature_engineer.transform(df)

        # Compute SHAP values
        shap_values = self._explainer(df_transformed)
        
        # HistGradientBoosting binary returns values for pos class usually
        # shap_values.values shape is (1, num_features)
        sv = shap_values.values[0]
        base_value = float(shap_values.base_values[0])

        feature_names = self._feature_engineer.feature_names
        
        # Build feature contribution dict
        contributions: Dict[str, float] = {
            name: round(float(val), 6)
            for name, val in zip(feature_names, sv)
        }

        # Sort by absolute contribution
        sorted_features = sorted(
            contributions.items(), key=lambda x: abs(x[1]), reverse=True
        )

        # Top positive (pushing toward fraud)
        top_positive = [
            name for name, val in sorted_features if val > 0
        ][:top_k]

        # Top negative (pushing away from fraud)
        top_negative = [
            name for name, val in sorted_features if val < 0
        ][:top_k]

        explanation = SHAPExplanation(
            transaction_id=transaction_id,
            base_value=round(base_value, 6),
            feature_contributions=contributions,
            top_positive_features=top_positive,
            top_negative_features=top_negative,
        )

        logger.info(f"SHAP explanation generated for {transaction_id}")
        return explanation

    def get_global_importance(self, X_sample: pd.DataFrame) -> Dict[str, float]:
        """Compute global feature importance."""
        if not self._is_loaded:
            self.load()

        assert self._explainer is not None

        shap_values = self._explainer(X_sample)
        sv = shap_values.values

        mean_abs_shap = np.abs(sv).mean(axis=0)
        feature_names = self._feature_engineer.feature_names

        importance = {
            name: round(float(val), 6)
            for name, val in sorted(
                zip(feature_names, mean_abs_shap),
                key=lambda x: x[1],
                reverse=True,
            )
        }

        logger.info(f"Global importance computed over {len(X_sample)} samples")
        return importance
