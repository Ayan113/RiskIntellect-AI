"""
Feature engineering pipeline for fraud detection.

Transforms raw transaction data into model-ready features.
Includes log scaling, time decomposition, interaction features,
and robust scaling for outlier-heavy financial data.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import joblib

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)
config = Config()


class FeatureEngineer:
    """
    Feature engineering pipeline for credit card fraud detection.

    Design decisions:
    - RobustScaler over StandardScaler: Financial data has extreme outliers.
      RobustScaler uses IQR, making it resistant to fraudulent outlier amounts.
    - Log transform on Amount: Transaction amounts are right-skewed.
      Log1p compresses the range while preserving relative ordering.
    - Time decomposition: Raw 'Time' (seconds from first txn) is converted
      to hour-of-day, capturing temporal fraud patterns (e.g. late-night fraud).
    - Interaction features: V1*V2, V3*V4 etc. capture nonlinear relationships
      that tree models can exploit.
    """

    def __init__(self) -> None:
        self.scaler: Optional[RobustScaler] = None
        self.feature_names: List[str] = []
        self._is_fitted: bool = False

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the feature pipeline on training data and transform it.

        Args:
            X: Raw training features DataFrame.

        Returns:
            Transformed features DataFrame.
        """
        logger.info(f"Fitting feature engineering pipeline on {X.shape[0]} samples")
        X_transformed = self._create_features(X)

        # Fit and transform the scaler on numeric columns
        self.scaler = RobustScaler()
        cols_to_scale = [c for c in X_transformed.columns if c not in ("hour_of_day",)]
        X_transformed[cols_to_scale] = self.scaler.fit_transform(
            X_transformed[cols_to_scale]
        )

        self.feature_names = list(X_transformed.columns)
        self._is_fitted = True

        logger.info(f"Feature engineering complete: {len(self.feature_names)} features")
        return X_transformed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform new data using the fitted pipeline.

        Args:
            X: Raw features DataFrame.

        Returns:
            Transformed features DataFrame.

        Raises:
            RuntimeError: If pipeline hasn't been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "Feature pipeline not fitted. Call fit_transform() first."
            )

        X_transformed = self._create_features(X)

        assert self.scaler is not None
        cols_to_scale = [c for c in X_transformed.columns if c not in ("hour_of_day",)]
        X_transformed[cols_to_scale] = self.scaler.transform(
            X_transformed[cols_to_scale]
        )

        # Ensure column order matches training
        X_transformed = X_transformed[self.feature_names]
        return X_transformed

    def _create_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features from raw data.

        Args:
            X: Raw features DataFrame.

        Returns:
            DataFrame with engineered features.
        """
        df = X.copy()

        # --- Log-transform Amount ---
        if "Amount" in df.columns:
            df["log_amount"] = np.log1p(df["Amount"])
            df.drop(columns=["Amount"], inplace=True)

        # --- Time decomposition ---
        if "Time" in df.columns:
            # Convert seconds to hour of day (cyclical within 24h)
            df["hour_of_day"] = (df["Time"] / 3600) % 24
            # Sine/cosine encoding for cyclical nature
            df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
            df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
            df.drop(columns=["Time"], inplace=True)

        # --- Interaction features ---
        # Capture pairwise interactions among top PCA components
        v_cols = [c for c in df.columns if c.startswith("V")]
        if len(v_cols) >= 4:
            df["V1_V2_interaction"] = df["V1"] * df["V2"]
            df["V3_V4_interaction"] = df["V3"] * df["V4"]
            df["V5_V6_interaction"] = df["V5"] * df["V6"]

        # --- Statistical aggregates across V-features ---
        if v_cols:
            df["V_mean"] = df[v_cols].mean(axis=1)
            df["V_std"] = df[v_cols].std(axis=1)
            df["V_skew"] = df[v_cols].skew(axis=1)

        # --- Magnitude features ---
        if "V1" in df.columns and "V2" in df.columns:
            df["V1_V2_magnitude"] = np.sqrt(df["V1"] ** 2 + df["V2"] ** 2)

        # Handle any NaN/Inf
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        return df

    def save(self, scaler_path: str | None = None, features_path: str | None = None) -> None:
        """
        Save the fitted scaler and feature names.

        Args:
            scaler_path: Path to save scaler. Defaults to config.
            features_path: Path to save feature names. Defaults to config.
        """
        if not self._is_fitted:
            raise RuntimeError("Cannot save unfitted pipeline.")

        scaler_path = Path(
            scaler_path or config.get("ml_engine.artifacts.scaler_path")
        )
        features_path = Path(
            features_path or config.get("ml_engine.artifacts.feature_names_path")
        )

        # Ensure directories exist
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        features_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.scaler, scaler_path)
        with open(features_path, "w") as f:
            json.dump(self.feature_names, f)

        logger.info(f"Scaler saved to {scaler_path}")
        logger.info(f"Feature names saved to {features_path}")

    def load(self, scaler_path: str | None = None, features_path: str | None = None) -> None:
        """
        Load a previously fitted scaler and feature names.

        Args:
            scaler_path: Path to load scaler from. Defaults to config.
            features_path: Path to load feature names from. Defaults to config.
        """
        scaler_path = Path(
            scaler_path or config.get("ml_engine.artifacts.scaler_path")
        )
        features_path = Path(
            features_path or config.get("ml_engine.artifacts.feature_names_path")
        )

        self.scaler = joblib.load(scaler_path)
        with open(features_path, "r") as f:
            self.feature_names = json.load(f)

        self._is_fitted = True
        logger.info(f"Feature pipeline loaded: {len(self.feature_names)} features")
