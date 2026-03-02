"""
Data loading and splitting for the fraud detection pipeline.

Handles loading the credit card fraud dataset with stratified
train/validation/test splits to preserve class distribution.
"""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)
config = Config()


class DataLoader:
    """
    Loads and splits the credit card fraud dataset.

    Expected format: CSV with features V1–V28, Time, Amount, and Class (0/1).
    Implements stratified splitting to handle the severe class imbalance
    (~0.17% fraud rate in the standard Kaggle dataset).
    """

    def __init__(self, data_path: str | None = None) -> None:
        """
        Initialize the data loader.

        Args:
            data_path: Path to the CSV file. Defaults to config value.
        """
        self.data_path = Path(
            data_path or config.get("ml_engine.data.raw_path")
        )
        self.test_size: float = config.get("ml_engine.data.test_size", 0.15)
        self.val_size: float = config.get("ml_engine.data.val_size", 0.15)
        self.random_state: int = config.get("ml_engine.data.random_state", 42)
        self._df: pd.DataFrame | None = None

    def load(self) -> pd.DataFrame:
        """
        Load the raw dataset from CSV.

        Returns:
            DataFrame with raw features and target.

        Raises:
            FileNotFoundError: If the data file doesn't exist.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Download the Kaggle credit card fraud dataset and place it in data/raw/"
            )

        logger.info(f"Loading dataset from {self.data_path}")
        self._df = pd.read_csv(self.data_path)

        logger.info(
            f"Dataset loaded: {self._df.shape[0]} rows, {self._df.shape[1]} columns"
        )
        logger.info(
            f"Class distribution:\n{self._df['Class'].value_counts().to_dict()}"
        )
        return self._df

    def get_splits(
        self,
    ) -> Tuple[
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
    ]:
        """
        Create stratified train/validation/test splits.

        Strategy:
            1. Split off test set (15%) from full data.
            2. Split remaining into train (70%) and validation (15%).
            Stratification ensures fraud ratio is preserved in all splits.

        Returns:
            Tuple of ((X_train, y_train), (X_val, y_val), (X_test, y_test)).
        """
        if self._df is None:
            self.load()

        assert self._df is not None
        X = self._df.drop(columns=["Class"])
        y = self._df["Class"]

        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        # Second split: separate validation from training
        val_ratio = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=self.random_state,
            stratify=y_temp,
        )

        logger.info(
            f"Split sizes — Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )
        self._log_split_stats("Train", y_train)
        self._log_split_stats("Val", y_val)
        self._log_split_stats("Test", y_test)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    @staticmethod
    def _log_split_stats(name: str, y: pd.Series) -> None:
        """Log class distribution for a split."""
        fraud_count = int(y.sum())
        total = len(y)
        fraud_pct = (fraud_count / total) * 100 if total > 0 else 0
        logger.info(
            f"  {name}: {total} samples, {fraud_count} fraud ({fraud_pct:.3f}%)"
        )

    def get_dataset_stats(self) -> Dict:
        """
        Compute summary statistics for the loaded dataset.

        Returns:
            Dictionary of dataset statistics.
        """
        if self._df is None:
            self.load()

        assert self._df is not None
        return {
            "num_rows": int(self._df.shape[0]),
            "num_features": int(self._df.shape[1] - 1),
            "fraud_count": int(self._df["Class"].sum()),
            "legitimate_count": int((self._df["Class"] == 0).sum()),
            "fraud_rate": float(self._df["Class"].mean()),
            "feature_names": list(self._df.drop(columns=["Class"]).columns),
            "missing_values": int(self._df.isnull().sum().sum()),
        }
