"""
Model training module with MLflow experiment tracking.

Trains XGBoost for fraud detection with:
- Class imbalance handling via scale_pos_weight
- Early stopping on validation AUC
- Full MLflow logging (params, metrics, model artifacts)
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)
config = Config()


class FraudModelTrainer:
    """
    HistGradientBoostingClassifier fraud detection model trainer.
    
    Switched from XGBoost to HistGradientBoostingClassifier to avoid 
    native library dependencies (libomp) on Mac/Linux environments.
    """

    def __init__(self) -> None:
        self.model: Optional[HistGradientBoostingClassifier] = None
        self.hyperparams: Dict[str, Any] = config.get(
            "ml_engine.model.hyperparameters", {}
        )
        # Adapt XGBoost hyperparams to HistGradientBoosting
        self.adapted_params = {
            "max_iter": self.hyperparams.get("n_estimators", 100),
            "max_depth": self.hyperparams.get("max_depth", 6),
            "learning_rate": self.hyperparams.get("learning_rate", 0.1),
            "l2_regularization": self.hyperparams.get("reg_lambda", 0.0),
            "random_state": config.get("ml_engine.data.random_state", 42),
        }
        
        self.mlflow_experiment: str = config.get(
            "ml_engine.mlflow.experiment_name", "fraud_detection"
        )
        self.mlflow_tracking_uri: str = config.get(
            "ml_engine.mlflow.tracking_uri", "mlruns"
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> HistGradientBoostingClassifier:
        """
        Train the HistGradientBoostingClassifier model.
        """
        # Compute class weights (HistGradientBoosting uses class_weight parameter)
        # However, it doesn't have a direct scale_pos_weight like XGBoost in the same way.
        # We'll use the 'auto' class_weight or balanced.
        
        logger.info(f"Training HistGradientBoostingClassifier on {len(X_train)} samples")

        self.model = HistGradientBoostingClassifier(
            **self.adapted_params,
            class_weight='balanced',
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )

        # Setup MLflow (optional, wrapped in try-except if not used)
        try:
            import mlflow
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            mlflow.set_experiment(self.mlflow_experiment)
            
            with mlflow.start_run(run_name="hgb_fraud_detection") as run:
                self.model.fit(X_train, y_train)
                
                val_proba = self.model.predict_proba(X_val)[:, 1]
                val_auc = roc_auc_score(y_val, val_proba)
                val_ap = average_precision_score(y_val, val_proba)
                
                mlflow.log_params(self.adapted_params)
                mlflow.log_metrics({
                    "val_roc_auc": val_auc,
                    "val_avg_precision": val_ap
                })
                
                logger.info(f"Training complete — Val ROC-AUC: {val_auc:.4f}")
        except Exception as e:
            logger.warning(f"MLflow tracking failed or not available: {e}. Proceeding with local training.")
            self.model.fit(X_train, y_train)
            
            val_proba = self.model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, val_proba)
            logger.info(f"Training complete (local) — Val ROC-AUC: {val_auc:.4f}")

        return self.model

    def save_model(self, path: str | None = None) -> Path:
        """Save the trained model to disk."""
        if self.model is None:
            raise RuntimeError("No trained model to save. Call train() first.")

        model_path = Path(
            path or config.get("ml_engine.artifacts.model_path")
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)

        logger.info(f"Model saved to {model_path}")
        return model_path

    def load_model(self, path: str | None = None) -> HistGradientBoostingClassifier:
        """Load a previously saved model."""
        model_path = Path(
            path or config.get("ml_engine.artifacts.model_path")
        )
        self.model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        return self.model
