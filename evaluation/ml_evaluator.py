"""
ML model evaluation suite.

Comprehensive evaluation of the fraud detection model with
multiple metrics, threshold analysis, and visual report generation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)
config = Config()


class MLEvaluator:
    """
    Evaluation suite for the fraud detection model.

    Computes:
    - ROC-AUC: Overall discriminative ability
    - Average Precision (PR-AUC): Critical for imbalanced data
    - Precision/Recall at multiple thresholds
    - Confusion matrix
    - Threshold optimization for business objectives
    - Classification report

    Design decision: PR-AUC over ROC-AUC as primary metric
    - ROC-AUC can be misleadingly high (>0.99) on heavily imbalanced data.
    - PR-AUC focuses on the minority class (fraud), which is what we care about.
    - We report both, but optimize decisions based on PR-AUC.
    """

    def __init__(self) -> None:
        self.output_dir = Path(
            config.get("evaluation.output_dir", "artifacts/evaluation_reports")
        )
        self.thresholds: List[float] = config.get(
            "evaluation.ml.thresholds", [0.3, 0.5, 0.7, 0.9]
        )

    def evaluate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        split_name: str = "test",
    ) -> Dict[str, Any]:
        """
        Run full evaluation suite.

        Args:
            y_true: Ground truth labels (0/1).
            y_proba: Predicted probabilities for the positive class.
            split_name: Name of the data split (train/val/test).

        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info(f"Evaluating on {split_name} set ({len(y_true)} samples)")

        results: Dict[str, Any] = {
            "split": split_name,
            "num_samples": int(len(y_true)),
            "num_positive": int(np.sum(y_true)),
            "num_negative": int(np.sum(y_true == 0)),
        }

        # ROC-AUC
        results["roc_auc"] = float(roc_auc_score(y_true, y_proba))

        # Average Precision (PR-AUC)
        results["average_precision"] = float(average_precision_score(y_true, y_proba))

        # Threshold analysis
        results["threshold_analysis"] = self._threshold_analysis(y_true, y_proba)

        # Best threshold by F1
        best_threshold, best_metrics = self._find_optimal_threshold(y_true, y_proba)
        results["optimal_threshold"] = best_threshold
        results["optimal_threshold_metrics"] = best_metrics

        # Default threshold (0.5) metrics
        y_pred_default = (y_proba >= 0.5).astype(int)
        results["default_threshold"] = {
            "threshold": 0.5,
            "precision": float(precision_score(y_true, y_pred_default, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred_default, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred_default, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred_default)),
        }

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_default)
        results["confusion_matrix"] = {
            "true_negative": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_positive": int(cm[1][1]),
        }

        # Classification report
        results["classification_report"] = classification_report(
            y_true, y_pred_default, output_dict=True, zero_division=0
        )

        # ROC curve data points (for plotting)
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        results["roc_curve"] = {
            "fpr": [round(float(x), 4) for x in fpr[::max(1, len(fpr) // 100)]],
            "tpr": [round(float(x), 4) for x in tpr[::max(1, len(tpr) // 100)]],
        }

        # PR curve data points
        precision_vals, recall_vals, pr_thresholds = precision_recall_curve(y_true, y_proba)
        results["pr_curve"] = {
            "precision": [round(float(x), 4) for x in precision_vals[::max(1, len(precision_vals) // 100)]],
            "recall": [round(float(x), 4) for x in recall_vals[::max(1, len(recall_vals) // 100)]],
        }

        logger.info(
            f"Evaluation complete — ROC-AUC: {results['roc_auc']:.4f}, "
            f"Avg Precision: {results['average_precision']:.4f}, "
            f"Optimal threshold: {best_threshold:.3f}"
        )

        return results

    def _threshold_analysis(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> List[Dict[str, float]]:
        """Evaluate metrics at multiple thresholds."""
        analysis: List[Dict[str, float]] = []
        for threshold in self.thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            analysis.append({
                "threshold": threshold,
                "precision": float(precision_score(y_true, y_pred, zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "flagged_rate": float(np.mean(y_pred)),
            })
        return analysis

    def _find_optimal_threshold(
        self, y_true: np.ndarray, y_proba: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """Find the threshold that maximizes F1 score."""
        precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, y_proba)

        f1_scores = np.where(
            (precision_vals[:-1] + recall_vals[:-1]) > 0,
            2 * precision_vals[:-1] * recall_vals[:-1] / (precision_vals[:-1] + recall_vals[:-1]),
            0,
        )

        best_idx = np.argmax(f1_scores)
        best_threshold = float(thresholds[best_idx])

        return best_threshold, {
            "precision": float(precision_vals[best_idx]),
            "recall": float(recall_vals[best_idx]),
            "f1": float(f1_scores[best_idx]),
        }

    def save_report(self, results: Dict[str, Any], filename: str = "ml_evaluation.json") -> Path:
        """Save evaluation results to JSON file."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / filename

        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"ML evaluation report saved to {report_path}")
        return report_path
