"""
Evaluation report generator.

Aggregates ML and RAG evaluation results into structured
reports with logging and persistence.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)
config = Config()


class EvaluationReport:
    """
    Aggregates and persists evaluation results across all components.

    Generates a single comprehensive report combining:
    - ML model evaluation metrics
    - RAG pipeline evaluation metrics
    - Adversarial test results
    - System metadata (timestamps, versions, configs)
    """

    def __init__(self) -> None:
        self.output_dir = Path(
            config.get("evaluation.output_dir", "artifacts/evaluation_reports")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        ml_results: Optional[Dict[str, Any]] = None,
        rag_results: Optional[List[Dict[str, Any]]] = None,
        adversarial_results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.

        Args:
            ml_results: ML evaluator output.
            rag_results: List of RAG evaluator outputs.
            adversarial_results: Adversarial test suite output.

        Returns:
            Full evaluation report dictionary.
        """
        report: Dict[str, Any] = {
            "report_metadata": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "report_version": "1.0.0",
                "system": config.get("app.name", "Financial Risk Intelligence Copilot"),
            },
        }

        # ML Evaluation Summary
        if ml_results:
            report["ml_evaluation"] = {
                "roc_auc": ml_results.get("roc_auc"),
                "average_precision": ml_results.get("average_precision"),
                "optimal_threshold": ml_results.get("optimal_threshold"),
                "confusion_matrix": ml_results.get("confusion_matrix"),
                "threshold_analysis": ml_results.get("threshold_analysis"),
            }

        # RAG Evaluation Summary
        if rag_results:
            avg_composite = sum(
                r.get("composite_score", 0) for r in rag_results
            ) / max(len(rag_results), 1)

            avg_faithfulness = sum(
                r.get("faithfulness", {}).get("automated_score", 0)
                for r in rag_results
            ) / max(len(rag_results), 1)

            report["rag_evaluation"] = {
                "num_queries_evaluated": len(rag_results),
                "avg_composite_score": round(avg_composite, 4),
                "avg_faithfulness": round(avg_faithfulness, 4),
                "individual_results": rag_results,
            }

        # Adversarial Test Summary
        if adversarial_results:
            report["adversarial_testing"] = adversarial_results

        # Overall system health assessment
        report["system_health"] = self._compute_health_score(report)

        logger.info("Full evaluation report generated")
        return report

    @staticmethod
    def _compute_health_score(report: Dict[str, Any]) -> Dict[str, Any]:
        """Compute an overall system health score from component metrics."""
        scores: List[float] = []
        issues: List[str] = []

        # ML health
        ml_eval = report.get("ml_evaluation", {})
        if ml_eval:
            roc_auc = ml_eval.get("roc_auc", 0)
            scores.append(roc_auc)
            if roc_auc < 0.9:
                issues.append(f"ML ROC-AUC below 0.9 ({roc_auc:.4f})")

        # RAG health
        rag_eval = report.get("rag_evaluation", {})
        if rag_eval:
            composite = rag_eval.get("avg_composite_score", 0)
            scores.append(composite)
            if composite < 0.6:
                issues.append(f"RAG composite score below 0.6 ({composite:.4f})")

        # Adversarial health
        adv = report.get("adversarial_testing", {})
        if adv:
            pass_rate = adv.get("pass_rate", 0)
            scores.append(pass_rate)
            if pass_rate < 0.8:
                issues.append(f"Adversarial pass rate below 80% ({pass_rate:.2%})")

        overall = sum(scores) / max(len(scores), 1)

        return {
            "overall_score": round(overall, 4),
            "status": "HEALTHY" if overall >= 0.8 else "DEGRADED" if overall >= 0.6 else "UNHEALTHY",
            "component_scores": scores,
            "issues": issues,
        }

    def save_report(
        self, report: Dict[str, Any], filename: str = "full_evaluation_report.json"
    ) -> Path:
        """Save the full evaluation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{timestamp}_{filename}"
        report_path = self.output_dir / report_filename

        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"Full evaluation report saved to {report_path}")
        return report_path
