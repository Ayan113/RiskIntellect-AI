"""
RAG evaluation metrics.

Evaluates retrieval and generation quality using:
- Faithfulness: Does the answer stay true to the retrieved context?
- Context Precision: How relevant are the retrieved documents?
- Answer Relevance: Does the answer address the query?

Uses LLM-as-judge pattern for semantic evaluation.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)
config = Config()


class RAGEvaluator:
    """
    Evaluation suite for the RAG pipeline.

    Metrics (inspired by RAGAS framework):
    1. Faithfulness: Fraction of claims in the answer grounded in context.
    2. Context Precision: Relevance of retrieved documents to the query.
    3. Answer Relevance: Semantic similarity between answer and query intent.

    Design decision: LLM-as-judge vs. automated metrics
    - BLEU/ROUGE: Fast but don't capture semantic correctness.
    - BERTScore: Better semantic matching, but still surface-level.
    - LLM-as-judge: Best semantic understanding, but expensive + requires API.
    We provide both automated and LLM-based evaluation options.
    """

    def __init__(self) -> None:
        self.output_dir = Path(
            config.get("evaluation.output_dir", "artifacts/evaluation_reports")
        )
        self._llm = None

    def _get_llm(self):
        """Lazy-load LLM for evaluation."""
        if self._llm is None:
            from llm_layer.llm_provider import LLMProvider
            self._llm = LLMProvider()
        return self._llm

    def evaluate_faithfulness(
        self,
        answer: str,
        contexts: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate faithfulness of the answer to the provided context.

        Args:
            answer: Generated answer text.
            contexts: List of retrieved context chunks.

        Returns:
            Dictionary with faithfulness score and details.
        """
        # Automated check: keyword overlap
        context_text = " ".join(contexts).lower()
        answer_words = set(answer.lower().split())
        context_words = set(context_text.split())

        overlap = len(answer_words & context_words) / max(len(answer_words), 1)

        result = {
            "metric": "faithfulness",
            "automated_score": round(min(overlap * 1.5, 1.0), 4),
            "word_overlap_ratio": round(overlap, 4),
        }

        # LLM-based evaluation (optional, more accurate)
        try:
            llm_score = self._llm_judge_faithfulness(answer, contexts)
            result["llm_score"] = llm_score
        except Exception as e:
            logger.warning(f"LLM faithfulness evaluation failed: {e}")
            result["llm_score"] = None

        return result

    def evaluate_context_precision(
        self,
        query: str,
        contexts: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate the precision/relevance of retrieved contexts.

        Args:
            query: Original query.
            contexts: Retrieved context chunks.

        Returns:
            Dictionary with context precision score.
        """
        if not contexts:
            return {"metric": "context_precision", "score": 0.0, "relevant_count": 0}

        # Automated: keyword match between query and each context
        query_words = set(query.lower().split())
        relevant_count = 0
        context_scores: List[float] = []

        for ctx in contexts:
            ctx_words = set(ctx.lower().split())
            overlap = len(query_words & ctx_words) / max(len(query_words), 1)
            context_scores.append(min(overlap * 2, 1.0))
            if overlap > 0.1:
                relevant_count += 1

        avg_precision = sum(context_scores) / len(context_scores)

        result = {
            "metric": "context_precision",
            "score": round(avg_precision, 4),
            "relevant_count": relevant_count,
            "total_contexts": len(contexts),
            "per_context_scores": [round(s, 4) for s in context_scores],
        }

        return result

    def evaluate_answer_relevance(
        self,
        query: str,
        answer: str,
    ) -> Dict[str, Any]:
        """
        Evaluate relevance of the answer to the original query.

        Args:
            query: Original user query.
            answer: Generated answer.

        Returns:
            Dictionary with answer relevance score.
        """
        # Automated: query-answer keyword overlap
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())

        if not query_words:
            return {"metric": "answer_relevance", "score": 0.0}

        overlap = len(query_words & answer_words) / len(query_words)

        result = {
            "metric": "answer_relevance",
            "automated_score": round(min(overlap * 1.5, 1.0), 4),
        }

        try:
            llm_score = self._llm_judge_relevance(query, answer)
            result["llm_score"] = llm_score
        except Exception as e:
            logger.warning(f"LLM relevance evaluation failed: {e}")
            result["llm_score"] = None

        return result

    def evaluate_full(
        self,
        query: str,
        answer: str,
        contexts: List[str],
    ) -> Dict[str, Any]:
        """
        Run all RAG evaluation metrics.

        Args:
            query: Original query.
            answer: Generated answer.
            contexts: Retrieved context chunks.

        Returns:
            Combined evaluation results.
        """
        logger.info(f"Running full RAG evaluation for query: '{query[:80]}...'")

        faithfulness = self.evaluate_faithfulness(answer, contexts)
        context_precision = self.evaluate_context_precision(query, contexts)
        answer_relevance = self.evaluate_answer_relevance(query, answer)

        # Composite score (weighted average)
        scores = [
            faithfulness.get("automated_score", 0),
            context_precision.get("score", 0),
            answer_relevance.get("automated_score", 0),
        ]
        composite = sum(scores) / len(scores)

        results = {
            "query": query,
            "faithfulness": faithfulness,
            "context_precision": context_precision,
            "answer_relevance": answer_relevance,
            "composite_score": round(composite, 4),
        }

        logger.info(f"RAG evaluation complete — composite: {composite:.4f}")
        return results

    def _llm_judge_faithfulness(
        self, answer: str, contexts: List[str]
    ) -> float:
        """Use LLM to judge faithfulness on a 0-1 scale."""
        llm = self._get_llm()
        context_text = "\n---\n".join(contexts)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evaluator. Rate the faithfulness of an answer to its "
                    "source context on a scale from 0.0 to 1.0. A score of 1.0 means "
                    "every claim in the answer is supported by the context. "
                    "Respond with ONLY a JSON object: {\"score\": <float>}"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context_text}\n\nAnswer:\n{answer}",
            },
        ]

        response = llm.generate_json(messages)
        return float(response.get("score", 0.0))

    def _llm_judge_relevance(self, query: str, answer: str) -> float:
        """Use LLM to judge answer relevance on a 0-1 scale."""
        llm = self._get_llm()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an evaluator. Rate how relevant an answer is to the "
                    "given question on a scale from 0.0 to 1.0. A score of 1.0 means "
                    "the answer perfectly addresses the question. "
                    "Respond with ONLY a JSON object: {\"score\": <float>}"
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{query}\n\nAnswer:\n{answer}",
            },
        ]

        response = llm.generate_json(messages)
        return float(response.get("score", 0.0))

    def save_report(
        self, results: List[Dict[str, Any]], filename: str = "rag_evaluation.json"
    ) -> Path:
        """Save RAG evaluation results."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / filename

        with open(report_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"RAG evaluation report saved to {report_path}")
        return report_path
