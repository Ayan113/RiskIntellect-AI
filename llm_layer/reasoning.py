"""
LLM reasoning orchestrator for the Financial Risk Intelligence Copilot.

Combines ML predictions, SHAP explanations, and regulatory context
to produce grounded, structured risk assessments.
"""

from typing import Any, Dict, List, Optional

from llm_layer.guardrails import HallucinationGuard
from llm_layer.llm_provider import LLMProvider
from llm_layer.prompt_builder import build_assessment_prompt, build_rag_query_prompt
from utils.logger import get_logger

logger = get_logger(__name__)


class RiskAssessment:
    """Structured risk assessment output."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.risk_level: str = data.get("risk_level", "UNKNOWN")
        self.confidence: float = float(data.get("confidence", 0.0))
        self.explanation: str = data.get("explanation", "")
        self.regulatory_basis: str = data.get("regulatory_basis", "")
        self.recommended_action: str = data.get("recommended_action", "")
        self._raw: Dict[str, Any] = data

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "regulatory_basis": self.regulatory_basis,
            "recommended_action": self.recommended_action,
        }

    def __repr__(self) -> str:
        return f"RiskAssessment(level={self.risk_level}, confidence={self.confidence})"


class ReasoningEngine:
    """
    Orchestrates the full copilot reasoning pipeline.

    Pipeline:
    1. Receives transaction data + ML score + SHAP explanation + RAG context
    2. Builds structured prompt with grounding constraints
    3. Sends to LLM for assessment
    4. Validates response against hallucination guardrails
    5. Returns structured RiskAssessment

    Design decision: Separate reasoning from retrieval
    - Reasoning engine is agnostic to HOW context was retrieved.
    - It only cares about the structured inputs it receives.
    - This allows swapping retrieval strategies without touching reasoning.
    """

    def __init__(self) -> None:
        self.llm = LLMProvider()
        self.hallucination_guard = HallucinationGuard()

    def assess_transaction(
        self,
        transaction_details: Dict[str, float],
        fraud_probability: float,
        risk_tier: str,
        feature_explanations: Optional[Dict[str, float]] = None,
        regulatory_context: Optional[List[str]] = None,
        query: Optional[str] = None,
    ) -> RiskAssessment:
        """
        Perform full risk assessment on a transaction.

        Args:
            transaction_details: Key transaction features.
            fraud_probability: ML fraud probability.
            risk_tier: ML risk tier.
            feature_explanations: SHAP contributions.
            regulatory_context: Retrieved regulatory chunks.
            query: Optional analyst question.

        Returns:
            RiskAssessment with grounded analysis.
        """
        logger.info(
            f"Assessing transaction: prob={fraud_probability:.4f}, tier={risk_tier}"
        )

        # Build prompt
        messages = build_assessment_prompt(
            transaction_details=transaction_details,
            fraud_probability=fraud_probability,
            risk_tier=risk_tier,
            feature_explanations=feature_explanations,
            regulatory_context=regulatory_context,
            query=query,
        )

        # Generate LLM response
        response = self.llm.generate_json(messages)

        # Validate against hallucination guardrails
        if regulatory_context:
            is_grounded, issues = self.hallucination_guard.validate_response(
                response=response,
                provided_contexts=regulatory_context,
            )
            if not is_grounded:
                logger.warning(f"Hallucination detected: {issues}")
                response["_guardrail_warnings"] = issues
                # Append warning to explanation
                response["explanation"] = (
                    response.get("explanation", "")
                    + f"\n\n⚠️ Guardrail Warning: {'; '.join(issues)}"
                )

        assessment = RiskAssessment(response)
        logger.info(f"Assessment complete: {assessment}")
        return assessment

    def answer_regulatory_query(
        self,
        query: str,
        contexts: List[str],
    ) -> Dict[str, Any]:
        """
        Answer a regulatory question using retrieved context.

        Args:
            query: User's question about regulations.
            contexts: Retrieved regulatory document chunks.

        Returns:
            Dictionary with answer and metadata.
        """
        messages = build_rag_query_prompt(query, contexts)
        response = self.llm.generate(messages, json_mode=False)

        # Validate grounding
        is_grounded, issues = self.hallucination_guard.validate_text_response(
            response=response,
            provided_contexts=contexts,
        )

        return {
            "answer": response,
            "grounded": is_grounded,
            "guardrail_warnings": issues,
            "num_contexts_used": len(contexts),
        }
