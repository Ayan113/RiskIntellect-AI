"""
Prompt template construction for the LLM reasoning layer.

Builds structured prompts that combine transaction details,
fraud detection scores, and regulatory context for grounded
risk assessment.
"""

from typing import Dict, List, Optional

from utils.logger import get_logger

logger = get_logger(__name__)


SYSTEM_PROMPT = """You are a Financial Risk Intelligence Analyst specializing in fraud detection and regulatory compliance.

## Your Role
You assess transactions for fraud risk by analyzing:
1. Machine learning model fraud probability scores
2. Transaction feature explanations (SHAP values)
3. Relevant regulatory documents and guidelines

## Rules
- ONLY cite information present in the provided regulatory context.
- If no relevant regulation is found, state "No applicable regulation identified in the provided context."
- Never fabricate regulation numbers, circular references, or compliance guidelines.
- Provide actionable, specific recommendations.
- Be precise about confidence levels — do not overstate certainty.
- Always respond in the specified JSON format.

## Output Format
Respond ONLY with valid JSON in this exact structure:
{
    "risk_level": "LOW | MEDIUM | HIGH | CRITICAL",
    "confidence": "A float between 0.0 and 1.0",
    "explanation": "Detailed explanation of the risk assessment",
    "regulatory_basis": "Specific regulatory references from the provided context, or 'No applicable regulation identified'",
    "recommended_action": "Specific action items for the compliance team"
}"""


def build_assessment_prompt(
    transaction_details: Dict[str, float],
    fraud_probability: float,
    risk_tier: str,
    feature_explanations: Optional[Dict[str, float]] = None,
    regulatory_context: Optional[List[str]] = None,
    query: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Build the complete prompt for LLM risk assessment.

    Args:
        transaction_details: Key transaction features.
        fraud_probability: ML model fraud probability score.
        risk_tier: Risk tier from the ML model (LOW/MEDIUM/HIGH/CRITICAL).
        feature_explanations: SHAP feature contributions (optional).
        regulatory_context: Retrieved regulatory document chunks (optional).
        query: Specific analyst query about the transaction (optional).

    Returns:
        List of message dicts for the chat completion API.
    """
    # Build the user message
    user_parts: List[str] = []

    # Section 1: Transaction details
    user_parts.append("## Transaction Details")
    for key, value in transaction_details.items():
        user_parts.append(f"- **{key}**: {value}")

    # Section 2: ML model output
    user_parts.append(f"\n## ML Fraud Detection Output")
    user_parts.append(f"- **Fraud Probability**: {fraud_probability:.4f}")
    user_parts.append(f"- **Risk Tier**: {risk_tier}")

    # Section 3: SHAP explanations
    if feature_explanations:
        user_parts.append(f"\n## Key Feature Contributions (SHAP)")
        # Sort by absolute contribution
        sorted_feats = sorted(
            feature_explanations.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:10]
        for feat_name, shap_val in sorted_feats:
            direction = "↑ fraud" if shap_val > 0 else "↓ legitimate"
            user_parts.append(f"- **{feat_name}**: {shap_val:+.4f} ({direction})")

    # Section 4: Regulatory context
    if regulatory_context:
        user_parts.append(f"\n## Relevant Regulatory Context")
        for i, context in enumerate(regulatory_context, 1):
            user_parts.append(f"\n### Source {i}")
            user_parts.append(context)
    else:
        user_parts.append(
            "\n## Regulatory Context\nNo regulatory documents were retrieved for this query."
        )

    # Section 5: Analyst query
    if query:
        user_parts.append(f"\n## Analyst Query\n{query}")
    else:
        user_parts.append(
            "\n## Task\nProvide a comprehensive risk assessment for this transaction."
        )

    user_message = "\n".join(user_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    logger.info(
        f"Prompt built: {len(user_message)} chars, "
        f"{len(regulatory_context or [])} regulatory contexts"
    )
    return messages


def build_rag_query_prompt(
    query: str,
    contexts: List[str],
) -> List[Dict[str, str]]:
    """
    Build a prompt for RAG-based regulatory question answering.

    Args:
        query: User's regulatory question.
        contexts: Retrieved document chunks.

    Returns:
        List of message dicts for chat completion API.
    """
    system = (
        "You are a regulatory compliance expert. Answer questions ONLY based on "
        "the provided regulatory document context. If the context does not contain "
        "enough information, say so explicitly. Never fabricate regulations or citations."
    )

    user_parts = ["## Regulatory Context"]
    for i, ctx in enumerate(contexts, 1):
        user_parts.append(f"\n### Document {i}\n{ctx}")
    user_parts.append(f"\n## Question\n{query}")

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(user_parts)},
    ]
