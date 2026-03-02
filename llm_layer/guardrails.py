"""
Hallucination detection and grounding guardrails for the LLM layer.

Verifies that LLM responses are grounded in the provided context
and flags unsubstantiated claims.
"""

import re
from typing import Any, Dict, List, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)


class HallucinationGuard:
    """
    Validates LLM outputs for hallucination and groundedness.

    Strategies:
    1. Citation verification: Check that regulatory references in the
       response exist in the provided context.
    2. Claim extraction: Identify factual claims and verify against context.
    3. Confidence calibration: Flag overly confident statements about
       regulations not found in context.

    Design tradeoff: Rule-based vs. LLM-based hallucination detection
    - Rule-based: Fast, deterministic, no additional API cost.
      Catches obvious fabrications (fake regulation numbers).
    - LLM-based (e.g. GPT-4 as judge): Better semantic understanding,
      catches paraphrased hallucinations. Adds latency + cost.
    We use rule-based for speed. For production, add LLM-based as a
    secondary check for HIGH/CRITICAL risk assessments.
    """

    # Patterns that look like regulatory citations
    CITATION_PATTERNS = [
        re.compile(r"RBI/\d{4}[-–]\d{2,4}/\d+", re.I),
        re.compile(r"circular\s+(?:no\.?\s*)?[\w\-/]+", re.I),
        re.compile(r"section\s+\d+[\w]*\s+of\s+(?:the\s+)?[\w\s]+act", re.I),
        re.compile(r"regulation\s+\d+", re.I),
        re.compile(r"master\s+direction\s+[\w\-/]+", re.I),
        re.compile(r"PMLA", re.I),
        re.compile(r"FEMA", re.I),
        re.compile(r"KYC", re.I),
        re.compile(r"AML", re.I),
    ]

    def validate_response(
        self,
        response: Dict[str, Any],
        provided_contexts: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Validate a structured LLM response against provided context.

        Args:
            response: Parsed JSON response from LLM.
            provided_contexts: List of regulatory text chunks provided to LLM.

        Returns:
            Tuple of (is_grounded, list_of_issues).
        """
        issues: List[str] = []
        combined_context = " ".join(provided_contexts).lower()

        # Check 1: Validate regulatory_basis field
        reg_basis = response.get("regulatory_basis", "")
        if reg_basis and "no applicable regulation" not in reg_basis.lower():
            citations = self._extract_citations(reg_basis)
            for citation in citations:
                if citation.lower() not in combined_context:
                    issues.append(
                        f"Citation '{citation}' not found in provided context"
                    )

        # Check 2: Validate explanation doesn't contain ungrounded claims
        explanation = response.get("explanation", "")
        explanation_citations = self._extract_citations(explanation)
        for citation in explanation_citations:
            if citation.lower() not in combined_context:
                issues.append(
                    f"Explanation references '{citation}' not in context"
                )

        # Check 3: Confidence sanity check
        confidence = float(response.get("confidence", 0))
        if confidence > 0.9 and not provided_contexts:
            issues.append(
                "High confidence (>0.9) without regulatory context is suspicious"
            )

        is_grounded = len(issues) == 0
        if not is_grounded:
            logger.warning(f"Grounding validation failed: {issues}")

        return is_grounded, issues

    def validate_text_response(
        self,
        response: str,
        provided_contexts: List[str],
    ) -> Tuple[bool, List[str]]:
        """
        Validate a free-text LLM response against provided context.

        Args:
            response: Raw text response from LLM.
            provided_contexts: List of context chunks.

        Returns:
            Tuple of (is_grounded, list_of_issues).
        """
        issues: List[str] = []
        combined_context = " ".join(provided_contexts).lower()

        # Check citations in the text
        citations = self._extract_citations(response)
        for citation in citations:
            if citation.lower() not in combined_context:
                issues.append(f"Ungrounded citation: '{citation}'")

        # Flag if response is much longer than context (possible elaboration/hallucination)
        if len(response) > len(combined_context) * 3 and len(combined_context) > 100:
            issues.append(
                "Response significantly longer than provided context — "
                "may contain elaborated content"
            )

        return len(issues) == 0, issues

    def _extract_citations(self, text: str) -> List[str]:
        """
        Extract regulatory citation-like patterns from text.

        Args:
            text: Text to scan for citations.

        Returns:
            List of extracted citation strings.
        """
        citations: List[str] = []
        for pattern in self.CITATION_PATTERNS:
            matches = pattern.findall(text)
            citations.extend(matches)
        return list(set(citations))
