"""
Security and input sanitization utilities.

Implements prompt injection detection and input validation
for both API inputs and RAG query pipelines.
"""

import re
from typing import List, Tuple

from utils.logger import get_logger

logger = get_logger(__name__)

# Known prompt injection patterns
INJECTION_PATTERNS: List[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?(previous|above|prior)\s+(instructions|prompts)", re.I),
    re.compile(r"disregard\s+(all\s+)?(previous|above|prior)", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an|the)", re.I),
    re.compile(r"pretend\s+(you\s+are|to\s+be)", re.I),
    re.compile(r"act\s+as\s+(a|an|if)", re.I),
    re.compile(r"system\s*:\s*", re.I),
    re.compile(r"<\s*/?\s*(system|prompt|instruction)", re.I),
    re.compile(r"\[INST\]|\[/INST\]|\[SYSTEM\]", re.I),
    re.compile(r"forget\s+(everything|all|your)", re.I),
    re.compile(r"new\s+instructions?\s*:", re.I),
    re.compile(r"override\s+(your|the|all)", re.I),
    re.compile(r"jailbreak", re.I),
    re.compile(r"do\s+anything\s+now", re.I),
    re.compile(r"DAN\s+mode", re.I),
]

# Characters that should be escaped in user inputs
DANGEROUS_CHARS = ["{{", "}}", "${", "`"]


def detect_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    """
    Detect potential prompt injection attempts in user input.

    Args:
        text: User-provided text to analyze.

    Returns:
        Tuple of (is_injection_detected, list_of_matched_patterns).
    """
    matched: List[str] = []
    for pattern in INJECTION_PATTERNS:
        if pattern.search(text):
            matched.append(pattern.pattern)

    if matched:
        logger.warning(
            "Prompt injection detected",
            extra={"extra_data": {"patterns": matched, "input_preview": text[:200]}},
        )
    return len(matched) > 0, matched


def sanitize_input(text: str, max_length: int = 5000) -> str:
    """
    Sanitize user input for safe processing.

    Args:
        text: Raw user input.
        max_length: Maximum allowed input length.

    Returns:
        Sanitized text string.
    """
    # Truncate
    text = text[:max_length]

    # Remove null bytes
    text = text.replace("\x00", "")

    # Escape dangerous template characters
    for char in DANGEROUS_CHARS:
        text = text.replace(char, " ")

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def validate_transaction_input(data: dict) -> Tuple[bool, str]:
    """
    Validate transaction data input for the fraud detection endpoint.

    Args:
        data: Dictionary of transaction features.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not isinstance(data, dict):
        return False, "Input must be a dictionary"

    if not data:
        return False, "Input cannot be empty"

    # Check for reasonable numeric values
    for key, value in data.items():
        if not isinstance(key, str):
            return False, f"Feature names must be strings, got {type(key)}"
        if isinstance(value, (int, float)):
            if abs(value) > 1e10:
                return False, f"Feature '{key}' has unreasonably large value: {value}"
        elif value is not None:
            return False, f"Feature '{key}' must be numeric, got {type(value)}"

    return True, ""
