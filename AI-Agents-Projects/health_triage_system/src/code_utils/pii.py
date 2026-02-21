import re
from typing import Tuple, Dict

# Simple, practical redaction
# Reduce accidental leakage of personal identifiers to the LLM

_PATTERNS = [
    # India-specific 10-digit phone numbers
    (re.compile(r"\b\d{10}\b"), "[REDACTED_PHONE]"),
    # Generic international phone numbers
    # Matches +91xxxxxxxxxx, +1-xxx-xxxxxxx, etc.
    (re.compile(r"\b\+?\d{1,3}[-\s]?\d{6,14}\b"), "[REDACTED_PHONE]"),
    # Email addresses (case-insensitive)
    (
        re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
        "[REDACTED_EMAIL]",
    ),
    # Dates in DD-MM-YYYY or DD/MM/YYYY formats
    # for DOB / appointment dates
    (re.compile(r"\b\d{2}[-/]\d{2}[-/]\d{4}\b"), "[REDACTED_DATE]"),
    # Aadhaar-like 12 digit numbers
    # This is length-based and intentionally conservative.
    (re.compile(r"\b\d{12}\b"), "[REDACTED_ID]"),
    # Medical Record Number patterns
    (re.compile(r"\bMRN[:\s]*\w+\b", re.I), "[REDACTED_MRN]"),
    # Explicit patient identifiers
    (re.compile(r"\bPatient\s*ID[:\s]*\w+\b", re.I), "[REDACTED_PATIENT_ID]"),
]


def redact(text: str) -> Tuple[str, Dict[str, int]]:
    """
    Redact obvious PII from free-text input.

    Parameters:
        text: Original user-provided symptoms or notes.

    Returns:
        redacted_text: The text with sensitive patterns replaced by tokens.
    """
    redacted = text
    counts: Dict[str, int] = {}
    for pattern, repl in _PATTERNS:
        before = redacted
        redacted = pattern.sub(repl, redacted)
        if redacted != before:
            counts[repl] = counts.get(repl, 0) + 1
    return redacted.strip(), counts
