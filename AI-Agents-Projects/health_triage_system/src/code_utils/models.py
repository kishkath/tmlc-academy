from dataclasses import dataclass
from typing import List, Optional, Literal, Dict, Any

# Constrained urgency levels used across the entire system:
# - routing policy (queue vs autobook)
# - SLA windows (timeframe decisions)
# - escalation behavior (EMERGENCY fast path)
Urgency = Literal["EMERGENCY", "URGENT", "SOON", "ROUTINE"]


@dataclass
class TriageResult:
    """
    Typed triage output produced by the triage engine.

    This is intentionally small + stable:
    - enough to drive routing decisions and scheduling
    - enough to explain the result to a human reviewer
    - safe to store as an audit artifact (with optional raw payload)
    """

    urgency: Urgency  # Primary clinical urgency classification.
    confidence: float  # Model confidence in [0, 1].
    red_flags: List[str]  # Short list of red flags extracted from symptoms.
    suggested_department: str  # Department recommendation for routing and scheduling.
    short_rationale: str  # Human-readable explanation (kept short for UI and logs).
    recommended_timeframe_minutes: (
        int  # Recommended window in minutes for when the patient should be seen.
    )

    raw: Optional[Dict[str, Any]] = (
        None  # Optional raw model payload (full JSON), kept for debugging/audit.
    )
