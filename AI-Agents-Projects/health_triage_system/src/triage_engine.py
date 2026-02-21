from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal, Tuple
import time
import json

from pydantic import BaseModel, Field

from code_utils.models import TriageResult
from code_utils.openai_client import OpenAITriageClient

# Finite states for the agent loop.
AgentStatus = Literal["INIT", "TRIAGE", "REPAIR", "DONE", "FAILED"]


class TriageAgentState(BaseModel):
    """
    State container for the triage agent.

    Why state?
    - Streamlines "retry + repair" loops without spaghetti variables.
    - Provides an audit-friendly record of decisions (traces).
    - Makes it easy to persist to DB later for queue workflows.
    """

    # Inputs
    symptoms_text: str
    age: Optional[int] = None
    sex: Optional[str] = None
    allowed_departments: Optional[List[str]] = None

    # Outputs
    draft: Optional[Dict[str, Any]] = None
    result: Optional[TriageResult] = None

    # Control
    status: AgentStatus = "INIT"
    iteration: int = 0
    max_iterations: int = 3

    # Telemetry / audit

    errors: List[str] = Field(default_factory=list)
    traces: List[Dict[str, Any]] = Field(default_factory=list)
    started_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    def add_trace(self, kind: str, payload: Dict[str, Any]) -> None:
        """
        Append a trace event.

        This creates a lightweight, append-only audit trail.
        Later, you can store traces to DB for:
        - compliance / explainability
        - debugging prompt failures
        - replaying decisions
        """
        self.traces.append(
        {
            "ts_ms": int(time.time() * 1000),
            "iteration": self.iteration,
            "status": self.status,
            "kind": kind,
            "payload": payload,
        }
        )

def normalize_triage_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize and clamp model output into a safe, typed shape.

    This function should never throw unless data is extremely malformed.
    It's your "defensive parsing" layer even if the schema is strict.
    """
    urgency = str(data.get("urgency", "URGENT"))
    confidence = float(data.get("confidence", 0.3))
    red_flags = list(data.get("red_flags", []))
    dept = str(data.get("suggested_department", "General Medicine"))
    rationale = str(data.get("short_rationale", ""))
    timeframe = int(data.get("recommended_timeframe_minutes", 120))

    return {
        "urgency": urgency,
        "confidence": max(0.0, min(1.0, confidence)),
        "red_flags": [str(x) for x in red_flags][:12],
        "suggested_department": dept[:120],
        "short_rationale": rationale[:500],
        "recommended_timeframe_minutes": max(0, min(10080, timeframe)),
        "raw": data,
    }

def validate_required_fields(d: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Validate minimum correctness.

    This is NOT clinical validation.
    This is structural validation: fields exist + have correct types/ranges.

    Returns:
      (ok, errors)
    """
    errors: List[str] = []

    # Urgency must be one of allowed values.
    if d.get("urgency") not in {"EMERGENCY", "URGENT", "SOON", "ROUTINE"}:
        errors.append("urgency invalid or missing")

    # Confidence must be numeric in [0,1]
    conf = d.get("confidence")
    if not isinstance(conf, (int, float)) or not (0.0 <= float(conf) <= 1.0):
        errors.append("confidence invalid or missing")

    # Red flags must be a list (can be empty)
    if not isinstance(d.get("red_flags"), list):
        errors.append("red_flags must be a list")

    # Department and rationale are essential for routing + explainability.
    if not d.get("suggested_department"):
        errors.append("suggested_department missing")

    if not d.get("short_rationale"):
        errors.append("short_rationale missing")

    # Timeframe must be an int >= 0
    tf = d.get("recommended_timeframe_minutes")
    if not isinstance(tf, int) or tf < 0:
        errors.append("recommended_timeframe_minutes invalid or missing")

    return (len(errors) == 0), errors

class TriageEngine:
        """
    Agentic triage engine with:
    - bounded loop
    - validation
    - repair step
    - conservative fallback
    - trace logging

    The output is always a TriageResult (never None), so callers can
    safely proceed with routing policies even under failures.
    """

        def __init__(self) -> None:
            self.client = OpenAITriageClient()

        def run(
                self,
                symptoms_text: str,
                age: int | None,
                sex: str | None,
                allowed_departments: list[str] | None = None,
                max_iterations: int = 3,
        ) -> TriageResult:
            """
        Main agent loop.

        Strategy:
        1) Call triage() to get structured result
        2) Validate required fields
        3) If invalid, try triage_repair() to fix ONLY structural issues
        4) Stop after max_iterations and return conservative fallback
        """
            state = TriageAgentState(
                symptoms_text=symptoms_text,
                age=age,
                sex=sex,
                allowed_departments=allowed_departments,
                max_iterations=max_iterations,
                status="INIT",
            )

            while state.iteration < state.max_iterations:
                state.iteration += 1

                # TRIAGE step
                state.status = "TRIAGE"
                state.add_trace("step_start", {"step": "triage"})

                data = self.client.triage(
                    state.symptoms_text,
                    age=state.age,
                    sex=state.sex,
                    allowed_departments=state.allowed_departments,
                )
                # Save raw output in state for validation/repair.
                state.draft = data
                state.add_trace("llm_output", {"data": safe_compact(data)})

                # Validate
                ok, errs = validate_required_fields(state.draft)
                if ok:
                    normalized = normalize_triage_dict(state.draft)
                    state.result = TriageResult(**normalized)  # type: ignore
                    state.status = "DONE"
                    state.add_trace("done", {"reason": "valid_output"})
                    return state.result

                # Emergency fast path: if urgency says EMERGENCY, accept with defaults + log errors
                urgency = str(state.draft.get("urgency", "")).upper()
                if urgency == "EMERGENCY":
                    normalized = normalize_triage_dict(state.draft)
                    state.errors.extend(errs)
                    state.result = TriageResult(**normalized)  # type: ignore
                    state.status = "DONE"
                    state.add_trace(
                        "done", {"reason": "emergency_fast_path", "errors": errs}
                    )
                    return state.result

                # REPAIR step
                state.status = "REPAIR"
                state.errors.extend(errs)
                state.add_trace("repair_needed", {"errors": errs})

                repaired = self.client.triage_repair(
                    original=state.draft,
                    errors=errs,
                    symptoms_text=state.symptoms_text,
                    age=state.age,
                    sex=state.sex,
                    allowed_departments=state.allowed_departments,
                )
                state.draft = repaired
                state.add_trace("llm_repair_output", {"data": safe_compact(repaired)})

                ok2, errs2 = validate_required_fields(state.draft)
                if ok2:
                    normalized = normalize_triage_dict(state.draft)
                    state.result = TriageResult(**normalized)  # type: ignore
                    state.status = "DONE"
                    state.add_trace("done", {"reason": "repaired_output"})
                    return state.result

                # else loop continues until max_iterations

            # Fallback if repeatedly failing
            state.status = "FAILED"
            state.add_trace("failed", {"errors": state.errors[-10:]})

            # Conservative fallback result
            fallback = {
                "urgency": "URGENT",
                "confidence": 0.2,
                "red_flags": [],
                "suggested_department": "General Medicine",
                "short_rationale": "Unable to reliably generate a structured triage result. Please escalate to human review.",
                "recommended_timeframe_minutes": 120,
                "raw": {"errors": state.errors, "traces": state.traces[-8:]},
            }
            return TriageResult(**fallback)  # type: ignore

        @staticmethod
        def to_json(result: TriageResult) -> str:
            return json.dumps(
                {
                    "urgency": result.urgency,
                    "confidence": result.confidence,
                    "red_flags": result.red_flags,
                    "suggested_department": result.suggested_department,
                    "short_rationale": result.short_rationale,
                    "recommended_timeframe_minutes": result.recommended_timeframe_minutes,
                },
                ensure_ascii=False,
            )

def safe_compact(obj, max_len: int = 2000) -> Any:
        """
    Compact large objects before storing in traces/logs.

    Why:
    - prevents huge Streamlit payloads
    - avoids DB bloat when traces get persisted
    - reduces accidental sensitive data leakage in logs
    """
        try:
            s = json.dumps(obj, ensure_ascii=False)
            if len(s) > max_len:
                return {"_truncated": True, "preview": s[:max_len]}
            return obj
        except Exception:
            # If obj isn't JSON-serializable, store minimal metadata only.
            return {"_nonserializable": True, "type": str(type(obj))}
