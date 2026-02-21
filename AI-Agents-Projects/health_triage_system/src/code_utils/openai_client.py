import json
from typing import Any, Dict, List, Optional
from openai import OpenAI
from code_utils.config import SETTINGS


def build_triage_json_schema(
    allowed_departments: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a strict JSON Schema for the triage output.

    Why dynamic schema?
    - If allowed_departments is provided, we hard-constrain department fields using enum.
    - This prevents hallucinated departments and improves routing consistency.
    - If not provided, we fall back to "string" to keep the system usable in dev/demo.
    """
    dept_field: Dict[str, Any] = {"type": "string", "minLength": 1}
    if allowed_departments:
        dept_field = {"type": "string", "enum": allowed_departments}

    return {
        "name": "triage_result",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "urgency": {
                    "type": "string",
                    "enum": ["EMERGENCY", "URGENT", "SOON", "ROUTINE"],
                },
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "red_flags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 12,
                },
                "department_candidates": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "department": dept_field,
                            "score": {"type": "number", "minimum": 0, "maximum": 1},
                        },
                        "required": ["department", "score"],
                    },
                },
                "suggested_department": dept_field,
                "needs_human_routing": {"type": "boolean"},
                "short_rationale": {"type": "string", "minLength": 1, "maxLength": 500},
                "recommended_timeframe_minutes": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 10080,
                },
            },
            "required": [
                "urgency",
                "confidence",
                "red_flags",
                "department_candidates",
                "suggested_department",
                "needs_human_routing",
                "short_rationale",
                "recommended_timeframe_minutes",
            ],
        },
        "strict": True,
    }


SYSTEM_INSTRUCTIONS = """You are a hospital triage assistant for decision support.
The user input may be REDACTED. Do NOT attempt to infer or reconstruct PII.
You must:
- Output ONLY valid JSON that matches the schema.
- Be conservative: when in doubt, escalate urgency.
- Use common red flags (e.g., chest pain, breathing difficulty, stroke signs, severe bleeding, altered mental status, suicidal intent).
- If input suggests immediate danger, set urgency=EMERGENCY and timeframe=0-30 minutes.
- If department is uncertain, set needs_human_routing=true and provide department_candidates with scores.
This is not a diagnosis; it's urgency routing.
"""


class OpenAITriageClient:
    """
    OpenAI-backed triage client.

    Key design goal: ALWAYS return a JSON dict to callers.
    - Even when model output is missing/invalid, caller gets a safe fallback.
    - This prevents your Streamlit / scheduling code from crashing.
    """

    def __init__(self) -> None:
        if not SETTINGS.openai_api_key:
            raise RuntimeError("Missing OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=SETTINGS.openai_api_key)

    def triage(
        self,
        user_text: str,
        age: int | None = None,
        sex: str | None = None,
        allowed_departments: list[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Main triage call.

        Inputs are wrapped into a JSON payload to:
        - keep prompt formatting stable
        - make it easy to add/remove fields later
        - reduce prompt injection surface vs free-form text
        """
        user_payload = {
            "symptoms_text": user_text.strip(),
            "age": age,
            "sex": sex,
        }

        schema = build_triage_json_schema(allowed_departments)

        # If departments are provided, we reinforce the enum constraint at instruction level too.
        dept_guard = ""
        if allowed_departments:
            dept_guard = (
                "\nIMPORTANT: You MUST choose departments ONLY from this list:\n"
                + ", ".join(allowed_departments)
                + "\nDo NOT invent department names. If uncertain, set needs_human_routing=true."
            )

        # Responses API call with strict JSON schema output format.
        resp = self.client.responses.create(
            model=SETTINGS.openai_model,
            instructions=SYSTEM_INSTRUCTIONS + dept_guard,
            input=[
                {
                    "role": "user",
                    "content": (
                        "Triage this patient input. Return structured output.\n"
                        f"PatientInput: {json.dumps(user_payload, ensure_ascii=False)}"
                    ),
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema["name"],
                    "strict": schema["strict"],
                    "schema": schema["schema"],
                }
            },
            temperature=0.1,  # Low temperature improves determinism for production routing.
            max_output_tokens=400,  # Output bounded to avoid long rationales / runaway token usage.
        )

        # The SDK typically exposes resp.output_text (JSON string).
        # handle missing output.
        raw_text = getattr(resp, "output_text", None)
        if not raw_text:
            # Fallback: try to find first message content
            raw_text = json.dumps(
                {
                    "urgency": "URGENT",
                    "confidence": 0.3,
                    "red_flags": ["unparsed_output"],
                    "suggested_department": "General Medicine",
                    "short_rationale": "Model output missing.",
                    "recommended_timeframe_minutes": 120,
                }
            )

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            # fallback if model returned non-JSON (rare with strict schema)
            return {
                "urgency": "URGENT",
                "confidence": 0.3,
                "red_flags": ["invalid_json_from_model"],
                "suggested_department": "General Medicine",
                "short_rationale": "Model returned invalid JSON; escalated for safety.",
                "recommended_timeframe_minutes": 120,
            }

    def triage_repair(
        self,
        original: Dict[str, Any],
        errors: List[str],
        symptoms_text: str,
        age: Optional[int] = None,
        sex: Optional[str] = None,
        allowed_departments: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Repair ONLY structural issues in the original output.
        Do not change the decision intent unless absolutely necessary.

        Inputs:
          - original: the model's prior JSON-like output
          - errors: list of structural validation errors from validate_required_fields()
        """
        system = (
            "You are a strict JSON repair engine for a healthcare triage system. "
            "Your job is to output a corrected JSON object that satisfies the required fields and types. "
            "Do NOT add extra keys. Do NOT include markdown. Output JSON only."
        )

        # You can tighten this by providing a schema or explicit field requirements.
        requirements = {
            "urgency": "One of: EMERGENCY, URGENT, SOON, ROUTINE",
            "confidence": "number between 0 and 1",
            "red_flags": "array of strings (can be empty)",
            "suggested_department": "non-empty string",
            "short_rationale": "non-empty string",
            "recommended_timeframe_minutes": "integer >= 0",
            # optional flags you might support:
            # "needs_human_routing": "boolean",
            # "department_candidates": "array of {department, score}"
        }

        user = {
            "task": "Fix the JSON output to satisfy requirements. Preserve intent.",
            "symptoms_text": symptoms_text,
            "age": age,
            "sex": sex,
            "allowed_departments": allowed_departments,
            "validation_errors": errors,
            "requirements": requirements,
            "original_output": original,
        }

        # --- Call your model here ---
        # Example shape (replace with your actual OpenAI call):
        repaired_text = self._call_model(system=system, user_json=user)

        # Parse and return dict safely
        try:
            repaired = json.loads(repaired_text)
            if not isinstance(repaired, dict):
                return original
            return repaired
        except Exception:
            # If repair failed, return original so engine can retry/fallback
            return original
