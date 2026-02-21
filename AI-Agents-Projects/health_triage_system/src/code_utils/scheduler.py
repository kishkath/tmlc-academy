import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from code_utils.models import Urgency

UTC = timezone.utc


def parse_iso_z(s: str) -> datetime:
    """
    Parse ISO8601 strings that may end with 'Z' into a timezone-aware datetime.
    DB stores timestamps as strings like "2026-02-01T10:00:00Z".
    """
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)


def iso_z(dt: datetime) -> str:
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def urgency_to_max_delay_minutes(urgency: Urgency) -> int:
    """
    Map urgency class to the maximum acceptable delay before the patient is seen.
    This becomes the "urgency window" used by pick_best_slot().
    """
    return {
        "EMERGENCY": 30,
        "URGENT": 240,
        "SOON": 1440,
        "ROUTINE": 10080,
    }[urgency]


# Slot retrieval helper
def get_earliest_available_slot(conn, department: str, provider: Optional[str] = None):
    """
    Get the earliest AVAILABLE slot for a department (and optional provider).
    Used for:
    - dashboards
    - checking SLA breaches
    - explaining why preemption is required
    """
    base = "SELECT * FROM slots WHERE status='AVAILABLE' AND department=?"
    params = [department]
    if provider:
        base += " AND provider=?"
        params.append(provider)
    base += " ORDER BY start ASC LIMIT 1"
    return conn.execute(base, params).fetchone()


def pick_best_slot(
    conn: sqlite3.Connection,
    provider: Optional[str],
    department: str,
    urgency: Urgency,
    allow_fallback_outside_window: bool = True,
) -> Optional[sqlite3.Row]:
    """
    Slot selection policy (preemption-aware):
      - EMERGENCY: must be within 0-30 minutes; if none -> return None
      - URGENT: must be within 4 hours; if none -> return None
      - SOON: try within 24h; else (optionally) earliest overall
      - ROUTINE: earliest overall

    Why allow_fallback_outside_window?
    - For EMERGENCY/URGENT in intake, you often want allow_fallback_outside_window=False
      so the caller can attempt preemption instead of silently violating SLA.
    - For SOON/ROUTINE, fallback is acceptable (book something rather than nothing).
    """
    now = datetime.now(tz=UTC)
    max_delay = timedelta(minutes=urgency_to_max_delay_minutes(urgency))
    latest = now + max_delay

    # available slots in a department, optionally filtered by provider.
    base = "SELECT * FROM slots WHERE status='AVAILABLE' AND department=?"
    params = [department]

    if provider:
        base += " AND provider=?"
        params.append(provider)

    # select earliest slot within the urgency window
    q1 = base + " AND start>=? AND start<=? ORDER BY start ASC LIMIT 1"
    row = conn.execute(q1, params + [iso_z(now), iso_z(latest)]).fetchone()
    if row:
        return row

    # If caller forbids fallback, return None so caller can take an alternate action
    if not allow_fallback_outside_window:
        return None

    # Fallback: earliest available slot regardless of urgency window
    q2 = base + " ORDER BY start ASC LIMIT 1"
    return conn.execute(q2, params).fetchone()


def free_old_slot_if_any(conn: sqlite3.Connection, appointment_id: int) -> None:
    """
    Free the slot currently attached to an appointment (if any).
    Used when rescheduling or bumping an appointment.

    Note:
    - This assumes there is at most one slot row with appointment_id = X.
    - History should be stored separately (audit log), not in the slots row itself.
    """
    conn.execute(
        "UPDATE slots SET status='AVAILABLE', appointment_id=NULL WHERE appointment_id=?",
        (appointment_id,),
    )


UTC = timezone.utc

# Priority mapping useful if you later want sorting / comparisons.
URGENCY_PRIORITY = {
    "EMERGENCY": 4,
    "URGENT": 3,
    "SOON": 2,
    "ROUTINE": 1,
}


def _now_iso_z() -> str:
    return (
        datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )


def can_preempt(incoming_urgency: str, victim_urgency: str) -> bool:
    """
    Preemption eligibility policy (business rule):
      - EMERGENCY can bump SOON/ROUTINE
      - URGENT can bump ROUTINE
      - SOON/ROUTINE never bump anyone

    Note: This policy must align with clinical governance / hospital rules.
    """
    if incoming_urgency == "EMERGENCY":
        return victim_urgency in ("SOON", "ROUTINE")
    if incoming_urgency == "URGENT":
        return victim_urgency in ("ROUTINE",)
    return False


def find_preemptable_victim(
    conn, department: str, incoming_urgency: str, horizon_minutes: int = 180
) -> Optional[object]:
    """
    Find a booked appointment in the same department that can be preempted.

    Strategy:
    - restrict search to near-future horizon (default 3 hours)
    - only consider victim urgencies allowed by policy
    - pick the earliest upcoming victim slot, prioritizing "lowest urgency first"

    Returns:
    - a sqlite row containing both appointment + slot details
    - None if no suitable victim found
    """
    now = _now_iso_z()
    # horizon window for which we consider preemption candidates
    horizon = (
        (datetime.now(tz=UTC) + timedelta(minutes=horizon_minutes))
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    # Determine which urgencies are eligible to be bumped given incoming urgency.
    if incoming_urgency == "EMERGENCY":
        victim_urgencies = ("SOON", "ROUTINE")
    elif incoming_urgency == "URGENT":
        victim_urgencies = ("ROUTINE",)
    else:
        return None

    # SQL query returns combined appointment+slot context to support:
    # - moving victim appointment to another slot
    # - assigning victim slot to incoming appointment
    q = f"""
    SELECT
        a.id AS appointment_id,
        a.urgency AS appt_urgency,
        a.provider AS appt_provider,
        a.department AS appt_department,
        a.scheduled_start AS appt_start,
        a.scheduled_end AS appt_end,
        s.id AS slot_id,
        s.provider AS slot_provider,
        s.department AS slot_department,
        s.start AS slot_start,
        s.end AS slot_end
    FROM appointments a
    JOIN slots s ON s.appointment_id = a.id
    WHERE a.department = ?
      AND a.status IN ('SCHEDULED', 'RESCHEDULED')
      AND a.urgency IN ({",".join(["?"] * len(victim_urgencies))})
      AND s.start >= ?
      AND s.start <= ?
    ORDER BY
      CASE a.urgency
        WHEN 'ROUTINE' THEN 1
        WHEN 'SOON' THEN 2
        ELSE 3
      END,
      s.start ASC
    LIMIT 1
    """

    params = [department, *victim_urgencies, now, horizon]
    return conn.execute(q, params).fetchone()


def find_replacement_slot(
    conn, department: str, avoid_slot_id: int, min_start_iso: str, limit: int = 1
):
    """
    Find an AVAILABLE slot to move the victim to.

    Constraints:
    - same department
    - starts at or after min_start_iso (typically victim slot end)
    - not the avoid_slot_id (can't reassign victim back to same slot)
    """
    q = """
    SELECT *
    FROM slots
    WHERE status='AVAILABLE'
      AND department=?
      AND start >= ?
      AND id != ?
    ORDER BY start ASC
    LIMIT ?
    """
    return conn.execute(q, (department, min_start_iso, avoid_slot_id, limit)).fetchone()


def preempt_and_book(
    conn,
    *,
    patient_id: int,
    reason_text: str,
    incoming_urgency: str,
    department: str,
    create_appointment_fn,
    update_appointment_time_fn,
    attach_slot_to_appointment_fn,
    add_history_fn,
):
    """
    Preempt a victim appointment and book incoming appointment into victim slot.

    Transaction intent:
    - Victim gets moved to a replacement slot (rescheduled)
    - Incoming appointment takes over victim's old slot
    - History entries are written for both appointments

    Return:
      dict with keys:
        - success: bool
        - appt_id: int | None
        - victim_appt_id: int | None
        - msg: str
    """
    victim = find_preemptable_victim(
        conn, department=department, incoming_urgency=incoming_urgency
    )
    if not victim:
        return {
            "success": False,
            "appt_id": None,
            "victim_appt_id": None,
            "msg": "No preemptable victim found in horizon.",
        }

    # Replacement slot must start after victim slot end (or at least after now)
    replacement = find_replacement_slot(
        conn,
        department=department,
        avoid_slot_id=int(victim["slot_id"]),
        min_start_iso=str(victim["slot_end"]),
    )
    if not replacement:
        return {
            "success": False,
            "appt_id": None,
            "victim_appt_id": int(victim["appointment_id"]),
            "msg": "No replacement slot available; cannot preempt safely.",
        }

    victim_appt_id = int(victim["appointment_id"])
    victim_slot_id = int(victim["slot_id"])

    # Create incoming appointment using victim slot timing/provider
    incoming_appt_id = create_appointment_fn(
        conn,
        patient_id=patient_id,
        provider=victim["slot_provider"],
        department=victim["slot_department"],
        reason=reason_text,
        urgency=incoming_urgency,
        start=victim["slot_start"],
        end=victim["slot_end"],
    )

    # Move victim to replacement slot (book replacement for victim)
    attach_slot_to_appointment_fn(
        conn, slot_id=int(replacement["id"]), appointment_id=victim_appt_id
    )
    update_appointment_time_fn(
        conn,
        victim_appt_id,
        replacement["start"],
        replacement["end"],
        victim["appt_urgency"],
        "RESCHEDULED",
        provider=replacement["provider"],
        department=replacement["department"],
    )

    # Reassign victim slot to incoming appointment (keeps status BOOKED)
    conn.execute(
        "UPDATE slots SET appointment_id=? WHERE id=?",
        (incoming_appt_id, victim_slot_id),
    )

    # History logs
    add_history_fn(
        conn,
        appointment_id=victim_appt_id,
        event_type="RESCHEDULED",
        prev_slot_id=victim_slot_id,
        new_slot_id=int(replacement["id"]),
        prev_start=victim["slot_start"],
        prev_end=victim["slot_end"],
        new_start=replacement["start"],
        new_end=replacement["end"],
        prev_provider=victim["slot_provider"],
        new_provider=replacement["provider"],
        prev_department=victim["slot_department"],
        new_department=replacement["department"],
        prev_urgency=victim["appt_urgency"],
        new_urgency=victim["appt_urgency"],
        note=f"Preempted by {incoming_urgency} appointment #{incoming_appt_id}",
    )

    add_history_fn(
        conn,
        appointment_id=incoming_appt_id,
        event_type="CREATED",
        new_slot_id=victim_slot_id,
        new_start=victim["slot_start"],
        new_end=victim["slot_end"],
        new_provider=victim["slot_provider"],
        new_department=victim["slot_department"],
        new_urgency=incoming_urgency,
        note=f"Booked via preemption (bumped appointment #{victim_appt_id})",
    )

    return {
        "success": True,
        "appt_id": incoming_appt_id,
        "victim_appt_id": victim_appt_id,
        "msg": "Preempted successfully.",
    }
