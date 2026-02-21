import sqlite3
from contextlib import contextmanager
from typing import Iterator, Optional, Any, Dict, List, Tuple
from datetime import datetime
import json

# SQLite schema for the triage + scheduling system.

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS patients (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  full_name TEXT NOT NULL,
  phone TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS appointments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  provider TEXT NOT NULL,
  department TEXT NOT NULL,
  reason TEXT NOT NULL,
  urgency TEXT NOT NULL,
  status TEXT NOT NULL, -- SCHEDULED / RESCHEDULED / CANCELLED / QUEUED
  scheduled_start TEXT,
  scheduled_end TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  FOREIGN KEY(patient_id) REFERENCES patients(id)
);

CREATE TABLE IF NOT EXISTS slots (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  provider TEXT NOT NULL,
  department TEXT NOT NULL,
  start TEXT NOT NULL,
  end TEXT NOT NULL,
  status TEXT NOT NULL, -- AVAILABLE / BOOKED
  appointment_id INTEGER,
  FOREIGN KEY(appointment_id) REFERENCES appointments(id)
);

-- New: appointment history (immutable audit trail)
CREATE TABLE IF NOT EXISTS appointment_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  appointment_id INTEGER NOT NULL,
  event_type TEXT NOT NULL, -- CREATED / BOOKED / RESCHEDULED / CANCELLED / QUEUED / NOTE
  prev_slot_id INTEGER,
  new_slot_id INTEGER,
  prev_start TEXT,
  prev_end TEXT,
  new_start TEXT,
  new_end TEXT,
  prev_provider TEXT,
  new_provider TEXT,
  prev_department TEXT,
  new_department TEXT,
  prev_urgency TEXT,
  new_urgency TEXT,
  note TEXT,
  created_at TEXT NOT NULL,
  FOREIGN KEY(appointment_id) REFERENCES appointments(id)
);

-- New: triage queue (nurse review before booking)
CREATE TABLE IF NOT EXISTS triage_queue (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  patient_id INTEGER NOT NULL,
  input_text_redacted TEXT NOT NULL,
  triage_json TEXT NOT NULL,
  status TEXT NOT NULL, -- NEW / IN_REVIEW / APPROVED / REJECTED / BOOKED
  assigned_to TEXT,
  created_at TEXT NOT NULL,
  updated_at TEXT NOT NULL,
  appointment_id INTEGER,
  reject_note TEXT,
  FOREIGN KEY(patient_id) REFERENCES patients(id),
  FOREIGN KEY(appointment_id) REFERENCES appointments(id)
);

CREATE TABLE IF NOT EXISTS triage_events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  appointment_id INTEGER,
  input_text TEXT NOT NULL,
  triage_json TEXT NOT NULL,
  created_at TEXT NOT NULL,
  FOREIGN KEY(appointment_id) REFERENCES appointments(id)
);

CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  action TEXT NOT NULL,
  entity TEXT NOT NULL,
  entity_id INTEGER,
  details TEXT,
  created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_slots_provider_status_start ON slots(provider, status, start);
CREATE INDEX IF NOT EXISTS idx_slots_dept_status_start ON slots(department, status, start);
CREATE INDEX IF NOT EXISTS idx_appts_patient ON appointments(patient_id);
CREATE INDEX IF NOT EXISTS idx_hist_appt ON appointment_history(appointment_id, created_at);
CREATE INDEX IF NOT EXISTS idx_queue_status_created ON triage_queue(status, created_at);
"""


@contextmanager
def connect(db_path: str) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: str) -> None:
    import os

    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def log_audit(
    conn: sqlite3.Connection,
    action: str,
    entity: str,
    entity_id: int | None,
    details: str | None,
) -> None:
    conn.execute(
        "INSERT INTO audit_log(action, entity, entity_id, details, created_at) VALUES (?,?,?,?,?)",
        (action, entity, entity_id, details, now_iso()),
    )


def create_patient(conn: sqlite3.Connection, full_name: str, phone: str | None) -> int:
    cur = conn.execute(
        "INSERT INTO patients(full_name, phone, created_at) VALUES (?,?,?)",
        (full_name.strip(), (phone or "").strip() or None, now_iso()),
    )
    pid = int(cur.lastrowid)
    log_audit(conn, "CREATE", "patient", pid, f"name={full_name}")
    return pid


def list_providers(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT DISTINCT provider FROM slots ORDER BY provider"
    ).fetchall()
    return [r["provider"] for r in rows]


def list_departments(conn: sqlite3.Connection) -> List[str]:
    rows = conn.execute(
        "SELECT DISTINCT department FROM slots ORDER BY department"
    ).fetchall()
    return [r["department"] for r in rows]


def fetch_available_slots(
    conn: sqlite3.Connection,
    provider: str | None,
    department: str | None,
    limit: int = 50,
):
    q = "SELECT * FROM slots WHERE status='AVAILABLE'"
    params: list[Any] = []
    if provider:
        q += " AND provider=?"
        params.append(provider)
    if department:
        q += " AND department=?"
        params.append(department)
    q += " ORDER BY start ASC LIMIT ?"
    params.append(limit)
    return conn.execute(q, params).fetchall()


def create_appointment(
    conn: sqlite3.Connection,
    patient_id: int,
    provider: str,
    department: str,
    reason: str,
    urgency: str,
    start: str,
    end: str,
) -> int:
    t = now_iso()
    cur = conn.execute(
        """INSERT INTO appointments(patient_id, provider, department, reason, urgency, status,
                                    scheduled_start, scheduled_end, created_at, updated_at)
           VALUES (?,?,?,?,?,'SCHEDULED',?,?,?,?)""",
        (patient_id, provider, department, reason, urgency, start, end, t, t),
    )
    appt_id = int(cur.lastrowid)
    log_audit(
        conn, "CREATE", "appointment", appt_id, f"urgency={urgency} start={start}"
    )
    return appt_id


def attach_slot_to_appointment(
    conn: sqlite3.Connection, slot_id: int, appointment_id: int
) -> None:
    conn.execute(
        "UPDATE slots SET status='BOOKED', appointment_id=? WHERE id=? AND status='AVAILABLE'",
        (appointment_id, slot_id),
    )
    log_audit(
        conn, "UPDATE", "slot", slot_id, f"booked_for_appointment={appointment_id}"
    )


def get_appointment(conn: sqlite3.Connection, appointment_id: int):
    return conn.execute(
        "SELECT * FROM appointments WHERE id=?", (appointment_id,)
    ).fetchone()


def update_appointment_time(
    conn: sqlite3.Connection,
    appointment_id: int,
    start: str,
    end: str,
    urgency: str,
    status: str,
    provider: str | None = None,
    department: str | None = None,
) -> None:
    # Update provider/department only if passed
    if provider is not None and department is not None:
        conn.execute(
            """UPDATE appointments
               SET scheduled_start=?, scheduled_end=?,
                   urgency=?, status=?,
                   provider=?, department=?,
                   updated_at=?
               WHERE id=?""",
            (
                start,
                end,
                urgency,
                status,
                provider,
                department,
                now_iso(),
                appointment_id,
            ),
        )
        log_audit(
            conn,
            "UPDATE",
            "appointment",
            appointment_id,
            f"{status} start={start} urgency={urgency} provider={provider} dept={department}",
        )
    else:
        conn.execute(
            """UPDATE appointments
               SET scheduled_start=?, scheduled_end=?,
                   urgency=?, status=?,
                   updated_at=?
               WHERE id=?""",
            (start, end, urgency, status, now_iso(), appointment_id),
        )
        log_audit(
            conn,
            "UPDATE",
            "appointment",
            appointment_id,
            f"{status} start={start} urgency={urgency}",
        )


def record_triage(
    conn: sqlite3.Connection,
    appointment_id: int | None,
    input_text: str,
    triage_json: str,
) -> None:
    conn.execute(
        "INSERT INTO triage_events(appointment_id, input_text, triage_json, created_at) VALUES (?,?,?,?)",
        (appointment_id, input_text, triage_json, now_iso()),
    )


def add_history(
    conn: sqlite3.Connection,
    appointment_id: int,
    event_type: str,
    prev_slot_id: int | None = None,
    new_slot_id: int | None = None,
    prev_start: str | None = None,
    prev_end: str | None = None,
    new_start: str | None = None,
    new_end: str | None = None,
    prev_provider: str | None = None,
    new_provider: str | None = None,
    prev_department: str | None = None,
    new_department: str | None = None,
    prev_urgency: str | None = None,
    new_urgency: str | None = None,
    note: str | None = None,
) -> None:
    conn.execute(
        """INSERT INTO appointment_history(
            appointment_id, event_type,
            prev_slot_id, new_slot_id,
            prev_start, prev_end, new_start, new_end,
            prev_provider, new_provider,
            prev_department, new_department,
            prev_urgency, new_urgency,
            note, created_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            appointment_id,
            event_type,
            prev_slot_id,
            new_slot_id,
            prev_start,
            prev_end,
            new_start,
            new_end,
            prev_provider,
            new_provider,
            prev_department,
            new_department,
            prev_urgency,
            new_urgency,
            note,
            now_iso(),
        ),
    )


def get_slot_by_appointment(conn: sqlite3.Connection, appointment_id: int):
    return conn.execute(
        "SELECT * FROM slots WHERE appointment_id=?", (appointment_id,)
    ).fetchone()


def create_queue_item(
    conn: sqlite3.Connection,
    patient_id: int,
    input_text_redacted: str,
    triage_json: str,
) -> int:
    t = now_iso()
    cur = conn.execute(
        """INSERT INTO triage_queue(patient_id, input_text_redacted, triage_json, status, created_at, updated_at)
           VALUES (?,?,?,'NEW',?,?)""",
        (patient_id, input_text_redacted, triage_json, t, t),
    )
    qid = int(cur.lastrowid)
    log_audit(conn, "CREATE", "triage_queue", qid, "status=NEW")
    return qid


def update_queue_status(
    conn: sqlite3.Connection,
    queue_id: int,
    status: str,
    assigned_to: str | None = None,
    appointment_id: int | None = None,
) -> None:
    conn.execute(
        "UPDATE triage_queue SET status=?, assigned_to=COALESCE(?, assigned_to), appointment_id=COALESCE(?, appointment_id), updated_at=? WHERE id=?",
        (status, assigned_to, appointment_id, now_iso(), queue_id),
    )
    log_audit(conn, "UPDATE", "triage_queue", queue_id, f"status={status}")


def list_queue(conn: sqlite3.Connection, status: str = "NEW", limit: int = 50):
    return conn.execute(
        "SELECT * FROM triage_queue WHERE status=? ORDER BY created_at ASC LIMIT ?",
        (status, limit),
    ).fetchall()


def read_queue_item(conn: sqlite3.Connection, queue_id: int):
    return conn.execute("SELECT * FROM triage_queue WHERE id=?", (queue_id,)).fetchone()


def list_history(conn: sqlite3.Connection, appointment_id: int):
    return conn.execute(
        "SELECT * FROM appointment_history WHERE appointment_id=? ORDER BY created_at ASC",
        (appointment_id,),
    ).fetchall()


def list_recent_autobooked(conn: sqlite3.Connection, limit: int = 25):
    """
    Returns recent system-booked appointments:
    - normal auto-book from intake
    - fallback booking
    - preemption booking
    Uses appointment_history as source of truth.
    """
    return conn.execute(
        """
        SELECT
            a.id as appointment_id,
            p.full_name as patient_name,
            p.phone as phone,
            a.urgency as urgency,
            a.department as department,
            a.provider as provider,
            a.scheduled_start as scheduled_start,
            a.scheduled_end as scheduled_end,
            h.created_at as booked_at,
            h.note as note,
            s.id as slot_id
        FROM appointment_history h
        JOIN appointments a ON a.id = h.appointment_id
        JOIN patients p ON p.id = a.patient_id
        LEFT JOIN slots s ON s.appointment_id = a.id
        WHERE h.event_type = 'CREATED'
        ORDER BY h.created_at DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()


def find_preemptable_appointment(conn, department: str):
    """
    Find the lowest-urgency upcoming appointment
    that can be safely rescheduled.
    """
    return conn.execute(
        """
        SELECT a.*, s.id as slot_id
        FROM appointments a
        JOIN slots s ON s.appointment_id = a.id
        WHERE a.department = ?
          AND a.urgency IN ('SOON', 'ROUTINE')
          AND a.status = 'SCHEDULED'
        ORDER BY
          CASE a.urgency
            WHEN 'ROUTINE' THEN 1
            WHEN 'SOON' THEN 2
          END,
          s.start ASC
        LIMIT 1
        """,
        (department,),
    ).fetchone()


def update_appointment_time(
    conn: sqlite3.Connection,
    appointment_id: int,
    start: str,
    end: str,
    urgency: str,
    status: str,
    provider: str | None = None,
    department: str | None = None,
) -> None:
    if provider is not None and department is not None:
        conn.execute(
            """UPDATE appointments
               SET scheduled_start=?, scheduled_end=?,
                   urgency=?, status=?,
                   provider=?, department=?,
                   updated_at=?
               WHERE id=?""",
            (
                start,
                end,
                urgency,
                status,
                provider,
                department,
                now_iso(),
                appointment_id,
            ),
        )
    else:
        conn.execute(
            """UPDATE appointments
               SET scheduled_start=?, scheduled_end=?,
                   urgency=?, status=?,
                   updated_at=?
               WHERE id=?""",
            (start, end, urgency, status, now_iso(), appointment_id),
        )


def create_queue_item_for_appointment(
    conn: sqlite3.Connection,
    patient_id: int,
    appointment_id: int,
    input_text_redacted: str,
    triage_json: str,
    status: str = "IN_REVIEW",
    assigned_to: str | None = None,
) -> int:
    t = now_iso()
    cur = conn.execute(
        """INSERT INTO triage_queue(
            patient_id, input_text_redacted, triage_json, status,
            assigned_to, created_at, updated_at, appointment_id
        ) VALUES (?,?,?,?,?,?,?,?)""",
        (
            patient_id,
            input_text_redacted,
            triage_json,
            status,
            assigned_to,
            t,
            t,
            appointment_id,
        ),
    )
    qid = int(cur.lastrowid)
    log_audit(
        conn,
        "CREATE",
        "triage_queue",
        qid,
        f"status={status} appointment_id={appointment_id}",
    )
    return qid
