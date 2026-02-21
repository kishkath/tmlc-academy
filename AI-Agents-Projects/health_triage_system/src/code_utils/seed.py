from datetime import datetime, timedelta, timezone
from db import init_db, connect, now_iso
from config import SETTINGS

UTC = timezone.utc


def iso_z(dt: datetime) -> str:
    """
    Convert a timezone-aware datetime into an ISO8601 string with 'Z' suffix.
    Example: 2026-02-01T10:00:00Z
    """
    return dt.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def seed_slots():
    """
    Seed the slots table with deterministic availability.

    What this script does:
    - Ensures DB + tables exist
    - Clears existing slots
    - Inserts 7 days worth of 30-minute slots for each provider
    - Leaves all slots as AVAILABLE and unassigned (appointment_id=NULL)
    """
    init_db(SETTINGS.db_path)
    with connect(SETTINGS.db_path) as conn:
        conn.execute("DELETE FROM slots")

        providers = [
            ("Dr. Patil", "General Medicine"),
            ("Dr. Sharma", "Cardiology"),
            ("Dr. Iyer", "Neurology"),
            ("Dr. Khan", "Pulmonology"),
        ]

        start = datetime.now(tz=UTC).replace(
            minute=0, second=0, microsecond=0
        ) + timedelta(minutes=30)
        for day in range(0, 7):
            for provider, dept in providers:
                for i in range(10):  # 10 slots/day/provider
                    s = start + timedelta(days=day, minutes=30 * i)
                    e = s + timedelta(minutes=30)
                    conn.execute(
                        "INSERT INTO slots(provider, department, start, end, status, appointment_id) VALUES (?,?,?,?, 'AVAILABLE', NULL)",
                        (provider, dept, iso_z(s), iso_z(e)),
                    )


if __name__ == "__main__":
    seed_slots()
    print("Seeded slots into:", SETTINGS.db_path)
