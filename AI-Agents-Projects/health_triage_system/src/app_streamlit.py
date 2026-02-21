import json
import streamlit as st
from datetime import datetime, timezone, timedelta

from code_utils.config import SETTINGS
from code_utils.db import (
    init_db,
    connect,
    create_patient,
    create_appointment,
    attach_slot_to_appointment,
    record_triage,
    create_queue_item,
    list_queue,
    read_queue_item,
    update_queue_status,
    fetch_available_slots,
    get_slot_by_appointment,
    update_appointment_time,
    add_history,
    list_history,
    list_departments,
    list_recent_autobooked,
)
from code_utils.scheduler import (
    pick_best_slot,
    free_old_slot_if_any,
    preempt_and_book,
    get_earliest_available_slot,
)
from triage_engine import TriageEngine
from code_utils.pii import redact

# DB timestamps look like ISO strings with "Z". This helper normalizes them and renders in IST.
IST = timezone(timedelta(hours=5, minutes=30))


def fmt_ist(iso_z: str) -> str:
    """
    Convert an ISO timestamp (optionally ending with 'Z') into IST and return a human-readable string.
    Used for UI display; does NOT affect persisted DB timestamps.
    """
    if not iso_z:
        return ""
    s = iso_z[:-1] + "+00:00" if iso_z.endswith("Z") else iso_z
    dt = datetime.fromisoformat(s).astimezone(IST)
    return dt.strftime("%d %b %Y, %I:%M %p")


# Important: Streamlit reruns the entire script on every interaction.
st.set_page_config(page_title="Hospital Triage", layout="wide")
# Ensures tables exist before any action. Safe to call repeatedly (idempotent).
init_db(SETTINGS.db_path)


@st.cache_resource
def get_engine():
    """
    Cache the LLM triage engine as a singleton for the Streamlit process.
    This avoids re-initializing clients on every rerun.
    """
    return TriageEngine()


engine = get_engine()

st.title("🏥 Hospital Triage + Scheduling")
st.caption(
    "Decision-support only. Always confirm with clinician. Emergency cases must follow hospital protocol."
)

# Separate flows into tabs so each section is simpler and uses rerun patterns cleanly.
tab_intake, tab_queue, tab_autobook, tab_reschedule = st.tabs(
    ["📝 Intake", "🧑‍⚕️ Triage Queue", "📅 Auto-booking", "🛠️ Manual Reschedule"]
)


# INTAKE TAB

with tab_intake:
    c1, c2 = st.columns([1, 1])

    with c1:
        # Patient input fiels
        st.subheader("Patient intake")
        full_name = st.text_input("Full name", placeholder="e.g., Asha Kulkarni")
        phone = st.text_input("Phone (optional)")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        sex = st.selectbox("Sex", ["", "Female", "Male", "Other"], index=0)

        symptoms = st.text_area(
            "Symptoms / complaint (natural language)",
            height=170,
            placeholder="e.g., Severe chest pain for 20 minutes, sweating, shortness of breath...",
        )

        # Policy Controls
        st.divider()
        ENABLE_PREEMPTION = st.checkbox(
            "Enable auto-preemption for EMERGENCY/URGENT", value=True
        )
        STRICT_QUEUE_ON_NEEDS_HUMAN = st.checkbox(
            "Always route to queue when LLM asks for human routing", value=True
        )

        run = st.button("Run triage", type="primary", use_container_width=True)

    with c2:
        st.subheader("Triage output + action")
        if run:
            # Patient identity fields (note: phone optional)
            if not full_name.strip() or not symptoms.strip():
                st.error("Please enter at least Full name and Symptoms.")
            else:
                # PII control: redact before model
                redacted_text, redaction_counts = redact(symptoms)
                if redaction_counts:
                    st.info(f"PII redaction applied: {redaction_counts}")

                try:
                    with connect(SETTINGS.db_path) as conn:
                        allowed_departments = list_departments(conn)

                    # If TriageEngine supports allowed_departments, pass it.
                    try:
                        result = engine.run(
                            symptoms_text=redacted_text,
                            age=int(age),
                            sex=sex or None,
                            allowed_departments=allowed_departments,
                        )
                    except TypeError:
                        # Backward compatible (if you haven't updated engine signature yet)
                        result = engine.run(
                            symptoms_text=redacted_text, age=int(age), sex=sex or None
                        )

                    triage_json = engine.to_json(result)

                    st.markdown(
                        f"### Urgency: **{result.urgency}** (confidence {result.confidence:.2f})"
                    )
                    if result.red_flags:
                        st.write("**Red flags:**", ", ".join(result.red_flags))
                    st.write("**Suggested department:**", result.suggested_department)
                    st.write("**Rationale:**", result.short_rationale)
                    st.write(
                        "**Recommended timeframe (minutes):**",
                        result.recommended_timeframe_minutes,
                    )

                    # Multi-dept routing info (from raw)
                    raw = result.raw or {}
                    # department_candidates: ranked alternative departments
                    # needs_human_routing: explicit model signal to avoid automation
                    candidates = raw.get("department_candidates", [])
                    needs_human = bool(raw.get("needs_human_routing", False))
                    top_score = candidates[0]["score"] if candidates else 0.0

                    if candidates:
                        st.write("**Department candidates:**")
                        st.dataframe(candidates, use_container_width=True)

                    # Routing policy
                    # Conservative rule: queue if LLM requested, or low confidence, or weak department match.
                    if STRICT_QUEUE_ON_NEEDS_HUMAN:
                        route_to_queue = (
                            needs_human
                            or (result.confidence < 0.55)
                            or (top_score < 0.55)
                        )
                    else:
                        # More permissive: ignore needs_human for SOON/ROUTINE if confidence/score are good
                        route_to_queue = (
                            (result.urgency in ["EMERGENCY", "URGENT"] and needs_human)
                            or (result.confidence < 0.55)
                            or (top_score < 0.55)
                        )

                    # Explain decision for operator trust + auditability.
                    reasons = []
                    if needs_human:
                        reasons.append("LLM requested human routing")
                    if result.confidence < 0.55:
                        reasons.append("Low confidence")
                    if top_score < 0.55:
                        reasons.append("Low department score")

                    st.info(
                        "Decision: "
                        + (
                            "QUEUE (" + ", ".join(reasons) + ")"
                            if route_to_queue
                            else "AUTO-BOOK"
                        )
                    )

                    with connect(SETTINGS.db_path) as conn:
                        patient_id = create_patient(
                            conn, full_name=full_name, phone=phone or None
                        )

                        if route_to_queue:
                            qid = create_queue_item(
                                conn,
                                patient_id=patient_id,
                                input_text_redacted=redacted_text,
                                triage_json=triage_json,
                            )
                            st.warning(
                                f"Routed to TRIAGE QUEUE for nurse review. Queue ID: {qid}"
                            )
                        else:
                            # Auto-book
                            # First try: only accept slots that fall within urgency window.
                            slot = pick_best_slot(
                                conn,
                                provider=None,
                                department=result.suggested_department,
                                urgency=result.urgency,
                                allow_fallback_outside_window=False,  # IMPORTANT: requires updated signature
                            )

                            if slot:
                                # Book normally within urgency window
                                appt_id = create_appointment(
                                    conn,
                                    patient_id=patient_id,
                                    provider=slot["provider"],
                                    department=slot["department"],
                                    reason=redacted_text,
                                    urgency=result.urgency,
                                    start=slot["start"],
                                    end=slot["end"],
                                )
                                attach_slot_to_appointment(
                                    conn,
                                    slot_id=int(slot["id"]),
                                    appointment_id=appt_id,
                                )
                                record_triage(
                                    conn,
                                    appointment_id=appt_id,
                                    input_text=redacted_text,
                                    triage_json=triage_json,
                                )

                                add_history(
                                    conn,
                                    appointment_id=appt_id,
                                    event_type="CREATED",
                                    new_slot_id=int(slot["id"]),
                                    new_start=slot["start"],
                                    new_end=slot["end"],
                                    new_provider=slot["provider"],
                                    new_department=slot["department"],
                                    new_urgency=result.urgency,
                                    note="Auto-booked from intake (within urgency window)",
                                )

                                st.success(
                                    f"✅ Auto-booked appointment #{appt_id} at {slot['start']}"
                                )

                            else:
                                # No slot within window: try preemption for EMERGENCY/URGENT
                                # For EMERGENCY/URGENT, attempt preemption (bump someone else) if enabled.
                                preempt_msg = ""
                                if ENABLE_PREEMPTION and result.urgency in (
                                    "EMERGENCY",
                                    "URGENT",
                                ):
                                    # 1) Slot within window?
                                    slot_in_window = pick_best_slot(
                                        conn,
                                        None,
                                        result.suggested_department,
                                        result.urgency,
                                        allow_fallback_outside_window=False,
                                    )

                                    if slot_in_window:
                                        slot = slot_in_window  # book it
                                    else:
                                        # No slot in window -> check earliest overall slot to see breach
                                        earliest = get_earliest_available_slot(
                                            conn, result.suggested_department
                                        )

                                        # If earliest exists but is outside SLA => try preemption

                                        outcome = preempt_and_book(
                                            conn,
                                            patient_id=patient_id,
                                            reason_text=redacted_text,
                                            incoming_urgency=result.urgency,
                                            department=result.suggested_department,
                                            create_appointment_fn=create_appointment,
                                            update_appointment_time_fn=update_appointment_time,
                                            attach_slot_to_appointment_fn=attach_slot_to_appointment,
                                            add_history_fn=add_history,
                                        )

                                        if outcome.get("success"):
                                            appt_id = outcome["appt_id"]
                                            record_triage(
                                                conn,
                                                appointment_id=appt_id,
                                                input_text=redacted_text,
                                                triage_json=triage_json,
                                            )
                                            st.success(
                                                f"🚑 Booked via PREEMPTION. New appointment #{appt_id}. "
                                                f"Bumped appointment #{outcome['victim_appt_id']}."
                                            )
                                        else:
                                            preempt_msg = outcome.get(
                                                "msg", "Preemption failed."
                                            )

                                # If preemption failed OR disabled OR urgency not critical:
                                # still book earliest possible slot (even outside window) AND route to nurse review.
                                if (
                                    (not ENABLE_PREEMPTION)
                                    or (result.urgency not in ("EMERGENCY", "URGENT"))
                                    or preempt_msg
                                ):
                                    fallback_slot = pick_best_slot(
                                        conn,
                                        provider=None,
                                        department=result.suggested_department,
                                        urgency=result.urgency,
                                        allow_fallback_outside_window=True,
                                    )

                                    if fallback_slot:
                                        appt_id = create_appointment(
                                            conn,
                                            patient_id=patient_id,
                                            provider=fallback_slot["provider"],
                                            department=fallback_slot["department"],
                                            reason=redacted_text,
                                            urgency=result.urgency,
                                            start=fallback_slot["start"],
                                            end=fallback_slot["end"],
                                        )
                                        attach_slot_to_appointment(
                                            conn,
                                            slot_id=int(fallback_slot["id"]),
                                            appointment_id=appt_id,
                                        )
                                        record_triage(
                                            conn,
                                            appointment_id=appt_id,
                                            input_text=redacted_text,
                                            triage_json=triage_json,
                                        )

                                        add_history(
                                            conn,
                                            appointment_id=appt_id,
                                            event_type="CREATED",
                                            new_slot_id=int(fallback_slot["id"]),
                                            new_start=fallback_slot["start"],
                                            new_end=fallback_slot["end"],
                                            new_provider=fallback_slot["provider"],
                                            new_department=fallback_slot["department"],
                                            new_urgency=result.urgency,
                                            note="Auto-booked fallback (outside urgency window)",
                                        )

                                        # Also queue for nurse review
                                        qid = create_queue_item(
                                            conn,
                                            patient_id=patient_id,
                                            input_text_redacted=redacted_text,
                                            triage_json=triage_json,
                                        )
                                        conn.execute(
                                            "UPDATE triage_queue SET appointment_id=?, status='IN_REVIEW', updated_at=? WHERE id=?",
                                            (
                                                appt_id,
                                                datetime.utcnow()
                                                .replace(microsecond=0)
                                                .isoformat()
                                                + "Z",
                                                qid,
                                            ),
                                        )

                                        msg = f"✅ Booked earliest slot #{appt_id} at {fallback_slot['start']} AND routed to queue #{qid} for nurse review."
                                        if preempt_msg:
                                            msg += (
                                                f" (Preemption failed: {preempt_msg})"
                                            )
                                        st.warning(msg)
                                    else:
                                        qid = create_queue_item(
                                            conn,
                                            patient_id=patient_id,
                                            input_text_redacted=redacted_text,
                                            triage_json=triage_json,
                                        )
                                        st.error(
                                            f"❌ No slots available at all. Sent to queue. Queue ID: {qid}"
                                        )

                except Exception as e:
                    st.exception(e)


# TRIAGE QUEUE TAB

with tab_queue:
    st.subheader("Triage queue (nurse review)")
    nurse_name = st.text_input(
        "Reviewer name (optional)", placeholder="e.g., Nurse Priya"
    )

    # Streamlit reruns wipe local vars, so use st.session_state for persistent UI state.
    if "queue_flash" not in st.session_state:
        st.session_state.queue_flash = None
    if "selected_queue_id" not in st.session_state:
        st.session_state.selected_queue_id = None

    # Flash messages persist across reruns so user sees feedback after DB updates + st.rerun().
    flash = st.session_state.queue_flash
    if flash:
        level, msg = flash
        if level == "success":
            st.success(msg)
        elif level == "warning":
            st.warning(msg)
        else:
            st.error(msg)
        st.session_state.queue_flash = None

    qcol1, qcol2 = st.columns([1, 1])

    with qcol1:
        status_filter = st.selectbox(
            "Queue status",
            ["NEW", "IN_REVIEW", "BOOKED", "REJECTED"],
            index=0,
            key="queue_status_filter1",
        )

        # "Refresh list" is optional; Streamlit reruns anyway, but it's fine to keep
        st.button("Refresh list", use_container_width=True, key="queue_refresh_btn2s")

        with connect(SETTINGS.db_path) as conn:
            items = list_queue(conn, status=status_filter, limit=50)

        options = [
            f"{r['id']} | patient_id={r['patient_id']} | {r['created_at']}"
            for r in items
        ]

        # Keep selection sticky after rerun
        default_idx = 0
        if st.session_state.selected_queue_id is not None:
            for i, opt in enumerate(options, start=1):
                if opt.startswith(f"{st.session_state.selected_queue_id} "):
                    default_idx = i
                    break

        selected = st.selectbox(
            "Select queue item",
            ["(none)"] + options,
            index=default_idx,
            key="queue_item_select1",
        )

        if selected != "(none)":
            st.session_state.selected_queue_id = int(selected.split("|", 1)[0].strip())
    # LEFT: list + filters + selection
    with qcol1:
        status_filter = st.selectbox(
            "Queue status",
            ["NEW", "IN_REVIEW", "BOOKED", "REJECTED"],
            index=0,
            key="queue_status_filter2",
        )

        # Optional explicit refresh button (rerun happens anyway).
        st.button("Refresh list", use_container_width=True, key="queue_refresh_btn1")

        with connect(SETTINGS.db_path) as conn:
            # List queue items filtered by status.
            items = list_queue(conn, status=status_filter, limit=50)

        # Friendly label for dropdown. (DB remains source of truth)
        options = [
            f"{r['id']} | patient_id={r['patient_id']} | {r['created_at']}"
            for r in items
        ]

        # Make selection sticky across reruns by deriving default index from session_state.
        default_idx = 0
        if st.session_state.selected_queue_id is not None:
            for i, opt in enumerate(options, start=1):
                if opt.startswith(f"{st.session_state.selected_queue_id} "):
                    default_idx = i
                    break

        selected = st.selectbox(
            "Select queue item",
            ["(none)"] + options,
            index=default_idx,
            key="queue_item_select2",
        )

        if selected != "(none)":
            st.session_state.selected_queue_id = int(selected.split("|", 1)[0].strip())

    # RIGHT: view details + actions
    with qcol2:
        queue_id = st.session_state.selected_queue_id
        if not queue_id:
            st.info("Select a queue item from the left.")
        else:
            with connect(SETTINGS.db_path) as conn:
                item = read_queue_item(conn, queue_id)
                if not item:
                    st.error("Queue item not found.")
                    st.session_state.selected_queue_id = None
                    st.stop()

                st.markdown(f"### Queue #{item['id']} — Status: **{item['status']}**")
                st.write("**Redacted input:**")
                st.code(item["input_text_redacted"])

                # Stored triage JSON is your “decision artifact” — what nurse is reviewing.
                triage = json.loads(item["triage_json"])
                st.write("**Triage JSON:**")
                st.json(triage)

                # ACTION 1: Start review (NEW → IN_REVIEW)
                # This is a state machine transition. Must commit + rerun to refresh UI list.
                if item["status"] == "NEW":
                    if st.button(
                        "Start review (IN_REVIEW)",
                        use_container_width=True,
                        key=f"inrev_{queue_id}",
                    ):
                        update_queue_status(
                            conn,
                            queue_id,
                            "IN_REVIEW",
                            assigned_to=(nurse_name or None),
                        )
                        conn.commit()
                        st.session_state.queue_flash = (
                            "success",
                            f"Queue #{queue_id} moved to IN_REVIEW.",
                        )
                        st.rerun()

                # ACTION 2: Approve & book (IN_REVIEW → BOOKED)
                st.divider()
                st.subheader("Approve & book")

                dept = triage.get("suggested_department", "General Medicine")
                urgency = triage.get("urgency", "URGENT")

                # If already booked, do not create again (idempotency / avoiding duplicates).
                if item["appointment_id"]:
                    appt_id = int(item["appointment_id"])
                    st.info(
                        f"This queue item is already linked to appointment #{appt_id}."
                    )
                else:
                    # Suggest a slot to the nurse; approval will actually book it.
                    slot = pick_best_slot(
                        conn, provider=None, department=dept, urgency=urgency
                    )
                    if slot:
                        st.write(
                            f"Suggested slot: {slot['provider']} at {slot['start']}"
                        )
                    else:
                        st.error("No slot available for suggested department.")

                    if st.button(
                        "Approve and book earliest slot",
                        type="primary",
                        use_container_width=True,
                        disabled=(slot is None),
                        key=f"approve_{queue_id}",
                    ):
                        patient_id = int(item["patient_id"])

                        # Create appointment from queue (reviewed path)
                        appt_id = create_appointment(
                            conn,
                            patient_id=patient_id,
                            provider=slot["provider"],
                            department=slot["department"],
                            reason=item["input_text_redacted"],
                            urgency=urgency,
                            start=slot["start"],
                            end=slot["end"],
                        )
                        attach_slot_to_appointment(
                            conn, slot_id=int(slot["id"]), appointment_id=appt_id
                        )

                        # Attach original triage artifact to appointment for traceability.
                        record_triage(
                            conn,
                            appointment_id=appt_id,
                            input_text=item["input_text_redacted"],
                            triage_json=item["triage_json"],
                        )

                        # Immutable history entry for audit.
                        add_history(
                            conn,
                            appointment_id=appt_id,
                            event_type="CREATED",
                            new_slot_id=int(slot["id"]),
                            new_start=slot["start"],
                            new_end=slot["end"],
                            new_provider=slot["provider"],
                            new_department=slot["department"],
                            new_urgency=urgency,
                            note=f"Booked from queue #{queue_id} by {nurse_name or 'reviewer'}",
                        )

                        # Update queue status + link the created appointment id (state machine transition).
                        update_queue_status(
                            conn,
                            queue_id,
                            "BOOKED",
                            assigned_to=(nurse_name or None),
                            appointment_id=appt_id,
                        )
                        conn.commit()

                        # Persist message across rerun and refresh the UI list.
                        st.session_state.queue_flash = (
                            "success",
                            f"Booked appointment #{appt_id}. Queue #{queue_id} moved to BOOKED.",
                        )
                        st.rerun()

                # ACTION 3: Reject / escalate (→ REJECTED)
                st.divider()
                st.subheader("Reject / escalate")

                reject_note = st.text_input(
                    "Reject note (optional)",
                    placeholder="e.g., incomplete info / refer to ER desk",
                    key=f"rej_note_{queue_id}",
                )

                if st.button(
                    "Reject", use_container_width=True, key=f"reject_{queue_id}"
                ):
                    # NOTE: reject_note currently isn't stored anywhere (only shown in flash msg).
                    # If you want auditability, store it in DB.
                    update_queue_status(
                        conn, queue_id, "REJECTED", assigned_to=(nurse_name or None)
                    )
                    conn.commit()
                    st.session_state.queue_flash = (
                        "warning",
                        f"Queue #{queue_id} moved to REJECTED. {('Note: ' + reject_note) if reject_note else ''}",
                    )
                    st.rerun()

                # If booked: show appointment history (read-only audit view)
                if item["appointment_id"]:
                    appt_id = int(item["appointment_id"])

                    appt = conn.execute(
                        "SELECT * FROM appointments WHERE id=?", (appt_id,)
                    ).fetchone()
                    if not appt:
                        st.error("Appointment not found (may have been deleted).")
                        st.stop()

                    st.divider()
                    st.subheader(f"Appointment #{appt_id} history")

                    hist = list_history(conn, appt_id)
                    st.dataframe(
                        [
                            {
                                "time": h["created_at"],
                                "event": h["event_type"],
                                "from": (
                                    f"{h['prev_provider']} {h['prev_start']}"
                                    if h["prev_start"]
                                    else ""
                                ),
                                "to": (
                                    f"{h['new_provider']} {h['new_start']}"
                                    if h["new_start"]
                                    else ""
                                ),
                                "note": h["note"] or "",
                            }
                            for h in hist
                        ],
                        use_container_width=True,
                    )

# AUTO-BOOKING TAB
with tab_autobook:
    st.subheader("📅 Auto-booking dashboard")
    st.caption(
        "Shows recent auto-booked appointments and the slots assigned by the scheduler."
    )

    colA, colB = st.columns([1, 1])

    with colA:
        st.markdown("### Auto-booking rules (current)")
        st.write("""
- Auto-book happens when routing policy says **no queue**
- A slot is picked by `pick_best_slot()` based on urgency window
- Booking writes an immutable entry in `appointment_history`
""")

    with colB:
        st.markdown("### Quick checks")
        with connect(SETTINGS.db_path) as conn:
            depts = conn.execute(
                "SELECT DISTINCT department FROM slots ORDER BY department"
            ).fetchall()
            total_slots = conn.execute("SELECT COUNT(*) as c FROM slots").fetchone()[
                "c"
            ]
            avail_slots = conn.execute(
                "SELECT COUNT(*) as c FROM slots WHERE status='AVAILABLE'"
            ).fetchone()["c"]
        st.write(
            {
                "departments_in_slots": len(depts),
                "total_slots": total_slots,
                "available_slots": avail_slots,
            }
        )

    st.divider()

    st.markdown("### Recent auto-booked appointments")

    # Controls how much DB data is pulled for the dashboard.
    limit = st.slider("How many to show?", min_value=5, max_value=100, value=25, step=5)

    with connect(SETTINGS.db_path) as conn:
        rows = list_recent_autobooked(conn, limit=limit)

    if not rows:
        st.info(
            "No auto-booked appointments found yet. Run triage in Intake and allow auto-book."
        )
    else:
        df = [
            {
                "booked_at": r["booked_at"],
                "appointment_id": r["appointment_id"],
                "patient": r["patient_name"],
                "phone": r["phone"] or "",
                "urgency": r["urgency"],
                "department": r["department"],
                "provider": r["provider"],
                "slot_id": r["slot_id"] or "",
                "start": r["scheduled_start"],
                "end": r["scheduled_end"],
                "note": r["note"] or "",
            }
            for r in rows
        ]

        st.dataframe(df, use_container_width=True)

        st.divider()
        st.markdown("### Inspect a specific auto-booked appointment")

        appt_ids = [str(r["appointment_id"]) for r in rows]
        selected_appt = st.selectbox("Select appointment", ["(none)"] + appt_ids)

        if selected_appt != "(none)":
            appt_id = int(selected_appt)
            with connect(SETTINGS.db_path) as conn:
                appt = conn.execute(
                    "SELECT * FROM appointments WHERE id=?", (appt_id,)
                ).fetchone()
                slot = conn.execute(
                    "SELECT * FROM slots WHERE appointment_id=?", (appt_id,)
                ).fetchone()
                hist = list_history(conn, appt_id)

            st.markdown("#### Appointment")
            st.json(dict(appt) if appt else {"error": "not found"})

            st.markdown("#### Slot")
            st.json(dict(slot) if slot else {"note": "No slot row linked (unexpected)"})

            st.markdown("#### History")
            st.dataframe(
                [
                    {
                        "time": h["created_at"],
                        "event": h["event_type"],
                        "from": (
                            f"{h['prev_provider']} {h['prev_start']}"
                            if h["prev_start"]
                            else ""
                        ),
                        "to": (
                            f"{h['new_provider']} {h['new_start']}"
                            if h["new_start"]
                            else ""
                        ),
                        "note": h["note"] or "",
                    }
                    for h in hist
                ],
                use_container_width=True,
            )

# MANUAL RESCHEDULE TAB
with tab_reschedule:
    st.subheader("🛠️ Manual rescheduling (standalone)")
    st.caption(
        "Search an appointment and change its slot with full traceability + history."
    )

    with connect(SETTINGS.db_path) as conn:
        #  quick counters
        c1, c2, c3 = st.columns(3)
        total_appts = conn.execute("SELECT COUNT(*) as c FROM appointments").fetchone()[
            "c"
        ]
        booked_slots = conn.execute(
            "SELECT COUNT(*) as c FROM slots WHERE status='BOOKED'"
        ).fetchone()["c"]
        avail_slots = conn.execute(
            "SELECT COUNT(*) as c FROM slots WHERE status='AVAILABLE'"
        ).fetchone()["c"]

    c1.metric("Total appointments", total_appts)
    c2.metric("Booked slots", booked_slots)
    c3.metric("Available slots", avail_slots)

    st.divider()

    #  Search controls
    s1, s2 = st.columns([2, 1])

    with s1:
        search_mode = st.radio(
            "Find appointment by",
            ["Appointment ID", "Patient name (contains)", "Recent appointments"],
            horizontal=True,
        )

    with s2:
        limit = st.selectbox("List limit", [10, 25, 50, 100], index=1)

    appt = None

    with connect(SETTINGS.db_path) as conn:
        if search_mode == "Appointment ID":
            appt_id_str = st.text_input("Enter appointment id", placeholder="e.g., 12")
            if appt_id_str.strip().isdigit():
                appt_id = int(appt_id_str.strip())
                appt = conn.execute(
                    "SELECT * FROM appointments WHERE id=?", (appt_id,)
                ).fetchone()

        elif search_mode == "Patient name (contains)":
            q = st.text_input("Search patient name", placeholder="e.g., Asha")
            if q.strip():
                rows = conn.execute(
                    """
                    SELECT a.* 
                    FROM appointments a
                    JOIN patients p ON p.id = a.patient_id
                    WHERE LOWER(p.full_name) LIKE ?
                    ORDER BY datetime(a.created_at) DESC
                    LIMIT ?
                    """,
                    (f"%{q.lower()}%", limit),
                ).fetchall()

                if rows:
                    opts = [
                        f"{r['id']} | patient_id={r['patient_id']} | {r['department']} | {r['scheduled_start']}"
                        for r in rows
                    ]
                    chosen = st.selectbox("Pick an appointment", ["(none)"] + opts)
                    if chosen != "(none)":
                        appt_id = int(chosen.split("|", 1)[0].strip())
                        appt = conn.execute(
                            "SELECT * FROM appointments WHERE id=?", (appt_id,)
                        ).fetchone()
                else:
                    st.info("No matching appointments.")

        else:  # Recent appointments
            rows = conn.execute(
                """
                SELECT * FROM appointments
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

            if rows:
                opts = [
                    f"{r['id']} | patient_id={r['patient_id']} | {r['department']} | {r['scheduled_start']}"
                    for r in rows
                ]
                chosen = st.selectbox("Pick an appointment", ["(none)"] + opts)
                if chosen != "(none)":
                    appt_id = int(chosen.split("|", 1)[0].strip())
                    appt = conn.execute(
                        "SELECT * FROM appointments WHERE id=?", (appt_id,)
                    ).fetchone()
            else:
                st.info("No appointments found yet.")

    st.divider()

    if not appt:
        st.info("Select an appointment to reschedule.")
        st.stop()

    #  Display appointment details
    appt_id = int(appt["id"])
    with connect(SETTINGS.db_path) as conn:
        patient = conn.execute(
            "SELECT * FROM patients WHERE id=?", (appt["patient_id"],)
        ).fetchone()
        old_slot = get_slot_by_appointment(conn, appt_id)  # helper

    left, right = st.columns([1, 1])

    with left:
        st.markdown(f"### Appointment #{appt_id}")
        st.write(
            {
                "patient": patient["full_name"] if patient else appt["patient_id"],
                "phone": (
                    patient["phone"] if patient and "phone" in patient.keys() else ""
                ),
                "urgency": appt["urgency"],
                "department": appt["department"],
                "provider": appt["provider"],
                "status": appt["status"],
                "scheduled_start": appt["scheduled_start"],
                "scheduled_end": appt["scheduled_end"],
            }
        )

        if old_slot:
            st.markdown("**Current slot**")
            st.write(
                {
                    "slot_id": old_slot["id"],
                    "provider": old_slot["provider"],
                    "department": old_slot["department"],
                    "start": old_slot["start"],
                    "end": old_slot["end"],
                    "status": old_slot["status"],
                }
            )
        else:
            st.warning(
                "No slot row linked to this appointment (unexpected but possible)."
            )

    with right:
        st.markdown("### Pick a new slot")

        with connect(SETTINGS.db_path) as conn:
            # by default show slots for same department
            dept = appt["department"]
            slots = fetch_available_slots(
                conn, provider=None, department=dept, limit=50
            )

        if not slots:
            st.error(f"No available slots found for department: {dept}")
            st.stop()

        options = [
            f"{r['id']} | {r['provider']} | {r['start']} → {r['end']}" for r in slots
        ]
        chosen_slot = st.selectbox(
            "Available slots", ["(none)"] + options, key=f"resched_pick_{appt_id}"
        )

        note = st.text_input(
            "Reschedule note (optional)",
            placeholder="e.g., patient requested later time / doctor change",
            key=f"resched_note_{appt_id}",
        )

        do_resched = st.button(
            "✅ Reschedule to selected slot", type="primary", use_container_width=True
        )

    if not do_resched:
        st.divider()
        st.markdown("### History")
        with connect(SETTINGS.db_path) as conn:
            hist = list_history(conn, appt_id)
        st.dataframe(
            [
                {
                    "time": h["created_at"],
                    "event": h["event_type"],
                    "from": (
                        f"{h['prev_provider']} {h['prev_start']}"
                        if h["prev_start"]
                        else ""
                    ),
                    "to": (
                        f"{h['new_provider']} {h['new_start']}"
                        if h["new_start"]
                        else ""
                    ),
                    "note": h["note"] or "",
                }
                for h in hist
            ],
            use_container_width=True,
        )
        st.stop()

    #  Execute reschedule
    if chosen_slot == "(none)":
        st.error("Pick a slot first.")
        st.stop()

    new_slot_id = int(chosen_slot.split("|", 1)[0].strip())

    # Prevent choosing the same slot
    if old_slot and int(old_slot["id"]) == new_slot_id:
        st.warning("Selected slot is already current. Pick a different one.")
        st.stop()

    with connect(SETTINGS.db_path) as conn:
        picked = conn.execute(
            "SELECT * FROM slots WHERE id=?", (new_slot_id,)
        ).fetchone()
        if not picked:
            st.error("Selected slot not found.")
            st.stop()

        if picked["status"] != "AVAILABLE":
            st.error("That slot is no longer available.")
            st.stop()

        # Book new slot first (atomic-ish)
        cur = conn.execute(
            "UPDATE slots SET status='BOOKED', appointment_id=? WHERE id=? AND status='AVAILABLE'",
            (appt_id, new_slot_id),
        )
        if cur.rowcount != 1:
            st.error("Failed to book slot (someone else took it). Try another.")
            st.stop()

        # Free old slot after booking new
        if old_slot:
            conn.execute(
                "UPDATE slots SET status='AVAILABLE', appointment_id=NULL WHERE id=? AND appointment_id=?",
                (int(old_slot["id"]), appt_id),
            )

        # Update appointment record including provider/department from picked slot
        update_appointment_time(
            conn,
            appt_id,
            picked["start"],
            picked["end"],
            appt["urgency"],
            "RESCHEDULED",
            provider=picked["provider"],
            department=picked["department"],
        )

        # History
        add_history(
            conn,
            appointment_id=appt_id,
            event_type="RESCHEDULED",
            prev_slot_id=int(old_slot["id"]) if old_slot else None,
            new_slot_id=int(picked["id"]),
            prev_start=old_slot["start"] if old_slot else appt["scheduled_start"],
            prev_end=old_slot["end"] if old_slot else appt["scheduled_end"],
            new_start=picked["start"],
            new_end=picked["end"],
            prev_provider=old_slot["provider"] if old_slot else appt["provider"],
            new_provider=picked["provider"],
            prev_department=old_slot["department"] if old_slot else appt["department"],
            new_department=picked["department"],
            prev_urgency=appt["urgency"],
            new_urgency=appt["urgency"],
            note=(note.strip() if note.strip() else "Manual reschedule"),
        )

        conn.commit()

    st.success(
        f"✅ Rescheduled appointment #{appt_id} to {picked['start']} → {picked['end']} (slot #{new_slot_id})"
    )
    st.rerun()
