# 🏥 Hospital Triage + Scheduling System

> **Decision-support system for intelligent patient triage, queue-based review, and automated appointment scheduling with preemption.**  
> Built for hospitals, clinics, and healthcare operators to reduce mis-triage, delays, and operational overload.

---

## 1. Overview

This project is a **production-oriented** of a Hospital Triage and Scheduling System that combines:

- Natural-language symptom understanding using LLMs  
- Structured triage decisions (urgency, department, timeframe)  
- Human-in-the-loop triage queue for safety  
- Automated appointment booking  
- Emergency/urgent **slot preemption** (bumping lower-priority cases)  
- Full audit trail and appointment history
---

## 2. What Problem Does This Solve?

### Real Hospital Problems
- ❌ Patients describe symptoms in free text → staff must manually interpret  
- ❌ Mis-triage leads to:
  - Emergency cases waiting too long  
  - Routine cases consuming scarce slots  
- ❌ Scheduling teams constantly reshuffle appointments under pressure  
- ❌ No transparent audit trail of “why this patient got this slot”  

### What This System Does
- Converts **natural language → structured triage**  
- Automatically routes:
  - 🔴 Emergency / Urgent → immediate booking or preemption  
  - 🟡 Ambiguous / low-confidence → nurse triage queue  
- Applies **explicit scheduling policies**  
- Maintains **immutable history** of every decision  

---

## 3. Key Capabilities

### 3.1 Intelligent Triage (LLM-powered)

From raw symptoms like:
```
Heavy pressure in my chest, pain going to left arm, sweating a lot
```

The system produces:
- Urgency: `EMERGENCY`
- Confidence score
- Red flags
- Suggested department
- Recommended timeframe (minutes)
- Department candidates with scores
- Optional flag: `needs_human_routing`

> ⚠️ LLM output is **decision-support only**.

---

### 3.2 Explicit Routing Policy

The system **does not blindly trust the LLM**.

Routing to **AUTO-BOOK** vs **TRIAGE QUEUE** depends on:
- LLM confidence
- Department score
- Explicit `needs_human_routing`
- Urgency level
- Operator-configurable strictness

---

### 3.3 Triage Queue (Human-in-the-Loop)

All uncertain cases enter a **database-backed queue** with statuses:

- `NEW`
- `IN_REVIEW`
- `BOOKED`
- `REJECTED`

Reviewers can:
- Inspect redacted input
- View full triage JSON
- Approve & book
- Reject or escalate

Queue transitions are persisted in SQLite (not UI state).

---

### 3.4 Automated Slot Booking

- Policy-driven slot selection
- Urgency-based time windows
- Writes to:
  - `appointments`
  - `slots`
  - `appointment_history`

---

### 3.5 Emergency / Urgent Preemption

If no suitable slot exists for high urgency:
- Lower-priority appointment is preempted
- Emergency case is booked immediately
- Both actions are logged

Example history note:
```
Booked via PREEMPTION. New appointment #2. Bumped appointment #1.
```

---

### 3.6 Full Audit Trail

Every appointment maintains immutable history:
- CREATED
- BOOKED_FROM_QUEUE
- PREEMPTED
- RESCHEDULED

This ensures explainability and medico-legal defensibility.

---

## 4. Architecture Overview

```
Streamlit UI
   │
   ├── Intake (Patient-facing)
   │     └── TriageEngine (LLM)
   │           └── OpenAI Client
   │
   ├── Triage Queue (Human Review)
   │     └── SQLite (triage_queue)
   │
   ├── Scheduler
   │     ├── pick_best_slot()
   │     ├── preempt_and_book()
   │
   └── Persistence Layer
         ├── patients
         ├── appointments
         ├── slots
         ├── triage_queue
         └── appointment_history
```

---

## 5. Tech Stack

- Python 3.11+
- Streamlit
- SQLite
- OpenAI API
- Pure Python architecture (no backend framework)

---

## 6. Database Design (High-Level)

| Table | Purpose |
|-----|-------|
| patients | Patient identity |
| appointments | Scheduled visits |
| slots | Provider availability |
| triage_queue | Human review workflow |
| appointment_history | Immutable audit log |

---

## 7. Safety & Compliance Principles

- 🔒 PII redaction before LLM calls  
- 🧑‍⚕️ Human review for low-confidence cases  
- 🚨 Emergency disclaimer always visible  
- 📜 Full decision traceability  
- 🔁 Manual override supported everywhere  

---

## 8. What This  Is (and Is Not)

### ✅ This IS
- A realistic hospital triage + scheduling engine  
- A foundation for enterprise deployment  

### ❌ This is NOT
- A replacement for clinicians  
- A diagnostic system  
- A regulatory-approved medical device  

---

## 9. Typical User Flows

### Flow A: Auto-book
```
Patient → Triage → High confidence → Slot booked → History written
```

### Flow B: Human review
```
Patient → Triage → Queue → Nurse review → Approve → Book
```

### Flow C: Emergency preemption
```
Patient → Emergency triage → No slot → Preempt → Immediate booking
```

---

## 10. Why This Design Matters

This system emphasizes:
- Explicit state transitions  
- Policy-driven automation  
- Human control  
- Auditability  

This is how **real healthcare AI systems** must be built.

---

## 11. Disclaimer

> This system provides **decision support only**.  
> All medical decisions must be confirmed by licensed clinicians.  
> Emergency cases must follow hospital protocol.