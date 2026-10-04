# Buro Assistant — Install Video Script

**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab  
**Phase:** Follow-up Phase 8 — Differentiation pack  
**Date:** 2026-10-09  

---

## Purpose

This document is the narration script for an install demonstration video.
It is not a video file. It describes what the presenter shows on screen and
what they say. A video editor or screen recorder can follow this script.

---

## Video metadata

- **Title:** Buro Assistant — Self-hosted install walkthrough
- **Duration (target):** 8–10 minutes
- **Audience:** Technical operators evaluating self-hosted mail desk software

---

## Script

### Scene 1 — Introduction (0:00–0:30)

**Screen:** Terminal on a clean Linux server (Ubuntu 22.04 LTS or equivalent).

**Narrator:**
> Welcome. This walkthrough shows how to install Buro Assistant on a
> self-hosted Linux server. You will need Git, Python 3.11 or later, and a
> running PostgreSQL 15 database. The whole install takes under ten minutes.
> Buro Assistant is self-hosted. You control where your data lives.

---

### Scene 2 — Clone the repository (0:30–1:00)

**Screen:** Terminal.

```bash
git clone <repo-url>
cd buro-assistant
```

**Narrator:**
> Clone the repository to your server. The main branch is the only
> supported branch.

---

### Scene 3 — Copy the environment file (1:00–1:45)

**Screen:** Terminal, then a text editor showing `.env`.

```bash
cp .env.example .env
```

**Narrator:**
> Copy the example environment file. Open it in your editor. You need to
> fill in exactly four values: DATABASE_URL with your PostgreSQL connection
> string, OPENAI_API_KEY if you want AI triage, PORT if you want a port
> other than 8000, and ALLOWED_ORIGINS for your CORS allowlist.

**Important for operator:**
> Never commit `.env` to version control. The file is in `.gitignore` by default.

---

### Scene 4 — Install dependencies (1:45–2:30)

**Screen:** Terminal running pip install.

```bash
pip install -r requirements.txt
```

**Narrator:**
> Install the Python dependencies. This takes about thirty seconds on a
> standard server. All packages are public PyPI packages.

---

### Scene 5 — Run database migrations (2:30–3:15)

**Screen:** Terminal running Alembic.

```bash
alembic upgrade head
```

**Narrator:**
> Run the database migrations. Alembic creates all required tables.
> Schema changes only happen through migrations, never on import.
> If the database is empty, this creates the schema from scratch.
> If you are upgrading, only the new tables and columns are added.

---

### Scene 6 — Seed a sandbox tenant (3:15–4:00)

**Screen:** Terminal.

```bash
python -m app.domain.sandbox
```

**Narrator:**
> Seed a sandbox tenant. This creates one tenant, one operator user,
> and a set of example messages. Use this to explore the desk before
> connecting a real mailbox. The sandbox user and messages are
> clearly marked as test data.

---

### Scene 7 — Start the server (4:00–4:45)

**Screen:** Terminal running run.sh, then browser opening the desk.

```bash
./run.sh
```

**Narrator:**
> Start the server. The desk opens at http://localhost:8000.
> Log in with the sandbox operator credentials printed by the seed command.
> You will see the Dashboard, Inbound, Decisions, Drafts, Approval,
> Audit, Quota, and Cost sections.

---

### Scene 8 — Verify health (4:45–5:15)

**Screen:** Browser or curl.

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

**Narrator:**
> Verify the health endpoints. Live returns 200 when the server process
> is running. Ready returns 200 when the database connection is working.
> Use these in your load balancer or container health check.

---

### Scene 9 — Run the tests (5:15–6:30)

**Screen:** Terminal running pytest.

```bash
python3 -m pytest tests/ -v
```

**Narrator:**
> Run the test suite. All tests should pass. Tests use SQLite in memory
> and the fake mail provider, so they run without a real PostgreSQL or
> IMAP server. A green result means the install is correct.

---

### Scene 10 — Connect a real mailbox (optional) (6:30–7:30)

**Screen:** Text editor showing `.env` with IMAP values.

**Narrator:**
> To connect a real IMAP mailbox, set IMAP_HOST, IMAP_PORT, IMAP_USER,
> and IMAP_PASSWORD in your `.env` file. Restart the server. The intake
> loop will begin polling your mailbox. Credentials stay in the environment;
> they are never logged or stored in the database.

---

### Scene 11 — What is not done (7:30–8:00)

**Screen:** README.md "What is not claimed" section.

**Narrator:**
> Buro Assistant is self-hosted software. There is no hosted service and
> no commercial subscription from Azimi Innovation Lab. Sale and hosting
> are out of scope. If you need a managed service, the three products
> compared in docs/COMPARISON.md offer that option.

---

### Scene 12 — Close (8:00–8:30)

**Screen:** README.md author section.

**Narrator:**
> That completes the install. The full documentation is in the docs/
> directory: INSTALL.md, RUNBOOK.md, THREAT_MODEL.md, and PRIVACY_DATA_MAP.md.
> Questions go to Amin Azimi, AI Architect, Azimi Innovation Lab.

---

## Notes for the video editor

- Do not show real credentials, database passwords, API keys, or mailbox passwords.
- Use placeholder values such as `sk-...` for OPENAI_API_KEY and
  `postgresql://user:password@localhost:5432/buro` for DATABASE_URL.
- Pause on each terminal command for at least two seconds before advancing.
- Captions should match the narrator text exactly.
