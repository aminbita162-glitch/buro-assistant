#!/usr/bin/env bash
# ============================================================
# Buro Assistant — External test script (Follow-up Phase 10)
# ============================================================
# Author:  Amin Azimi, AI Architect, Azimi Innovation Lab
# Purpose: Verify that a second account can install the system,
#          seed it, send a fake inbound message, see a draft on
#          the desk, and confirm that a cross-tenant read is
#          blocked.
#
# Status: not run
# When this script is run by a second account the runner
# records date, actor, and result in docs/TEST_HOUSE.md.
#
# Prerequisites
# -------------
#   • Python 3.11 or later
#   • Access to a PostgreSQL instance (or set DATABASE_URL to
#     sqlite:///buro_test.db for a local file-based run)
#   • The repository cloned to a clean directory
#   • No .env file committed — copy .env.example to .env and
#     fill in DATABASE_URL and OPENAI_API_KEY before running
#
# Usage
# -----
#   cp .env.example .env          # fill in DATABASE_URL and OPENAI_API_KEY
#   bash docs/EXTERNAL_TEST.sh
#
# The script exits 0 on pass, non-zero on the first failure.
# Each check prints PASS or FAIL with a short reason.
# ============================================================

set -euo pipefail

# ---- colour helpers ----
GREEN='\033[0;32m'
RED='\033[0;31m'
RESET='\033[0m'

pass() { echo -e "${GREEN}PASS${RESET}  $1"; }
fail() { echo -e "${RED}FAIL${RESET}  $1"; exit 1; }

# ---- load .env if present ----
if [ -f .env ]; then
  # shellcheck disable=SC1091
  set -a; source .env; set +a
fi

# ---- guard: DATABASE_URL must be set ----
if [ -z "${DATABASE_URL:-}" ]; then
  echo "DATABASE_URL is not set."
  echo "Copy .env.example to .env and set DATABASE_URL before running."
  exit 1
fi

# ---- guard: OPENAI_API_KEY must be set (may be a stub for offline runs) ----
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "OPENAI_API_KEY is not set."
  echo "Set it to any non-empty string for a local offline test."
  exit 1
fi

BASE_URL="${BURO_BASE_URL:-http://localhost:8000}"

echo ""
echo "========================================================"
echo " Buro Assistant — External test"
echo " BASE_URL = $BASE_URL"
echo "========================================================"
echo ""

# ================================================================
# Step 1 — Install dependencies
# ================================================================
echo "--- Step 1: install dependencies ---"
python3 -m pip install --quiet -r requirements.txt
pass "dependencies installed"

# ================================================================
# Step 2 — Run the automated test suite
# ================================================================
echo ""
echo "--- Step 2: automated test suite ---"
python3 -m pytest tests/ -q --tb=short 2>&1 | tail -5
# pytest exits non-zero on failure; set -e will catch it
pass "automated test suite passed"

# ================================================================
# Step 3 — Seed the demo tenant
# ================================================================
echo ""
echo "--- Step 3: seed demo tenant ---"
python3 -m app.domain.seed
pass "seed complete"

# ================================================================
# Step 4 — Start the server in the background
# ================================================================
echo ""
echo "--- Step 4: start server ---"
PORT="${PORT:-8000}"
uvicorn app.main:app --host 127.0.0.1 --port "$PORT" &
SERVER_PID=$!

# Give the server a moment to bind.
sleep 2

# Register a cleanup trap so the server is always stopped.
cleanup() {
  kill "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

# ================================================================
# Step 5 — Health check
# ================================================================
echo ""
echo "--- Step 5: health check ---"
HEALTH=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health")
if [ "$HEALTH" = "200" ]; then
  pass "GET /health → 200"
else
  fail "GET /health returned $HEALTH (expected 200)"
fi

# ================================================================
# Step 6 — Sign up tenant-A operator
# ================================================================
echo ""
echo "--- Step 6: sign up tenant-A operator ---"
SIGNUP_A=$(curl -s -X POST "$BASE_URL/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{"name":"External Tester A","email":"ext-tester-a@example.invalid","password":"exttest-A1"}')

STATUS_A=$(echo "$SIGNUP_A" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('user',{}).get('id',''))" 2>/dev/null)
if [ -n "$STATUS_A" ]; then
  pass "tenant-A operator created (user id=$STATUS_A)"
else
  # Already exists from a previous run — that is fine; login instead.
  echo "  (user may already exist, continuing)"
fi

# ================================================================
# Step 7 — Log in as tenant-A operator and capture token
# ================================================================
echo ""
echo "--- Step 7: log in as tenant-A ---"
LOGIN_A=$(curl -s -X POST "$BASE_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"ext-tester-a@example.invalid","password":"exttest-A1"}')

TOKEN_A=$(echo "$LOGIN_A" | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])" 2>/dev/null)
if [ -n "$TOKEN_A" ]; then
  pass "logged in as tenant-A (token captured)"
else
  fail "login failed for tenant-A: $LOGIN_A"
fi

# ================================================================
# Step 8 — POST a fake inbound message via /ingest
# ================================================================
echo ""
echo "--- Step 8: ingest a fake message ---"
INGEST=$(curl -s -X POST "$BASE_URL/ingest" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN_A" \
  -d '{
    "provider": "fake",
    "provider_message_id": "ext-test-msg-001",
    "subject": "External test invoice",
    "sender": "external-sender@example.invalid",
    "recipients": ["desk@example.invalid"],
    "body_text": "Please process this external test invoice.",
    "attachments": [],
    "raw": {}
  }')

INGEST_RESULT=$(echo "$INGEST" | python3 -c "import sys,json; print(json.load(sys.stdin).get('result',''))" 2>/dev/null)
if [ "$INGEST_RESULT" = "new" ]; then
  pass "message ingested (result=new)"
else
  fail "ingest returned unexpected result: $INGEST"
fi

# ================================================================
# Step 9 — Check the desk queue shows the message
# ================================================================
echo ""
echo "--- Step 9: desk queue contains the message ---"
QUEUE=$(curl -s "$BASE_URL/desk/queue" \
  -H "Authorization: Bearer $TOKEN_A")

COUNT=$(echo "$QUEUE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('count',0))" 2>/dev/null)
if [ "$COUNT" -ge 1 ] 2>/dev/null; then
  pass "desk/queue count=$COUNT (≥1)"
else
  fail "desk/queue returned count=$COUNT (expected ≥1): $QUEUE"
fi

# ================================================================
# Step 10 — Sign up tenant-B operator (second account, different tenant)
# ================================================================
echo ""
echo "--- Step 10: sign up tenant-B operator (second account) ---"
SIGNUP_B=$(curl -s -X POST "$BASE_URL/auth/signup" \
  -H "Content-Type: application/json" \
  -d '{"name":"External Tester B","email":"ext-tester-b@example.invalid","password":"exttest-B2"}')

STATUS_B=$(echo "$SIGNUP_B" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('user',{}).get('id',''))" 2>/dev/null)
if [ -n "$STATUS_B" ]; then
  pass "tenant-B operator created (user id=$STATUS_B)"
else
  echo "  (user may already exist, continuing)"
fi

LOGIN_B=$(curl -s -X POST "$BASE_URL/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email":"ext-tester-b@example.invalid","password":"exttest-B2"}')

TOKEN_B=$(echo "$LOGIN_B" | python3 -c "import sys,json; print(json.load(sys.stdin)['token'])" 2>/dev/null)
if [ -n "$TOKEN_B" ]; then
  pass "logged in as tenant-B (token captured)"
else
  fail "login failed for tenant-B: $LOGIN_B"
fi

# ================================================================
# Step 11 — Confirm cross-tenant read is blocked
# tenant-B must not see tenant-A's queue entries
# ================================================================
echo ""
echo "--- Step 11: cross-tenant read is blocked ---"
QUEUE_B=$(curl -s "$BASE_URL/desk/queue" \
  -H "Authorization: Bearer $TOKEN_B")

COUNT_B=$(echo "$QUEUE_B" | python3 -c "import sys,json; print(json.load(sys.stdin).get('count',0))" 2>/dev/null)
if [ "$COUNT_B" -eq 0 ] 2>/dev/null; then
  pass "tenant-B desk/queue count=0 (cross-tenant read blocked)"
else
  fail "tenant-B saw $COUNT_B items from tenant-A's queue (isolation failure)"
fi

# ================================================================
# Done
# ================================================================
echo ""
echo "========================================================"
echo " All checks passed."
echo " Record result in docs/TEST_HOUSE.md:"
echo "   Date:   $(date +%Y-%m-%d)"
echo "   Actor:  <your name or account>"
echo "   Action: bash docs/EXTERNAL_TEST.sh"
echo "   Result: passed"
echo "========================================================"
