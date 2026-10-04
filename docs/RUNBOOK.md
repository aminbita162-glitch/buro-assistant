# Buro Assistant — Runbook

**Product:** Buro Assistant
**Version:** 1.0.0
**Date:** 2026-10-03

This runbook covers the three most common operational scenarios: a service incident, a tenant quota breach, and a bad reply template. Follow each section in order. Escalate to the architecture owner (Amin Azimi, Azimi Innovation Lab) if the issue is not resolved after completing the steps.

---

## 1. Service incident

### Symptoms
- Health endpoint `/health` returns non-200, or
- Readiness endpoint `/ready` returns `503`, or
- Error rate visible in traces / application logs spikes above baseline.

### Triage steps

1. **Check liveness.**
   ```
   curl -s http://localhost:8000/health
   ```
   Expected: `{"status":"ok"}`. If this fails, the process is down — restart it (step 4).

2. **Check readiness (database).**
   ```
   curl -s http://localhost:8000/ready
   ```
   Expected: `{"status":"ok"}`. If `503`, the database is unreachable. Check `DATABASE_URL` and database connectivity before restarting.

3. **Check recent traces.**
   Query the `traces` table for `status = 'error'` rows in the last 15 minutes:
   ```sql
   SELECT stage, detail, created_at
   FROM traces
   WHERE status = 'error'
   ORDER BY created_at DESC
   LIMIT 20;
   ```

4. **Restart the process.**
   ```bash
   ./run.sh
   ```
   Alembic migrations run automatically on startup. If migrations fail, run them manually:
   ```bash
   alembic upgrade head
   ```

5. **Check audit log** for unusual event sequences:
   ```sql
   SELECT event, actor, created_at
   FROM audit_log
   ORDER BY created_at DESC
   LIMIT 50;
   ```

6. **Verify the dead-letter queue** — failed work items accumulate here:
   ```sql
   SELECT id, failure_reason, failed_at
   FROM dead_letter
   WHERE replayed_at IS NULL
   ORDER BY failed_at DESC;
   ```
   To replay a dead-letter item, call `app.workers.dlq.replay(db, dlq_item_id, tenant_id)`.

7. **Escalate** if the root cause is not clear after the above steps.

---

## 2. Quota breach

### Symptoms
- Operator reports that new messages are not being processed.
- Application logs contain `QuotaExceeded`.
- Dashboard shows no new `classified` messages despite inbound traffic.

### Triage steps

1. **Identify the exhausted tenant.**
   ```sql
   SELECT tenant_id, quota_date, tokens_used, cost_usd_used
   FROM quotas
   WHERE quota_date = CURRENT_DATE
   ORDER BY tokens_used DESC
   LIMIT 10;
   ```

2. **Compare against the configured limit.**
   The default limit is 100 000 tokens/day (set in tenant policy config under `daily_token_quota`).

3. **Temporary relief — reset the quota row** (emergency only; document the decision):
   ```sql
   UPDATE quotas
   SET tokens_used = 0, cost_usd_used = 0
   WHERE tenant_id = <id> AND quota_date = CURRENT_DATE;
   ```

4. **Permanent fix — raise the tenant quota.**
   Update the tenant's policy config to increase `daily_token_quota`, then redeploy or reload config.

5. **Review usage events** to understand what consumed the budget:
   ```sql
   SELECT event_type, quantity, unit, cost_usd, created_at
   FROM usage_events
   WHERE tenant_id = <id>
     AND created_at >= NOW() - INTERVAL '1 day'
   ORDER BY created_at DESC;
   ```

6. **Check the work queue** for backpressured items:
   ```sql
   SELECT priority, count(*) AS cnt
   FROM work_queue
   WHERE tenant_id = <id> AND state = 'pending'
   GROUP BY priority;
   ```
   Items enqueued during the quota breach will have `state = pending`. They will be processed automatically once the quota is restored.

---

## 3. Bad template

### Symptoms
- Reply drafts contain garbled text, missing variables, or forbidden phrases.
- Operator reports that the receipt template is wrong.
- Application logs contain `ValueError: Reply body contains forbidden content`.

### Triage steps

1. **Identify the offending template.**
   The `drafts` table stores `template_id` on every draft. Find recent failed drafts:
   ```sql
   SELECT id, template_id, subject, state, created_at
   FROM drafts
   WHERE state = 'failed'
   ORDER BY created_at DESC
   LIMIT 10;
   ```

2. **Inspect the template registry.**
   The default registry is loaded in `app/policy/templates.py`. The built-in receipt template is `receipt-v1`. To inspect the rendered output for a given set of variables, run in a Python shell:
   ```python
   from app.policy.templates import DEFAULT_REGISTRY
   entry = DEFAULT_REGISTRY.require("receipt-v1")
   print(entry.render({"original_subject": "test", "sender_name": "A", "reference": "R-1"}))
   ```

3. **Check for forbidden phrases.**
   Forbidden phrases are listed in `TemplateEntry.forbidden_phrases`. Any phrase in that set will cause `render()` to raise `ValueError`. Remove or rephrase the offending content in the template body.

4. **Check for missing variables.**
   If a required variable is absent from the `variables` dict passed to `render()`, a `KeyError` is raised. Ensure the policy layer populates all required variables before calling `draft_reply`.

5. **Deploy the corrected template.**
   Update the template body in `app/policy/templates.py`, redeploy, and re-process any failed drafts by replaying them from the approval queue or dead-letter queue.

6. **Log the incident** in the audit log:
   ```python
   from app.policy.audit import log_event
   log_event(db, tenant_id, "bad_template_incident",
             actor="operator@company.com", detail="template receipt-v1 corrected")
   ```

---

*End of runbook. For escalation contact: Amin Azimi, AI Architect, Azimi Innovation Lab.*
