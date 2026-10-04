# Buro Assistant — Comparison with public mail desk products

**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab  
**Phase:** Follow-up Phase 8 — Differentiation pack  
**Date:** 2026-10-09  

---

## Honesty note

The comparisons below are based on the publicly documented feature sets of each
product at the time of writing. Buro Assistant makes no throughput claims against
any of these products. No benchmarks have been run against them. Where
capabilities are marked "designed for", they are not externally verified.

---

## Products compared

| Product | Category | Source used |
|---|---|---|
| HelpScout (https://www.helpscout.com) | Shared inbox, support desk | Public product page and documentation |
| Front (https://front.app) | Collaborative inbox | Public product page and documentation |
| Freshdesk (https://freshdesk.com) | Helpdesk ticketing | Public product page and documentation |

---

## Feature comparison

| Feature | Buro Assistant | HelpScout | Front | Freshdesk |
|---|---|---|---|---|
| Deployment model | Self-hosted by operator | SaaS (cloud-hosted) | SaaS (cloud-hosted) | SaaS (cloud-hosted) |
| Data residency | Operator-controlled | Vendor-controlled | Vendor-controlled | Vendor-controlled |
| IMAP ingest | ✓ verified | ✓ | ✓ | ✓ |
| Rule-before-model triage | ✓ verified | not documented | not documented | not documented |
| Zero model tokens on rule hit | ✓ verified | n/a | n/a | n/a |
| Attachment bytes never sent to model | ✓ verified | n/a | n/a | n/a |
| Redaction before model call | ✓ verified | not documented | not documented | not documented |
| Multi-tenant isolation (per-tenant DB rows) | ✓ verified | designed for (SaaS) | designed for (SaaS) | designed for (SaaS) |
| Append-only audit log | ✓ verified | not documented | ✓ (docs) | ✓ (docs) |
| Legal hold blocking retention delete | ✓ verified | not documented | not documented | enterprise tier |
| Role split (admin / operator / auditor) | ✓ designed for | ✓ (docs) | ✓ (docs) | ✓ (docs) |
| Collision lock on drafts | ✓ designed for | not documented | ✓ (docs) | not documented |
| After-hours holding queue | ✓ designed for | not documented | not documented | not documented |
| VIP sender list | ✓ designed for | not documented | ✓ (docs) | ✓ (docs) |
| Per-department SLA | ✓ designed for | not documented | ✓ (docs) | ✓ (docs) |
| Saved reply snippets (template-checked) | ✓ designed for | ✓ (docs) | ✓ (docs) | ✓ (docs) |
| CSV audit log export | ✓ verified | not documented | ✓ (docs) | ✓ (docs) |
| Signed outbound webhooks (HMAC-SHA256) | ✓ verified | ✓ (docs) | ✓ (docs) | ✓ (docs) |
| Per-tenant daily token quota | ✓ verified | n/a | n/a | n/a |
| Shadow mode (draft stored, not sent) | ✓ verified | not documented | not documented | not documented |
| Open-source / inspectable | ✓ (proprietary — all rights reserved, Amin Azimi) | ✗ closed source | ✗ closed source | ✗ closed source |
| Commercial sale | ✗ not done | ✓ | ✓ | ✓ |
| Hosted service | ✗ not done | ✓ | ✓ | ✓ |

---

## Key differences

### Self-hosted vs SaaS

HelpScout, Front, and Freshdesk are cloud-hosted services. The operator's data
resides on the vendor's infrastructure. Buro Assistant is self-hosted by the
operator. The database region and access controls are fully under the operator's
control. This is a design choice, not a claim of superiority.

### Rule-before-model pipeline

Buro Assistant routes a message through a deterministic rule engine before any
model call. A rule match produces a decision with zero model tokens consumed.
The three products above do not publicly document an equivalent guarantee.

### Redaction before model call

Buro Assistant applies a redaction pass before every model call. PII fields
stripped at the ingest stage are never included in the model prompt. This is
a documented, tested behavior. The three products above do not publicly document
an equivalent mechanism for third-party model calls.

### Attachment bytes

Buro Assistant explicitly excludes attachment bytes from all model prompts.
This is verified by `tests/test_token_policy.py`. The three products above do
not publicly document this constraint.

### Commercial and hosting gap

Buro Assistant does not offer a commercial subscription or a hosted service.
The products above do. This is a known, documented gap — not a claim.

---

## What Buro Assistant does not claim

- No throughput figures have been measured against any of the three products.
- No latency figures have been measured.
- No cost figures have been compared.
- The role-split and collision-lock features are designed for and not yet
  externally verified.
- Buro Assistant is not recommended for factory or commercial use until an
  external test has been run (Phase 10).

---

*For questions contact: Amin Azimi, AI Architect, Azimi Innovation Lab.*
