# Phase 8 — Visual README

**Contract:** Buro Assistant — Final build contract
**Phase:** 8 of 10
**Status:** closed
**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab
**Date:** 2026-10-04

---

## What was done

Phase 8 replaced the plain README opening with a visual release surface. GitHub
cannot run arbitrary page animation, so only real rendered visuals were used.

### Assets added

| File | Description |
|---|---|
| `docs/assets/cover.svg` | 780×200 dark-background cover SVG. Shows the product name, tagline, three feature pills (Rule Engine · Shadow Mode · Audit Chain), and the three agent names. Inline SVG — no external image URL. |
| `docs/assets/buro-loop.gif` | 60×20 px two-frame looping GIF (GIF89a, Netscape 2.0 loop extension, loop count = 0 = infinite). A blue dot pulses on/off at 0.8 s per frame. Stored in-repo; no external URL. |

### README changes

The top of `README.md` was replaced. All content from "What it does" onward is
unchanged.

**Added elements (in order):**

1. **Cover SVG** — rendered inline by GitHub as a full-width banner.

2. **shields.io badges** (six, all public shields.io URLs):
   - version `1.0.0`
   - phase 8 closed
   - tests 504 passing
   - self-hosted / operator-controlled
   - send off by default
   - license

3. **Looping GIF** — the `docs/assets/buro-loop.gif` live indicator displayed
   at 60 px width under the badges.

4. **Blockquote tagline** — `Self-hosted · Multi-tenant · Rules before model · Append-only audit`

5. **"Five things a buyer checks first" converted to a table** — replaces the
   plain numbered list.

6. **Mermaid flowchart** — replaces the ASCII art architecture diagram. The
   Mermaid source is fenced with ` ```mermaid ` and renders natively on GitHub.
   It covers the full pipeline: external mail → ingest → Amin rule check →
   model path → Leila / Amilos → policy → workers → operator desk.

7. **Security and privacy callout table** — eight rows covering redaction,
   attachment exclusion, audit hash chain, shadow mode, sender auth, API keys,
   webhooks, and secret hygiene. Two blockquotes frame the table.

8. **Agent roles at a glance table** — three rows (Amin, Amilos, Leila) with
   Accepts / Returns / Cannot-do columns.

The original ASCII diagram is preserved at
[`docs/diagrams/architecture.txt`](../diagrams/architecture.txt).
The author line, external links section, and all other content are unchanged.

---

## What is not claimed

- No sale is done. The payment boundary still uses a fake adapter.
- No hosted service is offered. This is operator-hosted only.
- The looping GIF is a visual asset; it does not indicate a running service.
- shields.io badges use the public static-badge endpoint. No private URL appears
  anywhere in the README.

---

## Files changed

| File | Change |
|---|---|
| `README.md` | Opening replaced with visual surface; Agents section moved to callout table; architecture replaced with Mermaid |
| `docs/assets/cover.svg` | New file |
| `docs/assets/buro-loop.gif` | New file |
| `docs/releases/final-phase-8.md` | This report |

---

## Tests

No new tests. Phase 8 is a documentation-only change. The test suite baseline
from Phase 7 (504 tests, 0 failures) is unchanged.
