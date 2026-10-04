# Commercial Phase 2 — Gateway boundary

**Date:** 2026-10-13
**Branch:** main
**Author:** Amin Azimi, AI Architect, Azimi Innovation Lab

---

## What was built

A payment port with two adapters was added to the system.

### New package: `app/payment/`

| File | Role |
|---|---|
| `app/payment/__init__.py` | Package marker |
| `app/payment/port.py` | `PaymentPort` abstract base class; `ChargeResult` dataclass |
| `app/payment/fake_adapter.py` | `FakePaymentAdapter` — used when `GATEWAY_SECRET` is absent |
| `app/payment/live_adapter.py` | `LivePaymentAdapter` stub — used when `GATEWAY_SECRET` is set |
| `app/payment/factory.py` | `get_payment_adapter()` — selects adapter based on environment |

### `app/payment/port.py`

Defines the interface all adapters must satisfy:

```
PaymentPort.charge(tenant_id, amount_eur_cents, description) -> ChargeResult
```

`ChargeResult` fields:

| Field | Type | Notes |
|---|---|---|
| `success` | `bool` | True when the gateway accepted the charge |
| `gateway_reference` | `Optional[str]` | Opaque gateway ID; `None` on failure |
| `error_message` | `Optional[str]` | Human-readable error; `None` on success |

### `app/payment/fake_adapter.py`

- Every charge succeeds immediately with no network call.
- Returns a reference prefixed `fake-` followed by 8 random hex bytes.
- Used in all tests and in local development when `GATEWAY_SECRET` is absent.

### `app/payment/live_adapter.py`

- Reads `GATEWAY_SECRET` from the environment once at construction time.
- Raises `RuntimeError` if the variable is absent or empty.
- The secret is stored in a private attribute and **never** appears in `repr()`,
  `str()`, log output, or any returned object.
- `charge()` delegates to `_call_gateway()`, which is a stub raising
  `NotImplementedError` until a gateway vendor is chosen.
- No Stripe key, no card number, and no private URL appears in the file.

### `app/payment/factory.py`

`get_payment_adapter()` is the single place that reads `GATEWAY_SECRET`:

| `GATEWAY_SECRET` | Adapter returned |
|---|---|
| Absent or empty | `FakePaymentAdapter` |
| Set to any non-empty value | `LivePaymentAdapter` |

The app remains runnable without the secret because the fake adapter handles
all calls locally.

### Updated: `tests/conftest.py`

`os.environ.pop("GATEWAY_SECRET", None)` is called at import time so that no
test ever accidentally activates the live adapter.

### Updated: `.env.example`

Added a commented-out `GATEWAY_SECRET` entry with an explanation of the
fake/live switch.  The actual value is never committed.

---

## Security properties

- The gateway secret is read from the environment only.
- It is never logged, printed, or included in any response or repr.
- No Stripe key, card number, or private URL appears anywhere in the repo.
- If the secret is absent the fake gateway is used and the app still runs
  (satisfies the commercial decision locked in DIRECTIVE.txt).

---

## Test results

| Run | Command | Result |
|---|---|---|
| New tests only | `python3 -m pytest tests/test_gateway.py -v` | 15 passed, 0 failed |
| Full suite | `python3 -m pytest tests/ -q` | 635 passed, 0 failed |

Test file: `tests/test_gateway.py`

Covers:
- `ChargeResult` dataclass: success fields, failure fields, defaults (3 tests).
- `FakePaymentAdapter`: charge succeeds, reference starts with `fake-`,
  no error message, distinct references per call, two tenants both succeed,
  implements `PaymentPort` (6 tests).
- `LivePaymentAdapter`: raises `RuntimeError` when secret absent,
  `charge()` raises `NotImplementedError` (stub), `repr()` does not contain
  the secret value (3 tests).
- Factory: returns fake when secret absent, returns live when secret present,
  fake satisfies `PaymentPort` (3 tests).

---

## External test script

`docs/EXTERNAL_TEST.sh` has not been run by a second account.
The row in `docs/TEST_HOUSE.md` is not updated until that run happens.

---

## App runnable

The application starts normally with or without `GATEWAY_SECRET`.
When the secret is absent, the fake adapter is used silently.
No migration was required for this phase (no new database table).

---

## Scope boundary

Phase 2 ends here. The gateway port exists and both adapters are tested.
Plan enforcement (Desk cannot call Amin, etc.) is Phase 3.
The operator surface is Phase 4.
