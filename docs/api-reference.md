# API Reference

Service: **Production ML Model Serving API** · OpenAPI title version `1.0.0`
Interactive docs: `GET /docs` (Swagger UI) and `GET /openapi.json`.

All examples below were run against a live local process serving model version
`20261029T121219Z`, so the responses are **actual output**, not illustrations.

---

## Table of contents

- [Conventions](#conventions)
- [GET /health](#get-health)
- [POST /predict](#post-predict)
- [POST /predict/batch](#post-predictbatch)
- [Feature reference](#feature-reference)
- [Error model](#error-model)
- [Error code catalogue](#error-code-catalogue)
- [Middleware and response headers](#middleware-and-response-headers)
- [Decision threshold](#decision-threshold)
- [Client guidance](#client-guidance)

---

## Conventions

| | |
|---|---|
| Content type | `application/json` (UTF-8) |
| Auth | **none** — terminate authentication at a gateway/load balancer in front of this service |
| TLS | **not handled here** — run behind an ALB/nginx with a certificate |
| Field naming | `snake_case` throughout |
| Version echo | every successful response includes `model_version`, so a client can prove which model answered |
| Unknown fields | rejected by Pydantic (default behaviour) — a typo in a feature name is a 422, not a silently ignored key |

## GET /health

Liveness **and** readiness. Distinguishing "process is up" from "process can predict" lets an
orchestrator hold traffic until the model is actually loaded.

```bash
curl http://127.0.0.1:8000/health
```

**200 OK** — real response:

```json
{
  "status": "ok",
  "model_version": "20261029T121219Z",
  "loaded_at": "2026-10-29T13:31:28.019445+00:00",
  "model_source": "local"
}
```

**503 Service Unavailable** — service reachable, no model:

```json
{
  "error_code": "MODEL_NOT_LOADED",
  "message": "model failed to load or is not configured",
  "details": []
}
```

| Field | Type | Meaning |
|---|---|---|
| `status` | string | Always `"ok"` on 200 |
| `model_version` | string | Version id `YYYYMMDDTHHMMSSZ` currently serving |
| `loaded_at` | string (ISO-8601, UTC) | When the pipeline was deserialised at startup |
| `model_source` | string | `local` or `s3` |

> The app **starts successfully even if loading fails**. It logs the loader error and then
> answers 503 on every predictive route. Check the startup log line
> `model load failed at startup: ...` rather than assuming a booting container is healthy.

## POST /predict

Predict whether the customer behind this invoice places another order within 30 days.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "total_quantity": 12,
      "n_line_items": 3,
      "mean_unit_price": 4.16,
      "order_value": 49.92,
      "days_since_last_order": 21.0,
      "order_hour": 10.0,
      "country": "United Kingdom",
      "month": "9",
      "day_of_week": "2"
    }
  }'
```

**200 OK** — real response:

```json
{
  "prediction": 0,
  "probability": 0.48997093038044415,
  "model_version": "20261029T121219Z"
}
```

| Field | Type | Meaning |
|---|---|---|
| `prediction` | integer | `0` = no reorder within 30 days, `1` = reorder |
| `probability` | float | `P(reorder within 30 days)`, in `[0, 1]` |
| `model_version` | string | Version that produced this answer |

Request body is `{"features": { ... }}` — the feature object is **nested**, not top-level.

## POST /predict/batch

Body is a **bare JSON array** of feature objects (no `features` wrapper).

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '[
    {"total_quantity":12,"n_line_items":3,"mean_unit_price":4.16,"order_value":49.92,
     "days_since_last_order":21.0,"order_hour":10.0,"country":"United Kingdom",
     "month":"9","day_of_week":"2"},
    {"total_quantity":1,"n_line_items":1,"mean_unit_price":9.9,"order_value":9.9,
     "days_since_last_order":null,"order_hour":14.0,"country":"France",
     "month":"3","day_of_week":"5"}
  ]'
```

**200 OK** — real response:

```json
{
  "results": [
    { "prediction": 0, "probability": 0.48997093038044415, "model_version": "20261029T121219Z" },
    { "prediction": 0, "probability": 0.3355420546453075, "model_version": "20261029T121219Z" }
  ],
  "model_version": "20261029T121219Z",
  "size": 2
}
```

Rules:
- **Empty array** → `400 EMPTY_BATCH`.
- **More than `MAX_BATCH_SIZE` records** (default 100) → `400 BATCH_TOO_LARGE`.
- **Validation is all-or-nothing**: every record is schema-checked first, and if any record is
  invalid nothing is scored. Each problem is prefixed with its zero-based array index so the
  caller can repair the payload in one round trip:

```json
{
  "error_code": "SCHEMA_VIOLATION",
  "message": "some records violate the model schema",
  "details": [
    { "field": "[1].country", "issue": "unknown category 'NowhereLand'" }
  ]
}
```

## Feature reference

Every field is validated twice: once by Pydantic (type + hard bounds) and once against the
**trained version's** `schema.json` (observed domain). The second layer catches values that are
plausible in the abstract but that the model never encountered.

| Field | Type | Required | Pydantic constraint | Nullable | Notes |
|---|---|---|---|---|---|
| `total_quantity` | float | ✅ | `>= 1` | ❌ | Sum of item quantities on the invoice |
| `n_line_items` | float | ✅ | `>= 1` | ❌ | Distinct line items on the invoice |
| `mean_unit_price` | float | ✅ | `>= 0` | ❌ | Mean unit price across line items |
| `order_value` | float | ✅ | `>= 0` | ❌ | `Σ quantity × price` |
| `days_since_last_order` | float | — | `0 … 3650` | ✅ | **`null` for a customer's first order** |
| `order_hour` | float | ✅ | `0 … 23` | ❌ | Hour the invoice was raised |
| `country` | string | — | `max length 64` | ✅ | Must be a category seen in training |
| `month` | string | ✅ | — | ❌ | `"1"` … `"12"` — **as a string** |
| `day_of_week` | string | ✅ | — | ❌ | `"0"` (Mon) … `"6"` (Sun) — **as a string** |

Gotchas that will bite a hand-written client:

- **`month` and `day_of_week` are strings, not integers.** Sending `"month": 9` is coerced, but
  the value is compared against string categories from training. Send `"9"`.
- **`days_since_last_order` may be `null`** and should be for first-time buyers. Substituting
  `0` asserts "this customer ordered today"; substituting `9999` asserts "dormant for 27 years".
  Both are lies the model will faithfully use.
- **`country` may be `null`** (the imputer learned a `MISSING` category). An *unseen* country
  string is a 400, but `null` is fine.
- **Numeric fields are floats.** Integers are accepted; strings that are not numeric are not.
- The **key order of your JSON is irrelevant** — columns are re-ordered server-side from the
  artifact's own feature list, never from the request.

## Error model

Uniform envelope for **every** non-2xx response:

```json
{
  "error_code": "STRING_CONSTANT",
  "message": "human-readable summary",
  "details": [
    { "field": "path.to.field", "issue": "what is wrong" }
  ]
}
```

- Branch on `error_code`, never on `message` (messages are for humans and may change).
- `details` is always an array; it may be empty (e.g. `EMPTY_BATCH`).
- `field` is dot-joined with the `body` prefix stripped for single-record requests, and
  index-prefixed (`[i].field`) for batch.
- A 500's `details[0].issue` carries `request_id=<id>` — the server-side log has the traceback,
  so quote that id when reporting an incident. Internal error text is deliberately **not**
  leaked to clients.

## Error code catalogue

| HTTP | `error_code` | Raised when | Client remedy |
|---|---|---|---|
| **422** | `VALIDATION_ERROR` | Body failed Pydantic: missing required field, wrong type, outside `ge`/`le`, unknown category length violation | Fix the payload. `details[].field` names the exact key |
| **400** | `SCHEMA_VIOLATION` | Payload is well-formed but outside the trained domain: numeric below/above the schema min/max, or a category the model never saw | Check whether your feature pipeline diverged from training; for unseen categories, retrain or map to `null` |
| **400** | `EMPTY_BATCH` | `/predict/batch` with `[]` | Do not call the endpoint with nothing; treat as a no-op |
| **400** | `BATCH_TOO_LARGE` | Batch length > `MAX_BATCH_SIZE` | Chunk the request |
| **503** | `MODEL_NOT_LOADED` | Startup load failed or was misconfigured | **Retry-safe**: the service is up but cannot serve. Inspect `MODEL_SOURCE` / `MODEL_VERSION` / store permissions |
| **500** | `INTERNAL_ERROR` | Unhandled exception | Do not hammer retries. Report the `request_id` from `details[0].issue` |

Two distinct "your input is wrong" statuses is intentional:

- **422** = *malformed* — the request never reached model logic; any version would reject it.
- **400 SCHEMA_VIOLATION** = *well-formed but out of domain* — this specific trained version
  refuses it. A retrain can turn a 400 into a valid request; a client fix cannot.

Real measured 422 (`total_quantity: -5`):

```json
{
  "error_code": "VALIDATION_ERROR",
  "message": "request body failed validation",
  "details": [
    { "field": "features.total_quantity", "issue": "Input should be greater than or equal to 1" }
  ]
}
```

## Middleware and response headers

`RequestContextMiddleware` wraps every route.

| Header | Direction | Behaviour |
|---|---|---|
| `x-request-id` | request (optional) | Echoed if supplied, else a 12-char hex id is generated |
| `X-Request-ID` | response | The id used for this request |
| `X-Process-Time` | response | Server handling time in **seconds**, 4 decimals (e.g. `0.0002`) |

Structured access log per request (real lines emitted by the middleware, on stderr):

```
2026-10-29 20:14:33,114 INFO api request_id=c627e9b6f7ee method=GET path=/health status=200 elapsed=0.0002
2026-10-29 20:13:48,157 INFO api request_id=8f5bebeecb51 method=POST path=/predict/batch status=200 elapsed=0.0146
2026-10-29 20:14:33,102 INFO api request_id=9127b6d16b9e method=POST path=/predict/batch status=400 elapsed=0.0007
```

Uvicorn emits its own access line per request in parallel, so each call appears twice in the
container log — once from `api` (with `request_id`) and once from uvicorn (with the peer
address). Correlate on `X-Request-ID`, and note that these records go to **stderr**, which
matters for log collection in Docker/CloudWatch.

Practical uses:
- Propagate a correlation id from your front end by **sending** `x-request-id`; the service
  reuses it instead of minting a new one.
- `X-Process-Time` excludes network transfer, so a large gap between it and your client-side
  latency means the time is going to the network or queueing, not to the model. Note the
  benchmark script measures **client-side** wall time, which is the larger number.

## Decision threshold

```
prediction = 1  if  probability >= 0.5  else  0
```

The threshold is **fixed at 0.5 and not configurable per request.** The class-1 rate in the
held-out window was 0.441, so 0.5 is a reasonable default, but retention economics are
asymmetric: a missed reorder (false negative) usually costs more than an unnecessary campaign
(false positive). If your business case needs a different operating point, use `probability`
and threshold client-side rather than asking this service to change it — or expose it as a
query parameter, which is a deliberate design change, not a config flag.

## Client guidance

**Retries.** `GET /health` and `POST /predict` are idempotent — safe to retry. Retry 503 and
network errors with exponential backoff; **do not** retry 400/422 (they are deterministic —
retrying a bad payload a hundred times yields a hundred identical 400s). Retry 500 at most once.

**Freshness.** Compare `model_version` in responses against what you expect. Silently serving a
rolled-back version is a real incident class; the field exists so you can detect it.

**Readiness gating.** Do not send traffic on TCP connect alone. Wait for `/health` 200, which
also confirms which version is live.

**Feature construction.** Compute the nine features the same way training did
(`src/data/preprocess.py`). Feeding a raw line-item row where an invoice aggregate is expected
will produce a *valid* response with meaningless probability — the worst failure mode, because
nothing errors.

**Batch sizing.** Start well under `MAX_BATCH_SIZE` and watch `X-Process-Time` before raising it;
one large batch holds a worker and shows up in other clients' p99.

Next: [model-versioning.md](model-versioning.md) — how `model_version` gets its value.
