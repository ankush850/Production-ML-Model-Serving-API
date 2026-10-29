# Operations Runbook

Concrete playbooks for when something goes wrong. Each starts from an **observable symptom**,
not from an architecture diagram.

Rule that applies to all of them: **stabilise first, understand second.** Roll back, then
investigate — the artifact store keeps every version, so diagnosing later costs nothing you
cannot afford to spend now.

---

## Table of contents

- [Quick triage](#quick-triage)
- [Playbook 1: bad model deployed](#playbook-1-bad-model-deployed)
- [Playbook 2: service up but returns 503](#playbook-2-service-up-but-returns-503)
- [Playbook 3: drift alert](#playbook-3-drift-alert)
- [Playbook 4: latency regression](#playbook-4-latency-regression)
- [Playbook 5: everything is 422 / 400](#playbook-5-everything-is-422--400)
- [Playbook 6: training failed or produced a bad metric](#playbook-6-training-failed-or-produced-a-bad-metric)
- [Playbook 7: suspected artifact tampering](#playbook-7-suspected-artifact-tampering)
- [Escalation and evidence to collect](#escalation-and-evidence-to-collect)

---

## Quick triage

| Symptom | First check | Likely playbook |
|---|---|---|
| Predictions look wrong but return 200 | `curl /health` → is `model_version` the expected one? | [1](#playbook-1-bad-model-deployed) |
| `/health` → 503 `MODEL_NOT_LOADED` | startup log: `model load failed at startup: ...` | [2](#playbook-2-service-up-but-returns-503) |
| Drift CLI exited 1 | read `drifted: [...]` and each `psi` | [3](#playbook-3-drift-alert) |
| p95 climbing | `X-Process-Time` vs client-measured latency | [4](#playbook-4-latency-regression) |
| All/most requests 4xx | is it 422 or 400? they mean different things | [5](#playbook-5-everything-is-422--400) |
| Training errored or AUC jumped | `metrics.json` of the last good vs. current version | [6](#playbook-6-training-failed-or-produced-a-bad-metric) |
| Unexpected version or artifact size change | store write access | [7](#playbook-7-suspected-artifact-tampering) |

Useful one-liners:

```bash
python scripts/rollback.py --list          # what exists, and what LATEST points at
cat models/artifacts/models/LATEST         # same thing, local mode
curl -s localhost:8000/health              # what the running process actually loaded
```

Remember the resolution order: **explicit argument → `MODEL_VERSION` → `LATEST`.** A leftover
`MODEL_VERSION` env var silently overrides a rollback, and this is the single most common reason
"the rollback didn't take effect".

---

## Playbook 1: bad model deployed

**Symptom** — predictions degraded, AUC dropped in the offline eval, a client reports implausible
probabilities, and `/health` shows a version you did not intend to serve.

### 1. Confirm which version is live

```bash
curl -s http://127.0.0.1:8000/health
```

Note `model_version`. Compare against `python scripts/rollback.py --list`.

### 2. Roll back

```bash
python scripts/rollback.py --list
python scripts/rollback.py --version 20261029T121219Z     # the last known-good id
```

This rewrites the `LATEST` pointer only. Nothing is deleted; the bad version stays available for
post-incident analysis. Exit code `2` means the target version does not exist — re-check the id.

### 3. Restart the service (required)

The model is loaded once at startup; **a pointer move does not hot-swap it.**

```bash
# local / uvicorn: stop the process and start again
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# docker compose
docker compose restart api

# EC2: re-run the container, or trigger the deploy workflow again
```

### 4. Verify

```bash
curl -s http://127.0.0.1:8000/health    # model_version must now equal the rollback target
```

Do not close the incident until this matches. A restart that silently reloaded the bad version
(e.g. `MODEL_VERSION` still pinned in the environment) looks like a successful rollback.

### 5. Then diagnose

```bash
python -m json.tool models/artifacts/models/<bad-version>/metrics.json
```

Compare `roc_auc`, `n_train`, `n_test`, `split_cutoff`, `git_commit`, `sklearn_version` against
the known-good version. The usual culprits, in order of how often I would expect them:

- `git_commit` differs from what you meant to train from;
- `n_train`/`n_test` shifted a lot → the input data changed shape (a partial download, a new
  sheet, a schema rename);
- `split_cutoff` moved → the time split changed, so AUC is not comparable across versions;
- `sklearn_version` differs → unpickling a model trained on another sklearn version can behave
  differently or fail outright;
- `train_pos_rate` far from `test_pos_rate` → the target definition or the window is broken.

**Do not delete the bad version.** Immutability is what makes step 5 possible.

---

## Playbook 2: service up but returns 503

**Symptom** — the port accepts connections, `/health` returns 503 `MODEL_NOT_LOADED`, and every
predict call fails. The container may also be marked *unhealthy* (the healthcheck treats 503 as
failure on purpose, so a model-less container cannot pass as healthy).

The app deliberately survives a load failure at startup rather than crashing, so you get this
state instead of a crash loop.

### Diagnosis order

```bash
# 1. read the startup log — it names the actual cause
docker logs ml-serving-api 2>&1 | Select-String "model load failed"

# 2. does the version you expect actually exist?
python scripts/rollback.py --list

# 3. is a stale pin overriding LATEST?
echo "MODEL_SOURCE=$env:MODEL_SOURCE  MODEL_VERSION=$env:MODEL_VERSION"
```

| Log / observation | Cause | Fix |
|---|---|---|
| `LATEST not found; has any model been uploaded?` | Nothing trained in this store | `python -m src.training.train` |
| `version <id> not found under ...` | `MODEL_VERSION` pin points at a nonexistent version | Unset `MODEL_VERSION`, or set it to a real id |
| `FileNotFoundError: s3://.../LATEST not found` | Wrong `S3_BUCKET` / `S3_PREFIX` / `AWS_REGION` | Check all three; the prefix is stripped of slashes and must match exactly |
| `AccessDenied` from boto3 | Serving role lacks `s3:GetObject`/`ListBucket` | Fix the bucket policy. **Do not** grant write access to "just make it work" |
| `invalid version id ...; expected UTC timestamp` | `LATEST` corrupted or hand-edited | `python scripts/rollback.py --version <valid id>` |
| Paths look right but no artifacts | Container path mapping (`/app/models` → `/app/models/models`) | See [deployment.md §2](deployment.md#2-docker-compose) — `/app/models` is correct |
| `Unable to instantiate ... sklearn` on load | Artifact pickled with an incompatible sklearn | Retrain with the runtime's sklearn version |

### Recovery

Once the cause is fixed, restart and confirm `/health` returns 200 with the intended
`model_version`. If nothing is loadable and traffic must flow, roll back to any version that
does exist and load *that* — a slightly stale model beats no model.

---

## Playbook 3: drift alert

**Symptom** — `python -m src.training.drift ...` exited `1`, or a scheduled check flagged
features.

### 1. Read the report, do not react to the exit code

```bash
python -m src.training.drift --data data/raw/latest_window.csv --source local | tee drift.json
python -m json.tool <<< (Get-Content drift.json)   # or: jq .
```

Focus on `drifted`, then per feature: `psi`, `ks.statistic`, `null_shift`,
`new_or_missing_categories`.

Severity is not in the alert — it is in the magnitude. `psi: 0.21` and `psi: 6.90` both set
`has_drift: true` and are **not** the same incident. Rank before acting.

### 2. Rule out the boring explanations first

| Check | How |
|---|---|
| Did the *check* fail rather than drift occur? | Exit code `2` (not `1`) = missing CSV / unloadable artifact. A monitor failure is not a clean result and must not be reported as "no drift" |
| Is the reference from the right version? | `rollback.py --list`; reference stats come from the **training window** of the version under `--version` |
| Is the input the raw line-item CSV? | The CLI takes **raw** data and re-runs `build_features`. Passing a pre-engineered frame double-transforms it and produces nonsense PSI |
| Is this an expected seasonal change? | See the known `month` behaviour below |
| Is a data *pipeline* change masquerading as drift? | Check upstream schema renames / a new vendor file first |

### 3. Understand the two features that drift legitimately in this model

Measured on this dataset, the features that cross 0.2 in a later window are **`month`** and
**`days_since_last_order`** — and both are real, not noise:

- **`month`** — the training window covers Jul–Dec 2011 only. Months 1–6 have literally zero
  share in the reference, so when they appear in a new window PSI explodes (measured **6.90**).
  The actionable meaning is important, not subtle: **this model has never seen the first half of
  the calendar year and must not be trusted there.** Retrain over a full year.
- **`days_since_last_order`** — nulls encode "first order", which exists only in the earliest
  data. A later window legitimately has a lower null fraction (measured `null_shift` 0.1394) and
  a shifted recency distribution (PSI **0.73**). Usually expected; confirm the null rate did not
  change because of an upstream bug.

### 4. Do not retrain reflexively

Retraining on a drifted window *erases your ability to detect that drift* and can fit the
anomaly. Retrain only when you can also re-establish a clean, chronological holdout and re-report
`metrics.json`. If drift is seasonal and expected, the right fix is a **training window that
covers the seasons you deploy in**, not a monthly retrain chase.

### 5. Caveat you must know before trusting KS here

In measurements, **every KS `p_value` was `0.0` — including when comparing the reference window
against itself, where PSI ≈ 0.0004 proves there is no drift.** At tens of thousands of rows the KS
test rejects at differences far below anything meaningful, so the p-value carries no signal. The
**`D > 0.1` effect-size guard is what actually decides `ks_drift`.** Never page on a KS p-value
from this module. Details:
[drift-monitoring.md](drift-monitoring.md#known-weaknesses-read-this).

---

## Playbook 4: latency regression

**Symptom** — client-reported latency up, or the benchmark's p95 rose.

### 1. Separate server time from transport time

Every response carries `X-Process-Time` (seconds) — pure server handling. Benchmark output is
**client-side wall time**, which includes network, connection setup and queueing.

```bash
curl -s -D - -o NUL http://127.0.0.1:8000/health | Select-String -i "x-process-time"
```

| `X-Process-Time` | Benchmark p95 | Where the problem is |
|---|---|---|
| low | low | Not a problem |
| low | **high** | Network, LB, DNS, cold connections, client-side serialisation — **not the model** |
| **high** | **high** | In-process: model, CPU contention, worker saturation |
| high | low | Compare like with like; likely different measurement points |

### 2. Re-measure properly

```bash
python scripts/benchmark_latency.py --url http://127.0.0.1:8000 --n 1000 --warmup 20
```

Always use `--warmup` (default 20) — the first requests include lazy imports and OS page caching.
Check `errors=` too: a low p95 with 500s is not fast, it is failing.

Baseline from this repo (native local Uvicorn, 1000 requests, 0 errors):
**p50 5.5 ms · p95 7.0 ms · p99 8.1 ms · mean 5.7 ms · max 55.7 ms.**
Note `max` is ~8× `p99` — single-request outliers exist even in a healthy service, which is why
p95/p99 are the reported numbers and max is not.

### 3. Common causes

| Cause | Tell | Fix |
|---|---|---|
| Model loaded per worker, memory pressure | `max` spikes; degrades with worker count | Fewer workers, or a larger instance |
| Batch abuse | only slow under `/predict/batch` load | Lower `MAX_BATCH_SIZE`; one big batch occupies a worker and hurts other clients' p99 |
| Cold container / just restarted | first minute only | Wait for warmup; check `loaded_at` in `/health` |
| Cross-AZ or public-internet hop | `X-Process-Time` low, client latency high | Same-AZ placement, private connectivity |
| Noisy neighbour on a small instance | erratic p95 | Larger instance or a rescheduled run |
| GC / CPU throttling (containers) | sawtooth pattern | Raise CPU limits; check throttling metrics |
| Genuinely heavier model | `model_type` changed in `metrics.json` | Expected after moving LR→GB; size accordingly |

---

## Playbook 5: everything is 422 / 400

**Symptom** — a client starts failing en masse on validation.

The status code tells you which side is broken. **422 is a client bug. 400 `SCHEMA_VIOLATION`
means the model's trained domain and the client's data have diverged** — usually a shared bug.

### If 422 `VALIDATION_ERROR`

Read `details[].field` and `issue`. Frequent causes:
- `"month": 9` instead of `"9"` — **`month` and `day_of_week` are strings**;
- missing a required key (unknown fields are also rejected, so a renamed feature is a 422);
- `order_hour: 24`, or a negative `total_quantity`;
- sending a raw line item instead of an invoice aggregate, so required keys are absent.

This is a client-side fix. `/docs` shows the schema with a working example.

### If 400 `SCHEMA_VIOLATION`

`issue` reads like `above schema maximum 87167.0`, `below schema minimum 6.0`, or
`unknown category 'NowhereLand'`. Check `schema.json` for the live version:

```bash
python -m json.tool models/artifacts/models/<version>/schema.json
```

Real bounds from this dataset and the traps they create:

| Feature | Trained range | Incident it can cause |
|---|---|---|
| `order_hour` | **6.0 – 20.0** | Retail opening hours only. A legitimate **08:00 online** or **21:00** order is rejected |
| `total_quantity` | 1.0 – 87,167.0 | Legitimate, but a unit-scale bug (grams vs. tonnes) passes silently and scores garbage |
| `country` | 43 explicit values | A **new market is a 400**, not a warning. Expanding sales breaks the API |
| `days_since_last_order` | 0 – 714 | A long-dormant customer beyond ~2 years is rejected |

Decide which of the two it is:
- **Client/unit bug** → fix the producer. Do not "fix" it by widening the schema.
- **Business reality moved** (new country, new trading hours) → this is a genuine **retrain
  trigger**. Consider mapping unseen categories to `null` in the client as a stopgap, since the
  pipeline imputes a `MISSING` category.

⚠️ Do **not** work around a 400 by loosening `schema.json` by hand. It is a projection of the
fitted pipeline; editing the JSON does not change what the model can score, and you would create
a schema that lies about the artifact.

---

## Playbook 6: training failed or produced a bad metric

`train.py` refuses to publish a model it cannot evaluate.

| Message / symptom | Meaning | Action |
|---|---|---|
| `Validation failed: ...` (from `validate`) | Input is structurally wrong | Fix the data, not the validator |
| Class-poor split / AUC undefined | The held-out window has too few positives to compute ROC-AUC | Check `split_cutoff` and the window; do not "fix" by lowering the requirement |
| `roc_auc` far above expectation (e.g. > 0.95) | **Almost certainly leakage**, not a great model | Verify `time_split` is strictly chronological and the imputer/scaler fit on train only |
| `roc_auc` at/below ~0.5 | No signal, or features/target misaligned | Recompute `build_features`; check the 30-day window and that the target is not constant |
| All three candidates similar and low | Features are not informative for this target | Improve features, or accept that this task/window is hard |
| `ModuleNotFoundError: No module named 'src'` | Script run in an unsupported way | Use `python -m src.training.train`; `scripts/rollback.py` bootstraps `sys.path` itself |
| Download fails / unexpected columns | Upstream renamed headers again | `download_data.py` maps `Invoice`/`Price`/`Customer ID`; extend `COLUMN_RENAME` |

### Before publishing a suspicious model

The right guard is *not* a threshold on AUC alone. Compare versions:

```bash
python -m json.tool models/artifacts/models/<new>/metrics.json
python -m json.tool models/artifacts/models/<prev>/metrics.json
```

Sanity checklist: `n_train + n_test` ≈ expected invoice count; `split_cutoff` between the
previous training end and the new data; `train_pos_rate` ≈ `test_pos_rate` (large gaps mean the
target is unstable); `git_commit` is what you think; `sklearn_version` matches the serving image.

A large AUC *drop* is suspicious of data change. A large AUC *rise* is suspicious of **you** —
leakage is far more common than a breakthrough.

---

## Playbook 7: suspected artifact tampering

**Why this is a real scenario:** `joblib.load()` executes arbitrary code from the artifact.
Anyone with **write** access to `<prefix>/models/<version>/model.joblib` can run code inside the
API process, with that process's environment and credentials. This is the highest-severity issue
in the project and is documented in
[architecture.md](architecture.md#security-model-and-its-limits) rather than assumed away.

### Indicators

| Signal | Check |
|---|---|
| `list_versions` shows a version you did not train | `python scripts/rollback.py --list` vs. your CI run history |
| `LATEST` changed without a deploy | `metrics.json` `created_at` / `git_commit` for the current target |
| Artifact size changed for an *existing* version | Should be **impossible** — immutability raises `FileExistsError`. If it happened, write permissions are wrong |
| Unexpected `sklearn_version`, or new unread keys in `metrics.json` | `python -m json.tool .../metrics.json` |
| S3 access log shows a `PutObject` on a version prefix | CloudTrail / S3 server access logs |

### Response

1. **Repoint `LATEST` to a version you trust** and restart (Playbook 1). Trust is established by
   `git_commit` + `created_at` matching a known CI run, not by the version existing.
2. **Revoke write access immediately.** Confirm the serving role has only `Get`/`List`. A serving
   role with `PutObject` is the root cause, not a contributing factor.
3. **Treat the process as compromised**, not just the model: code ran with its credentials. Rotate
   whatever the container could read.
4. Preserve the suspicious artifact for analysis. **Do not load it** — loading is the exploit.
5. Report the `request_id`s and the version id. `metrics.json` for every version is intact and
   immutable, which is precisely what makes this reconstruction possible.

### Prevention

- Serving principal: `s3:GetObject` + `s3:ListBucket` **only**. Write access restricted to the
  training/CI role.
- Version immutability in the application layer (implemented) plus **S3 Object Lock / a deny-delete
  bucket policy** for WORM behaviour (not implemented — recommended).
- Not yet implemented, needed for real production: **artifact signing**, verified at load time.
  See [architecture.md](architecture.md#security-model-and-its-limits).

---

## Escalation and evidence to collect

For any incident, capture these **before** restarting repeatedly:

```bash
curl -s http://127.0.0.1:8000/health                      # version, loaded_at, source
python scripts/rollback.py --list                          # full version timeline + LATEST
python -m json.tool models/artifacts/models/<v>/metrics.json
docker logs --tail 200 ml-serving-api 2>&1 | Select-String "request_id=<the id>"
python -m src.training.drift --data data/raw/recent.csv --source local   # exit code + report
```

One `request_id` per request in the structured access log, echoed as `X-Request-ID` and embedded
in 500 responses (`details[0].issue = "request_id=..."`). Server-side exception detail is
deliberately **not** returned to clients, so **the request id is the only join key between a
customer complaint and the traceback.** Ask for it first, and grab the logs before any restart —
container restarts discard them.

**Do not delete versions to "clean up" during an incident.** You lose the comparison you need.
Archive them afterwards instead ([model-versioning.md](model-versioning.md#retention-and-lifecycle)).

Related: [runbook vs. architecture](architecture.md) · [deployment.md](deployment.md) ·
[drift-monitoring.md](drift-monitoring.md) · [model-versioning.md](model-versioning.md)
