# Model Versioning & Storage

The model registry for this project. Everything about how a trained model becomes a
deployable, immutable, addressable artifact — and how production moves between them.

Implemented in `src/storage/s3_store.py`.

---

## Table of contents

- [The contract](#the-contract)
- [Layout](#layout)
- [Version ids](#version-ids)
- [Artifact files](#artifact-files)
- [Immutability](#immutability)
- [The LATEST pointer](#the-latest-pointer)
- [Rollback](#rollback)
- [Two implementations, one interface](#two-implementations-one-interface)
- [Operating the store](#operating-the-store)
- [Retention and lifecycle](#retention-and-lifecycle)

---

## The contract

`ModelStore` exposes five operations:

| Method | Signature | Guarantees |
|---|---|---|
| `upload_version` | `(version: str, files: dict[str, bytes]) -> str` | Creates a version. Raises `FileExistsError` if it already exists |
| `list_versions` | `() -> list[str]` | All version ids, **sorted oldest → newest** |
| `get_latest` | `() -> str` | The version id `LATEST` points at; validates the format |
| `set_latest` | `(version: str) -> None` | Repoints `LATEST`. Raises `FileNotFoundError` for an unknown version |
| `download_version` | `(version: str, dest_dir: Path) -> Path` | Materialises the artifact files locally; raises if none exist |

Two module-level helpers:

- `new_version_id(dt=None)` → UTC timestamp string for a fresh version.
- `get_store(source=None)` → factory returning `S3ModelStore` or `LocalModelStore` based on
  `MODEL_SOURCE` (or an explicit override argument).

Validation is centralised in the base class, so both implementations enforce identical rules.
Version format and artifact names are checked before any bytes are written.

```
upload_version("20261029T145410Z", {"model.joblib": b"..."})
  └─ _check_upload      → VERSION_RE match? artifact name in ARTIFACT_FILES?
  └─ existence check    → version already in list_versions()? → FileExistsError
  └─ _write_version     → implementation-specific write
```

## Layout

```
s3://<S3_BUCKET>/<S3_PREFIX>/models/
├── 20261029T121219Z/
│   ├── model.joblib
│   ├── metrics.json
│   ├── schema.json
│   └── reference_stats.json
├── 20261029T145410Z/
│   ├── model.joblib
│   ├── metrics.json
│   ├── schema.json
│   └── reference_stats.json
└── LATEST                          # plain text, one version id
```

Key construction: `f"{S3_PREFIX}/models/{version}/{filename}"` (the prefix is normalised by
stripping slashes, so `foo/` and `foo` behave identically).

Local mode mirrors this exactly under `LOCAL_MODEL_DIR`:

```
models/artifacts/
└── models/
    ├── 20261029T145410Z/…
    └── LATEST
```

> **Real gotcha:** `LocalModelStore(root)` appends a `models/` level itself. Passing
> `root=models/artifacts` means artifacts actually live at `models/artifacts/models/<VERSION>/`.
> Tests must read from `store.root`, not the directory they passed in. This is intentional
> (both backends then share the `models/` subtree) but it surprises everyone once.

## Version ids

Format: **`YYYYMMDDTHHMMSSZ`** — a UTC timestamp, enforced by
`VERSION_RE = ^\d{8}T\d{6}Z$`.

```
20261029T145410Z
└┬──────┘└┬───┘└┬┘
 date    time   UTC
```

Why a timestamp and not a counter:
- **Globally unique with no coordination.** Two CI runners training simultaneously cannot
  collide, so no lock or sequence table is required.
- **Self-ordering.** Lexicographic sort equals chronological sort, which is why
  `list_versions()` is just `sorted(...)`.
- **Self-describing.** `20261029T145410Z` in a log line tells you when it was built; `v37` does not.

Costs:
- Two runs inside the same second would collide (second resolution). In practice training takes
  minutes, so this is not a real constraint — and if it ever were, failing loudly is better than
  silently versioning two different models as one.
- No semantic information (no "this was a canary", no "this used features v2"). Encode that in
  `metrics.json` if needed.

Both `get_latest()` and `set_latest()` re-validate the format on read **and** write, so a
hand-edited or corrupted `LATEST` file is caught rather than propagated.

## Artifact files

Exactly four names are permitted — `ARTIFACT_FILES`. Anything else raises
`ValueError: unknown artifact names`.

| File | Contents | Written by | Read by |
|---|---|---|---|
| `model.joblib` | The complete fitted `sklearn.Pipeline` (preprocessor + estimator, ~288 KB) | `train.py` | `model_loader.py` |
| `metrics.json` | `model_type`, `roc_auc`, `candidate_roc_auc`, `n_train`, `n_test`, `train_pos_rate`, `test_pos_rate`, `features`, `sklearn_version`, `split_cutoff`, `git_commit`, `created_at`, `version` | `train.py` | humans, CI gates, this doc |
| `schema.json` | `numeric` (dtype/min/max), `categorical` (dtype/allowed values), `target` | `train.py` | `/predict` out-of-domain checks |
| `reference_stats.json` | `numeric` (quantiles / null_fraction / n / **bins** / **expected**), `categorical` (population shares) | `train.py` | `drift.py` |

Actual sizes for version `20261029T145410Z`: `model.joblib` 287,579 B ·
`reference_stats.json` ~4.2 KB · `schema.json` ~2.0 KB · `metrics.json` 732 B.

**`schema.json` is a readable projection, not a second source of truth.** The live transform
state is inside the pipeline. `schema.json` exists so the API can reject out-of-domain input
without unpickling anything, and so a human can see the trained bounds. If they ever diverge,
the pipeline wins — it is what actually scores.

Real `schema.json` excerpt:

```json
{
  "numeric": {
    "total_quantity":          { "dtype": "float", "min": 1.0,   "max": 87167.0 },
    "n_line_items":            { "dtype": "float", "min": 1.0,   "max": 1350.0 },
    "days_since_last_order":   { "dtype": "float", "min": 0.0,   "max": 714.0 },
    "order_hour":              { "dtype": "float", "min": 6.0,   "max": 20.0 }
  },
  "categorical": { "country": { "dtype": "string", "allowed": ["Australia", "..."] } },
  "target": "reorder_30d"
}
```

Note `order_hour` is bounded **6–20** because the 2010–2011 retail data only contains trading
hours. A legitimate 8 a.m. online order would be rejected as out of domain. That is the honest
consequence of validating against what the model saw rather than what is physically possible —
and it is a concrete reason to retrain on a more representative window.

## Immutability

**A published version can never be overwritten.**

```python
def upload_version(self, version, files):
    self._check_upload(version, files)
    if version in self.list_versions():
        raise FileExistsError(f"version {version} already exists; versions are immutable")
    self._write_version(version, files)
    return version
```

Enforced in the shared base class, so `S3ModelStore` and `LocalModelStore` cannot differ. The
earlier draft of this code checked existence only in one implementation, which is exactly the
class of bug that lets a bad artifact silently replace a good one; hoisting it into the base
made it impossible to forget.

Why this is non-negotiable: `model_version` appears in every API response and every
`metrics.json`. If bytes behind a version id could change, then
- a latency/quality regression could not be attributed to a version,
- a rollback might not restore what previously ran,
- and "reproduce last Tuesday's predictions" would be impossible.

Consequence to be aware of: a **partially failed upload** leaves an incomplete version that
blocks re-uploading the same id. The remedy is to let the next attempt take a fresh timestamp,
not to force the write. If cleanup is needed, delete the incomplete prefix deliberately.

## The LATEST pointer

`LATEST` is a one-line object holding a version id. It is the **only mutable thing** in the
store, and that asymmetry is the design.

```
$ cat models/artifacts/models/LATEST
20261029T145410Z
```

Separating *newest* from *deployed* is what makes deployment a decision rather than a side
effect. If `LATEST` meant "the newest key", then:
- rollback would require deleting the newer artifact (destroying the evidence you need for
  post-incident review), or special-casing your own read path;
- promoting a vetted model would be impossible — an untested training run would deploy itself
  the moment it finished.

`set_latest` refuses unknown versions:

```python
if version not in self.list_versions():
    raise FileNotFoundError(f"cannot point LATEST at unknown version {version}")
```

so a typo in `--version` cannot point production at nothing.

**Resolution order in the serving layer** (`src/api/model_loader.py`):

1. explicit `version` argument →
2. else `MODEL_VERSION` env var, if non-empty →
3. else `store.get_latest()`.

Step 2 is the pinning mechanism: it bypasses `LATEST` for canaries, forensics, or reproducing a
specific customer report.

## Rollback

Rollback is **one pointer write**. No retraining, no artifact movement, no redeploy of new code.

```bash
python scripts/rollback.py --list                       # see what exists
python scripts/rollback.py --version 20261029T121219Z   # repoint LATEST
```

Real output of `--list`:

```
available versions (oldest -> newest):
  20261029T121219Z
  20261029T145410Z
LATEST -> 20261029T145410Z
```

The script prints `LATEST <previous> -> <new>`, exits `0` on success and `2` on an unknown
version or bad input, so it is safe to wrap in automation.

One-step is the supported semantic: you can always go back to the previously known state.
Because every version is retained and immutable, you can in practice jump to *any* older
version — the store does not restrict the target, only that it must exist.

**Rollback takes effect on the next service restart.** The version is resolved during
`lifespan` startup and held in memory; there is no hot-swap. Restart cost is therefore the
rollback RTO (one artifact download plus one `joblib.load`). See
[runbook.md](runbook.md#playbook-1-bad-model-deployed).

Verification after any pointer move:

```bash
curl http://127.0.0.1:8000/health     # model_version must be the rollback target
```

## Two implementations, one interface

| | `S3ModelStore` | `LocalModelStore` |
|---|---|---|
| Backend | boto3 S3 client | filesystem under `root/models/` |
| `list_versions` | `list_objects_v2` paginator with `Delimiter="/"`, keeping keys matching `VERSION_RE` | `iterdir()` over subdirectories matching `VERSION_RE` |
| `get_latest` | `GetObject` on the `LATEST` key | read `LATEST` file; custom message if absent |
| `_write_version` | `put_object` per artifact | `mkdir` + `write_bytes` per artifact |
| `download_version` | fetches each `ARTIFACT_FILES` entry, skipping absent ones; raises if nothing was written | `shutil.copytree` unless source and destination resolve to the same path |

Design points:
- `list_versions` uses `Delimiter="/"` and reads `CommonPrefixes`, so it costs one paginated
  `LIST` regardless of how many artifacts sit inside each version.
- `bucket` is a no-op in `LocalModelStore` (hardcoded `"local"`); the `prefix` slot carries the
  root directory. Deliberate, so the base-class key logic is shared rather than duplicated.
- The `ModelStore` base is also usable as a **test double** — inject any object satisfying the
  five methods to exercise `model_loader` without a network or a filesystem.
- `MODEL_SOURCE=local` requires **no AWS credentials at all**, which is what lets CI and a
  laptop run the full versioning story.

`download_version` skips missing artifact files instead of failing, because only
`model.joblib` is strictly required to serve; `metrics.json` and `reference_stats.json` are not
needed on the request path. It raises only if *nothing* could be fetched.

## Operating the store

```bash
# train and publish (creates a version, sets LATEST, uploads to S3 if MODEL_SOURCE=s3)
python -m src.training.train

# list
python scripts/rollback.py --list

# rollback / promote
python scripts/rollback.py --version <YYYYMMDDTHHMMSSZ>

# serve a pinned version instead of LATEST
$env:MODEL_SOURCE = "s3"; $env:MODEL_VERSION = "20261029T121219Z"
python -m uvicorn src.api.main:app --port 8000
```

With real S3, the process needs `s3:ListBucket`/`s3:GetObject`/`s3:PutObject` on
`<prefix>/models/*` for training, and **only** `ListBucket` + `GetObject` for serving. Granting
the serving role write access is what turns a stolen container into a model-replacement attack —
see [architecture.md](architecture.md#security-model-and-its-limits).

## Retention and lifecycle

Versions accumulate on every training run (roughly one per retrain, since ids are timestamps).
The `model.joblib` files are small (~288 KB), so storage cost is negligible; the operational
risk is *finding the right one* among hundreds.

Suggested policy:
- Keep every version referenced by a release or incident.
- Move versions older than N months to `GLACIER`/`GLACIER_IRS` via an S3 lifecycle rule —
  they stay restorable but stop cluttering hot storage.
- **Do not** let a lifecycle rule expire `LATEST` (it is a tiny object under the same prefix;
  scope expiration by the timestamped "folder" pattern).
- Never delete the version currently named by `LATEST`.

Related: [architecture.md](architecture.md) · [drift-monitoring.md](drift-monitoring.md) ·
[runbook.md](runbook.md)
