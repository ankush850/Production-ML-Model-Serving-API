# Setup & Configuration

---

## Table of contents

- [Prerequisites](#prerequisites)
- [Install](#install)
- [Makefile targets](#makefile-targets)
- [Configuration reference](#configuration-reference)
- [Dataset](#dataset)
- [Task definition](#task-definition)
- [Run order](#run-order)
- [Windows / PowerShell notes](#windows--powershell-notes)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

| Requirement | Version | Verified in this repo |
|---|---|---|
| Python | 3.11 (3.11.0 used) | ✅ |
| pip | current | ✅ |
| Git | 2.x | ✅ (2.55) |
| Docker | 20.10+ | only for `docker build` / compose |
| ~2 GB free disk | — | dataset cache (~44 MB xlsx + ~60 MB csv) + artifacts |

No AWS account, credentials, or network access to AWS is required for development or for the
test suite. S3 is mocked with `moto`.

## Install

```bash
git clone https://github.com/ankush850/Production-ML-Model-Serving-API.git
cd Production-ML-Model-Serving-API

python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements-dev.txt
copy .env.example .env          # cp on macOS/Linux
```

`requirements.txt` is the runtime set (pandas, scikit-learn, FastAPI, uvicorn, pydantic v2,
pydantic-settings, boto3, joblib, httpx). `requirements-dev.txt` adds pytest, pytest-cov,
`moto[server]`, ruff and httpx-test tooling.

### Verifying the install

```bash
python -m pytest -q --no-cov
```

Expect `54 passed`. This needs no dataset, no model and no network — it runs entirely on
seeded synthetic fixtures and a mocked S3 server.

## Makefile targets

| Target | Runs |
|---|---|
| `make install` | `pip install -r requirements.txt` |
| `make install-dev` | runtime + dev requirements |
| `make download` | `python scripts/download_data.py` |
| `make train` | `python -m src.training.train` |
| `make serve` | uvicorn on `0.0.0.0:8000` |
| `make test` | `pytest -q --cov=src --cov-report=term-missing` |
| `make lint` | `ruff check` then `ruff format` over `src tests scripts` |
| `make docker-build` | `docker build -t ml-serving-api:latest .` |
| `make docker-run` | `docker compose up --build` |
| `make benchmark` | `python scripts/benchmark_latency.py` |
| `make rollback VERSION=<id>` | `python scripts/rollback.py --version <id>` |

On Windows without `make`, run the printed command directly — every target is a single line.

## Configuration reference

All settings live in `src/config.py` as a `pydantic-settings` model, read from environment
variables (or `.env`). Nothing is hardcoded; unset values fall back to the defaults shown.

| Env var | Default | Used by | Notes |
|---|---|---|---|
| `MODEL_SOURCE` | `local` | loader, storage, drift, rollback | `local` \| `s3`. Anything else raises `ValueError` |
| `MODEL_VERSION` | *(empty)* | `model_loader.py` | Empty ⇒ follow the `LATEST` pointer. Set to pin a specific version |
| `LOCAL_MODEL_DIR` | `models/artifacts` | `LocalModelStore` | The store adds a `models/` level: real path is `<LOCAL_MODEL_DIR>/models/<VERSION>/` |
| `S3_BUCKET` | `ml-models-bucket` | `S3ModelStore` | Must already exist; the app never creates buckets |
| `S3_PREFIX` | `production-ml-serving` | `S3ModelStore` | Keys become `<prefix>/models/<VERSION>/<file>` |
| `AWS_REGION` | `us-east-1` | boto3 client | Must match the bucket's region |
| `AWS_ACCESS_KEY_ID` | *(none)* | boto3 | Prefer an IAM role / instance profile over static keys |
| `AWS_SECRET_ACCESS_KEY` | *(none)* | boto3 | Same |
| `RAW_DATA_PATH` | `data/raw/online_retail.csv` | `train.py` | Written by `download_data.py` |
| `ARTIFACT_DIR` | `models/artifacts` | `train.py` | Staging directory for artifact bytes before upload |
| `MAX_BATCH_SIZE` | `100` | `/predict/batch` | Requests larger than this return 400 `BATCH_TOO_LARGE` |

Settings are cached via `lru_cache`. Tests that change the environment must call
`get_settings.cache_clear()` — the `temp_settings` fixture in `tests/conftest.py` does this.

> `model_source` and `model_version` start with the reserved prefix `model_`, so both
> `Settings` and the response schemas declare `protected_namespaces=()`. Without it, Pydantic v2
> emits a `UserWarning` on every import.

## Dataset

**[UCI Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)**
(also published as "Online Retail II").

| Property | Value |
|---|---|
| Rows | 1,067,371 transactions (line items) |
| Period | Dec 2010 – Dec 2011 (used here) |
| Size | ~43.5 MB `.xlsx` inside the zip |
| Licence | CC BY 4.0 |
| Columns | `Invoice`, `StockCode`, `Description`, `Year`, `Month`, `Originated from (YYYY-MM)`, `Customer ID`, `Quantity`, `Price`, `MemberID`, `Country` |
| Has a date column | ✅ `InvoiceDate` / `Year`+`Month` |
| Has numerics | ✅ `Quantity`, `Price` |
| Has categoricals | ✅ `Country`, `StockCode`, `Description` |
| Has missing values | ✅ `Customer ID` is null for a large minority of rows |
| Binary target | ❌ not present — constructed below |

### Why this dataset

The brief required a public, tabular, binary-classification dataset with a date column, numeric
and categorical features, and missing values. This one satisfies all of them and — crucially —
supports a **non-fabricated** binary label that can be derived purely from the data itself
rather than invented. It is also genuinely messy (cancelled orders as negative quantities,
free items at zero price, anonymous customers), so the validation and preprocessing layers have
real work to do instead of decoration.

### Download

```bash
python scripts/download_data.py
```

Behaviour:
1. Fetches `https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip`.
2. Caches the raw zip at `data/raw/online_retail_ii.zip` and **reuses it** on subsequent runs
   (no repeated 43 MB download).
3. Reads every sheet and concatenates — the workbook historically ships two half-year sheets.
4. Renames upstream headers to the internal contract: `Invoice → InvoiceNo`,
   `Price → UnitPrice`, `Customer ID → CustomerID`.
5. Coerces `CustomerID` to numeric, writes the normalised CSV to `RAW_DATA_PATH`.

Both the old and new header spellings are tolerated, so the script keeps working if the
publisher renames columns again. `data/` is gitignored — the raw and derived data are never
committed.

## Task definition

**Invoice-level customer retention: will this customer place another order within 30 days?**

```
row         = one (CustomerID, InvoiceNo) purchase event
target      = reorder_30d ∈ {0, 1}
              1 if the same customer's next invoice is ≤ 30 days after this one
features    = total_quantity, n_line_items, mean_unit_price, order_value,
              days_since_last_order, order_hour, country, month, day_of_week
```

Deliberate choices inside `build_features`:

- **Cancelled and invalid rows are dropped** — `InvoiceNo` starting with `C`, and
  `Quantity <= 0` or `UnitPrice <= 0`. These are returns and free samples, not purchases; a
  model trained on them learns the wrong "order" concept.
- **Null `CustomerID` is kept, not deleted.** Anonymous buyers are assigned a synthetic
  singleton member id (`ANON_<InvoiceNo>`), so they become customers who by definition do not
  reorder within 30 days. Deleting those rows would silently discard a large slice of real
  revenue activity *and* bias the positive class upward. Missingness stays visible to the
  median imputer instead of being erased.
- **`days_since_last_order` is null for a customer's first order.** This is encoded by the
  imputer rather than filled with a made-up number like `0` (which would claim "just ordered")
  or `9999` (which would claim "dormant for decades").
- **`month` and `day_of_week` are categorical strings, not cyclic numerics**, so
  `OneHotEncoder` treats them as nominal. Cyclic encoding would be arguably better modelling;
  the point here is a correct, inspectable pipeline rather than squeezing out the last AUC.

### Split strategy

One chronological cutoff at the 80/20 mark of the time axis:

```
train: InvoiceDate <  2011-07-14T21:25:00   →  30,445 invoices
test : InvoiceDate >= 2011-07-14T21:25:00   →   9,632 invoices
```

No shuffling. The cutoff is written into `metrics.json` (`split_cutoff`). The evaluation
therefore answers "how well does this model predict the *future*", which is what the service
will be asked to do — a random split would leak tomorrow's behaviour into training and report
an optimistic AUC.

## Run order

```bash
# 1. data (once, ~1 min including the 43 MB download)
python scripts/download_data.py

# 2. train + publish a version (a few minutes; three candidates on 30k rows)
python -m src.training.train

# 3. serve
$env:MODEL_SOURCE = "local"
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000

# 4. check it is actually serving the version you expect
curl http://127.0.0.1:8000/health

# 5. benchmark
python scripts/benchmark_latency.py --url http://127.0.0.1:8000 --n 1000

# 6. drift against a later window
python -m src.training.drift --data data/raw/new_window.csv --source local
```

Training is idempotent per invocation but creates a **new version every run** (the id is the
current UTC time). Repeated runs accumulate versions; that is intended, and
`scripts/rollback.py --list` shows them all.

## Windows / PowerShell notes

Three real gotchas in this environment:

- **`&&` is not a statement separator** in PowerShell 5.x. Use `;`.
- **Heredocs (`<<`) are reserved.** Write scripts to a file and run them.
- **Native stderr is surfaced as `NativeCommandError`.** `git push`, `docker`, and `pytest -v`
  write progress to stderr; PowerShell renders that as a red error even on a successful exit.
  Judge by the actual result line (`* [new branch] master -> master`), not the colour.
- **`moto`'s threaded server must bind `127.0.0.1` explicitly.** With the default binding,
  `ThreadedMotoServer` is unreachable on Windows and every S3 test fails with connection refused.
- **Creating a bucket with a non-default `LocationConstraint` fails under moto**
  (`InvalidLocationConstraint`). Tests create a single default-region bucket.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `FileNotFoundError: .../LATEST not found; has any model been uploaded?` | Server started before any training run | `python -m src.training.train`, or set `MODEL_VERSION` to an existing id |
| Service starts but `/health` returns 503 `MODEL_NOT_LOADED` | Loading failed at startup; the app tolerates it so the container still reports | Read the startup log for the loader error; check `MODEL_SOURCE` / `LOCAL_MODEL_DIR` / bucket permissions |
| `ValueError: unknown MODEL_SOURCE 'S3 '` | Unquoted trailing space or wrong case in `.env` | `MODEL_SOURCE=s3` on one line, no spaces, no quotes |
| `missing required columns: InvoiceNo, UnitPrice` | A CSV from an older download script version | Re-run `python scripts/download_data.py` to regenerate with renames applied |
| `InvalidLocationConstraint` in tests | Bucket created with a region hint under moto | Use the `s3` fixture in `tests/conftest.py`, which creates one default-region bucket |
| `cannot point LATEST at unknown version` | Typo in `--version`, or that version was never uploaded | `python scripts/rollback.py --list` and copy the exact id |
| `BATCH_TOO_LARGE` | Payload exceeds `MAX_BATCH_SIZE` | Split the payload, or raise `MAX_BATCH_SIZE` deliberately after checking p99 |
| `SKLearnNotFittedError` on the loaded model | Artifact directory pointed at a non-artifact path | `LOCAL_MODEL_DIR` must be the store root (e.g. `models/artifacts`), not the version folder |

Next: [api-reference.md](api-reference.md) for the full HTTP contract.
