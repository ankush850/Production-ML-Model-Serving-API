# Testing

The suite is **fully offline**: no AWS account, no credentials, no dataset download, no
network. `pytest` on a clean clone passes in seconds.

---

## Table of contents

- [Measured results](#measured-results)
- [Layout and what each module proves](#layout-and-what-each-module-proves)
- [Running the suite](#running-the-suite)
- [Fixture architecture](#fixture-architecture)
- [Mocking S3 with moto](#mocking-s3-with-moto)
- [Determinism strategy](#determinism-strategy)
- [Coverage breakdown and the gaps](#coverage-breakdown-and-the-gaps)
- [Windows-specific test failures fixed](#windows-specific-test-failures-fixed)
- [Extending the suite](#extending-the-suite)

---

## Measured results

```
54 passed, 1 warning in 6.84s
TOTAL    751 statements    66 missed    91%
```

| | |
|---|---|
| Tests | **54** |
| Failures | **0** |
| Line coverage | **91%** (751 statements, 66 missed) |
| Wall time | ~7 s |
| Required services | none |
| Coverage bar from the brief | ≥ 80% → **met, 91%** |

The single warning is a `PendingDeprecationWarning` from **starlette** (`import multipart` →
`import python_multipart`). It originates in a third-party package, not in this codebase, and
cannot be fixed without a starlette upgrade. It is deliberately not suppressed with
`filterwarnings`, so a genuine new warning still shows up.

## Layout and what each module proves

Four test modules, one per acceptance criterion.

| Module | Tests | Proves |
|---|---|---|
| `tests/test_api.py` | **17** | `/health` (incl. 503 when unloaded), `/predict` happy path, 422 `VALIDATION_ERROR` envelope, 400 `SCHEMA_VIOLATION`, `/predict/batch` incl. `EMPTY_BATCH` and `BATCH_TOO_LARGE`, `X-Process-Time` / `X-Request-ID` middleware, correlation-id echo, model version echoed in every response |
| `tests/test_pipeline.py` | **12** | `build_features` semantics (cancellations dropped, anonymous customers retained, target windows), `time_split` ordering and cutoff, single-artifact pipeline round-trip, `LocalModelStore` + `S3ModelStore` versioning, immutability (`FileExistsError`), `list_versions` sorting, `LATEST` set/get, rollback to an older version |
| `tests/test_drift.py` | **10** | PSI ≈ 0 on the reference, PSI rises under a shifted window, `psi_from_bins` vs legacy `psi_numeric`, tied-bin regression case, `ks_2samp` on identical samples → `(0, 1)`, categorical share shift, `check_drift` report shape, CLI exit codes 0 / 1 / 2 |
| `tests/test_validation.py` | **15** | Valid frame passes; missing column; custom schema respected; numeric-as-string and datetime-as-string rejected; nulls in each required column (parametrised ×4); >95% null fraction rejected; negative quantity and negative price range checks; `raise_if_invalid` typed error with structured `.details()`; default schema covers the raw columns |

Two of these deserve a note on **why they exist**:

- **Immutability test** (`test_pipeline.py`) is the regression test for a real bug: the existence
  check originally lived in only one store implementation, so `S3ModelStore` would happily
  overwrite a published version. The shared base class now enforces it, and the test asserts the
  behaviour on **both** backends so it cannot silently diverge again.
- **Tied-bin PSI test** (`test_drift.py`) is the regression test for `n_line_items` reporting
  **PSI 0.577 against its own reference**. Any future change to bin construction that reintroduces
  quantile ties fails here. See
  [drift-monitoring.md](drift-monitoring.md#why-bins-are-stored-instead-of-recomputed--the-bug-this-fixed).

## Running the suite

```bash
# full suite with coverage (what CI runs)
python -m pytest -q --cov=src --cov-report=term-missing

# fast feedback on one area
python -m pytest tests/test_api.py -q

# see every test name
python -m pytest --collect-only -q

# fail fast, show locals
python -m pytest -x -vv
```

`make test` runs the first form. Configuration lives in `pyproject.toml` under `[tool.pytest.ini_options]`
and `[tool.coverage.*]`.

## Fixture architecture

`tests/conftest.py` provides the shared plumbing. Fixtures are layered so each test declares
only what it actually needs:

```
make_raw_rows(n=240, seed=7)          # synthetic line-item rows, ~10% null CustomerID
   └─ raw_df                          # validated raw frame
        └─ built_df                   # invoice-level feature frame
             ├─ tiny_pipeline         # fitted sklearn Pipeline (fast, small)
             ├─ schema_dict           # schema.json equivalent
             ├─ reference_stats       # reference stats for drift tests
             └─ loaded_model          # LoadedModel wrapping the above
moto_server                           # module-scoped ThreadedMotoServer @ 127.0.0.1
   └─ s3                              # function-scoped client, bucket emptied between tests
temp_settings                         # env override + get_settings.cache_clear()
artifact_bytes / TEST_VERSION         # 20260101T000000Z + 4 artifact payloads
```

Design points:

- **`make_raw_rows` generates data that exercises the real edge cases** — ~10% null `CustomerID`,
  cancelled invoices, non-positive quantities — rather than a uniformly clean frame. Tests over
  pristine synthetic data would pass while production breaks.
- **`tiny_pipeline` is genuinely fitted**, not mocked. API tests therefore run the *actual*
  sklearn transform path, so a broken `ColumnTransformer` cannot hide behind a fake.
- **`temp_settings` must call `get_settings.cache_clear()`** because `Settings` is wrapped in
  `lru_cache`. Without it, an env-var change is silently ignored and the test proves nothing.
- **`s3` empties the bucket between tests** (`_empty_bucket`) so `list_versions()` assertions are
  not polluted by a previous test's uploads — module-scoped moto server, function-scoped isolation.

## Mocking S3 with moto

`moto[server]` runs a **real HTTP** S3-compatible endpoint in-process, so boto3 is exercised
end-to-end: request signing, pagination, `Delimiter`/`CommonPrefixes`, and the
`NoSuchKey` → `FileNotFoundError` mapping in `ModelStore._get`.

```python
server = ThreadedMotoServer(ip_address="127.0.0.1", port=5000)
server.start()
boto3.client("s3", endpoint_url="http://127.0.0.1:5000", region_name="us-east-1")
```

Using the server (not the `mock_aws` decorator) matters here: the store is also used from the
FastAPI app and from subprocess-like CLI paths, and an in-process patch does not reach a real
HTTP client call the same way. **What the tests therefore cover:** every storage code path
except AWS-specific behaviour (bucket policies, KMS, cross-region rules, real IAM). That gap is
acknowledged rather than papered over — see [deployment.md](deployment.md#4-real-s3).

Consequence to respect: moto **accepts any credentials**, so a missing-credential bug will not
fail the test suite. Credential handling must be verified in the real-S3 manual step.

## Determinism strategy

A flaky test is worse than no test. Sources of nondeterminism and how each is handled:

| Source | Handling |
|---|---|
| Random synthetic data | Fixed `seed=7` in `make_raw_rows` |
| sklearn RNG | `tiny_pipeline` built from a seeded estimator; `random_state` set on candidates in `train.py` |
| Shared moto state | `s3` fixture empties the bucket per test; module-scoped server avoids bind races |
| Port collisions | Single fixed port with `ip_address="127.0.0.1"`; server created once per module |
| Global config cache | `temp_settings` clears the `lru_cache` before and after |
| Version timestamps | Constant `TEST_VERSION = "20260101T000000Z"` instead of `now()` |
| Filesystem leakage | Everything writes under pytest `tmp_path` |
| Trained-model metrics | **Not asserted against a hardcoded value** — `test_pipeline.py` asserts the
  best candidate's ROC-AUC beats the logistic baseline and that `metrics.json` keys exist, rather
  than pinning `0.7717`. Asserting an exact float would break on a different sklearn/BLAS build |

The last point is the one most often got wrong: hardcoding `assert roc_auc == 0.7717` makes the
test suite a hostage to the numeric library stack, and a legitimate 0.77 on another machine reads
as a regression.

## Coverage breakdown and the gaps

```
Name                       Stmts   Miss  Cover   Missing
--------------------------------------------------------
src\api\__init__.py            0      0   100%
src\api\main.py              131      9    93%   55-59, 115-117, 135, 138, 222
src\api\model_loader.py       38      2    95%   49, 78
src\api\schemas.py            44      0   100%
src\config.py                 20      0   100%
src\data\__init__.py           0      0   100%
src\data\preprocess.py        46      0   100%
src\data\validate.py          76      2    97%   55, 142
src\storage\__init__.py        0      0   100%
src\storage\s3_store.py      135     14    90%   71-72, 98, 106, 110, 114, 158-159, 162, 207, 216-217, 225, 228
src\training\__init__.py       0      0   100%
src\training\drift.py        135     20    85%   67-82, 126, 163, 169, 189, 238
src\training\train.py        126     19    85%   55-56, 197, 228-231, 244-261, 265
--------------------------------------------------------
TOTAL                        751     66    91%
```

**100%:** `schemas.py`, `config.py`, `preprocess.py` — the pure-logic modules are fully covered,
which is where you want completeness.

What is **not** covered, and why that is acceptable rather than an oversight:

| Lines | What | Assessment |
|---|---|---|
| `train.py` 244-261 | `main()` CLI wrapper: argparse, logging setup, printing | Orchestration only. The logic it calls (`run_training`) *is* tested. Testing the wrapper would mostly test argparse |
| `train.py` 228-231 | S3 upload branch inside `run_training` | Needs a store with `MODEL_SOURCE=s3`; exercisable but currently covered only via the direct store tests |
| `drift.py` 67-82 | legacy `psi_numeric()` tied-edge merge path | Only reached for **pre-fix artifacts** lacking `bins`. Backward-compat path; the new path is covered. Intentional dead-ish code kept so old versions stay checkable |
| `drift.py` 238 | `sys.exit(main())` under `__main__` | Standard module entrypoint |
| `s3_store.py` 71-72, 158-162 | `_get` `ClientError` mapping; absent-artifact skip in `download_version` | Moto raises different error shapes than real S3 for some of these; partially covered |
| `s3_store.py` 98, 110, 114 | abstract `NotImplementedError` bodies on the base class | Unreachable by construction |
| `main.py` 55-59, 115-117 | middleware exception re-raise and the 500 handler | Triggered only by a genuine unhandled crash; faking one adds little signal |
| `validate.py` 55, 142 · `model_loader.py` 49, 78 | defensive branches | Small; worth covering if the modules change shape |

The honest framing: **91% overall, with the gaps concentrated in CLI wrappers and defensive
branches, not in core logic.** Chasing a round number by testing argparse would add maintenance
cost without adding safety.

## Windows-specific test failures fixed

Each of these was a real failure in this suite, in this environment.

| Failure | Root cause | Fix |
|---|---|---|
| `Connection refused` on every S3 test | `ThreadedMotoServer` defaults to a binding Windows does not accept | `ip_address="127.0.0.1"` explicitly |
| `InvalidLocationConstraint` creating a `eu-west-1` bucket | moto validates `LocationConstraint` against its own region table | Create **one** default-region bucket, no `LocationConstraint` |
| `KeyError: 'model.joblib'` on assertions | `LocalModelStore(root)` adds a `models/` level, so files land at `root/models/<VERSION>/` | Assert against `store.root`, not the directory passed in |
| `AssertionError: PSI 0.577 != ~0` | Tied quantile edges on integer features | Store `bins` + `expected`; add `psi_from_bins` (see [drift doc](drift-monitoring.md)) |
| `ValueError: setting an array element with a sequence` / dtype warning | Assigning `"ANON_..."` strings into a float `CustomerID` column | `df["CustomerID"] = df["CustomerID"].astype("object")` before assignment |
| pydantic `UserWarning: field name "model_version" conflicts with protected namespace` | `model_` is reserved in Pydantic v2 | `protected_namespaces=()` on `Settings` and on `PredictResponse` / `BatchResponse` / `HealthResponse` |
| TestClient lifespan errored — no model loaded | `TestClient` runs `lifespan`, which tried to load from an empty store | The `client` fixture uploads a tiny artifact set to a temp store first, then asserts `app.state.model is not None` |

Related: [runbook.md](runbook.md) · [architecture.md](architecture.md) ·
[deployment.md](deployment.md#3-github-actions-ci)
