# Deployment

Container, object storage, CI/CD and EC2.

> **Verification status — read first.** Everything in §1–§2 is implemented and its Python-side
> path logic is verified. **Docker itself was not built or run in the environment this repo was
> written in** (the Docker daemon was unavailable), and **no real AWS/EC2 interaction has been
> performed**. Steps 3–5 are therefore *instructions to execute*, not claims of success. Nothing
> here reports a container image size, EC2 latency, or a green Actions run that was not actually
> observed. See the checklist at the end.

---

## Table of contents

- [1. Docker image](#1-docker-image)
- [2. Docker Compose](#2-docker-compose)
- [3. GitHub Actions CI](#3-github-actions-ci)
- [4. Real S3](#4-real-s3)
- [5. ECR + EC2 deploy](#5-ecr--ec2-deploy)
- [6. Post-deploy verification](#6-post-deploy-verification)
- [Manual verification checklist](#manual-verification-checklist)

---

## 1. Docker image

`Dockerfile` — two stages, non-root, healthchecked.

```
builder: python:3.11-slim
  └─ pip install --prefix=/install -r requirements.txt      # wheels only, no caches
runtime: python:3.11-slim
  ├─ COPY --from=builder /install /usr/local                # only resolved site-packages
  ├─ useradd appuser  →  USER appuser                       # never root
  ├─ COPY src ./src ; COPY scripts ./scripts
  ├─ ENV MODEL_SOURCE=local  LOCAL_MODEL_DIR=/app/models
  ├─ EXPOSE 8000
  ├─ HEALTHCHECK → urllib GET /health, exit 0 only on HTTP 200
  └─ CMD python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

Build and run:

```bash
docker build -t ml-serving-api:latest .
docker run --rm -p 8000:8000 \
  -v "${PWD}/models/artifacts:/app/models:ro" \
  ml-serving-api:latest
```

Notes on the design:

- **`--prefix=/install` in the builder** keeps pip metadata, build tools and caches out of the
  final layer. Only `/install` is copied forward.
- **`.dockerignore`** excludes `reference/`, `data/`, `.venv`, `__pycache__`, `.git`, tests and
  docs, so the build context stays small and the 96 MB dataset is never copied into an image.
- **`HEALTHCHECK` uses `urllib`, not `curl`** — the slim image has no curl, and installing one
  just for a healthcheck adds a package to the attack surface for no benefit.
- **Health ≠ liveness is deliberate.** The healthcheck fails if `/health` returns 503, i.e. the
  container is running but has no model. Docker will mark it unhealthy rather than letting a
  model-less container look healthy and silently 503 all predictions.
- **No model is baked into the image.** Artifacts come from the bind mount (local) or S3
  (`MODEL_SOURCE=s3`). Keeping the model out of the image is what allows one image tag to serve
  any version, and prevents training data/pipeline state from leaking into a widely pullable artifact.

## 2. Docker Compose

```yaml
services:
  api:
    build: .
    image: ml-serving-api:latest
    ports: ["8000:8000"]
    env_file: [{ path: .env, required: false }]
    environment:
      MODEL_SOURCE: ${MODEL_SOURCE:-local}
      LOCAL_MODEL_DIR: /app/models
    volumes: ["./models/artifacts:/app/models:ro"]
    restart: unless-stopped
```

```bash
docker compose up --build
```

**The path mapping is the one thing that will confuse you**, because `LocalModelStore(root)`
appends a `models/` level itself:

| Layer | Path |
|---|---|
| Host | `models/artifacts/models/<VERSION>/model.joblib` |
| Bind mount | `./models/artifacts` → `/app/models` |
| Container | `/app/models/<VERSION>/model.joblib` |
| `LOCAL_MODEL_DIR` | `/app/models` |
| `store.root` (computed) | `/app/models/models` ⚠️ |
| Store actually reads | `/app/models/models/<VERSION>/` |

Verified by constructing a store in Python:

```
LOCAL_MODEL_DIR=/app/models   -> store.root=\app\models\models     ✓ matches the mount
LOCAL_MODEL_DIR=/app/models/models -> store.root=...\models\models  ✗ one level too deep
```

So **`/app/models` is correct** — despite looking off-by-one at a glance. An earlier draft of
this doc claimed compose was broken by a missing level; that claim was wrong and was produced by
reasoning about the path instead of measuring it. Anyone "fixing" this by adding a `models`
segment will break a working deploy. The mount and the variable are correct as configured;
what is *not* verified is the runtime behaviour (see the checklist).

- `env_file` with `required: false` means the stack starts on a clean clone with no `.env`.
- The mount is `:ro` — the serving process has no business writing to the artifact tree.
- `MODEL_SOURCE: ${MODEL_SOURCE:-local}` lets `MODEL_SOURCE=s3 docker compose up` switch backends
  without editing files.

## 3. GitHub Actions CI

`.github/workflows/ci.yml`, triggered on `push` (and `pull_request`).

### `build-test` (always runs)

| Step | Command | Fails the build if |
|---|---|---|
| Checkout + setup-python | `python-version: 3.11`, pip cache | — |
| Install | `pip install -r requirements-dev.txt` | dependency resolution breaks |
| Lint | `ruff check src tests scripts` | any rule violation |
| Tests | `pytest -q --cov=src` | any failure or exception |
| Docker build | `docker build .` | Dockerfile or context error |

The test job needs **no AWS credentials** — S3 is mocked with moto, so CI is free and cannot be
blocked by an expired key.

### `deploy` (guarded)

Runs only on `workflow_dispatch` or a `v*` tag, **and** only when every deploy secret exists:

```yaml
if: |
  (github.event_name == 'workflow_dispatch' || startsWith(github.ref, 'refs/tags/v')) &&
  vars... # plus a step that verifies all seven secrets are non-empty
```

A dedicated **guard step** checks all seven secrets and fails loudly rather than half-deploying.
This satisfies the brief's requirement that the job *must not run unless secrets are configured* —
a missing secret otherwise becomes an empty string and would produce a confusing SSH or ECR error
mid-deploy.

Secrets (all seven required):

| Secret | Used for |
|---|---|
| `AWS_ACCESS_KEY_ID` | ECR login + push |
| `AWS_SECRET_ACCESS_KEY` | ECR login + push |
| `AWS_REGION` | ECR endpoint, boto3 |
| `ECR_REPOSITORY` | image repository name |
| `EC2_HOST` | SSH target |
| `EC2_USER` | SSH user |
| `EC2_SSH_KEY` | SSH private key |

Deploy sequence: configure credentials → `aws ecr get-login-password | docker login` →
build + tag `:latest` and `:<git sha>` → push → `appleboy/ssh-action` to EC2:
`docker pull` → stop/remove the old container → run the new one with `--env-file`.

**No secret value is echoed** anywhere in the workflow, and nothing is committed. The private key
lives only in GitHub Actions secrets.

## 4. Real S3

One-time bucket setup:

```bash
aws s3api create-bucket --bucket your-model-bucket --region us-east-1
# serving role needs only:
#   s3:ListBucket on arn:aws:s3:::your-model-bucket  (prefix-conditioned)
#   s3:GetObject  on arn:aws:s3:::your-model-bucket/<prefix>/models/*
# training/CI role additionally needs s3:PutObject on the same prefix
```

Versioned artifacts live under one prefix; **S3 native object versioning is not required** —
the app implements its own versioning through timestamped keys, which is simpler, visible in the
console, and does not depend on a bucket feature that could be switched off.

```bash
# train and publish to S3
$env:MODEL_SOURCE="s3"; $env:S3_BUCKET="your-model-bucket"; $env:S3_PREFIX="production-ml-serving"
$env:AWS_REGION="us-east-1"        # credentials via `aws configure` or an IAM role
python -m src.training.train

# verify
aws s3 ls s3://your-model-bucket/production-ml-serving/models/
aws s3api get-object --bucket your-model-bucket \
  --key production-ml-serving/models/LATEST /dev/stdout

# serve from S3
docker run -p 8000:8000 \
  -e MODEL_SOURCE=s3 -e S3_BUCKET=your-model-bucket -e S3_PREFIX=production-ml-serving \
  -e AWS_REGION=us-east-1 \
  ml-serving-api:latest
```

On EC2/ECS prefer an **IAM instance role** over static keys, so no credential ever lands on disk
or in an environment file.

> moto accepts any credentials, so **a broken credential configuration will not fail the test
> suite**. Real-S3 auth must be verified here, not inferred from green tests.

## 5. ECR + EC2 deploy

Minimal EC2 preparation (manual, once):

```bash
# on the instance
sudo yum install -y docker   # or apt-get install docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user   # then re-login
aws ecr create-repository --repository-name ml-serving-api --region us-east-1
mkdir -p ~/ml-serving && cat > ~/ml-serving/.env <<'EOF'
MODEL_SOURCE=s3
S3_BUCKET=your-model-bucket
S3_PREFIX=production-ml-serving
AWS_REGION=us-east-1
EOF
```

The workflow's SSH step then runs, on the host:

```bash
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS \
  --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker pull <account>.dkr.ecr.<region>.amazonaws.com/ml-serving-api:latest
docker stop ml-serving-api || true ; docker rm ml-serving-api || true
docker run -d --name ml-serving-api --restart unless-stopped \
  -p 8000:8000 --env-file ~/ml-serving/.env \
  <account>.dkr.ecr.<region>.amazonaws.com/ml-serving-api:latest
```

**Security-group hygiene:** allow ingress on 8000 only from your ALB/gateway or VPN, never
`0.0.0.0/0` — this API has no authentication layer. Put an ALB with TLS and an authn policy in
front for anything internet-reachable.

## 6. Post-deploy verification

Run after every deploy, in order. Stop at the first failure.

```bash
# 1. container is up and healthy
docker ps --filter name=ml-serving-api --format '{{.Status}}'   # expect "Up ... (healthy)"

# 2. it is serving the version you expect
curl -s http://<host>:8000/health
# {"status":"ok","model_version":"20261029T145410Z","loaded_at":"...","model_source":"s3"}

# 3. it predicts
curl -s -X POST http://<host>:8000/predict -H 'Content-Type: application/json' \
  -d '{"features":{"total_quantity":12,"n_line_items":3,"mean_unit_price":4.16,
       "order_value":49.92,"days_since_last_order":21.0,"order_hour":10.0,
       "country":"United Kingdom","month":"9","day_of_week":"2"}}'

# 4. the error contract survived the deploy
curl -s -o /dev/null -w '%{http_code}\n' -X POST http://<host>:8000/predict \
  -H 'Content-Type: application/json' -d '{"features":{}}'      # expect 422

# 5. latency is not a surprise — run from a *different* host than the server
python scripts/benchmark_latency.py --url http://<host>:8000 --n 1000
```

The `model_version` in step 2 is the real deployment gate: if it is not what you intended, the
deploy did not do what you think, regardless of everything else passing.

## Manual verification checklist

These five items are **not verified** in this environment. Each is required before the README's
performance claims can be extended to Docker/EC2.

| # | Action | Command / location | What "passing" looks like | Status |
|---|---|---|---|---|
| 1 | Build the image | `docker build -t ml-serving-api:latest .` | Exit 0; no `data/` or `reference/` in any layer | ⬜ **not run** — Docker daemon unavailable |
| 2 | Run + healthcheck | `docker compose up` then `curl /health` | `Up (healthy)`; `/health` returns `model_version` = LATEST | ⬜ **not run** |
| 3 | Predict through Docker | `curl -X POST /predict` against the container | 200 + a `probability` | ⬜ **not run** |
| 4 | Real S3 round-trip | §4 above | Version appears under the prefix; `LATEST` readable; service boots with `MODEL_SOURCE=s3` | ⬜ **not run** — no AWS account/credentials |
| 5 | CI + deploy | push a `v*` tag | `build-test` green; `deploy` **skipped** with no secrets, and runs with them | ⬜ **not verified** — Actions logs not readable from here |
| 6 | Docker/EC2 latency | §6 step 5 | Record real p50/p95/p99; only then update the README results table | ⬜ **not run** |

Item 6 matters for an honest claim: the README's **p95 = 7.0 ms** was measured against a **native
local Uvicorn process**, not this container over a network. Do not cite the résumé's ~14 ms or
this project's 7 ms as a containerised number until item 6 is done.

Related: [model-versioning.md](model-versioning.md) · [runbook.md](runbook.md) ·
[architecture.md](architecture.md)
