# syntax=docker/dockerfile:1

# ---- build stage: resolve wheels into a self-contained venv ----
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --prefix=/install -r requirements.txt

# ---- runtime stage: slim, non-root, healthchecked ----
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MODEL_SOURCE=local

WORKDIR /app

# copy only the resolved site-packages from the builder
COPY --from=builder /install /usr/local

# create an unprivileged user
RUN useradd --create-home --shell /usr/sbin/nologin appuser

COPY src ./src
COPY scripts ./scripts
USER appuser

# model artifacts are mounted/provided at runtime via LOCAL_MODEL_DIR or S3
ENV LOCAL_MODEL_DIR=/app/models

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health',timeout=4).status==200 else 1)"

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
