"""FastAPI serving app: lifespan model load, middleware, structured errors.

Endpoints:
    GET  /health          liveness + active model metadata (503 if not loaded)
    POST /predict         single-record prediction, schema-checked against the
                          model version's ``schema.json``
    POST /predict/batch   batched predictions (size capped by MAX_BATCH_SIZE)

Errors are always the structured envelope
``{error_code, message, details:[{field, issue}]}`` - never a raw stack trace.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import HTTPConnection

from src.api.model_loader import LoadedModel, load_model
from src.api.schemas import (
    BatchRequest,
    BatchResponse,
    ErrorResponse,
    PredictRequest,
    PredictResponse,
)
from src.config import get_settings
from src.data.preprocess import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger("api")


def _error_payload(code: str, message: str, details: list[dict[str, str]] | None = None) -> dict:
    return {"error_code": code, "message": message, "details": details or []}


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign a request id, time the request and log structured access lines."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "unhandled error request_id=%s path=%s", request_id, request.url.path
            )
            raise
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        response.headers["X-Request-ID"] = request_id
        logger.info(
            "request_id=%s method=%s path=%s status=%d elapsed=%.4f",
            request_id, request.method, request.url.path, response.status_code, elapsed,
        )
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the LATEST (or pinned) model at startup; tolerate failure (503 path)."""
    settings = get_settings()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    try:
        app.state.model = load_model()
        logger.info("model %s loaded (%s)", app.state.model.version, settings.model_source)
    except Exception as e:  # noqa: BLE001 - startup must not crash the process
        app.state.model = None
        logger.error("model load failed at startup: %s", e)
    yield


app = FastAPI(
    title="Production ML Model Serving API",
    description="Customer reorder (30-day retention) predictions from a timestamp-versioned model store.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(RequestContextMiddleware)


# -- error handlers ---------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Map Pydantic/validation failures onto the structured error envelope."""
    details = []
    for err in exc.errors():
        loc = [str(p) for p in err.get("loc", ()) if p not in ("body",)]
        details.append({"field": ".".join(loc) or "body", "issue": err.get("msg", "invalid")})
    payload = ErrorResponse(
        error_code="VALIDATION_ERROR",
        message="request body failed validation",
        details=details,
    )
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content=payload.model_dump())


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    """Generic 500 - details logged server-side, never returned to the client."""
    rid = getattr(request.state, "request_id", "unknown")
    logger.exception("unexpected error request_id=%s", rid)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=_error_payload("INTERNAL_ERROR", "unexpected server error", [{"field": "", "issue": f"request_id={rid}"}]),
    )


# -- helpers ----------------------------------------------------------------
def _get_model(conn: HTTPConnection) -> LoadedModel | None:
    return getattr(conn.app.state, "model", None)


def _check_against_schema(features: dict[str, Any], schema: dict[str, Any]) -> list[dict[str, str]]:
    """Range/allowed-category checks against the model version's schema.json."""
    problems: list[dict[str, str]] = []
    for col in NUMERIC_FEATURES:
        spec = schema.get("numeric", {}).get(col)
        value = features.get(col)
        if spec is None or value is None:
            continue
        lo, hi = spec.get("min"), spec.get("max")
        if lo is not None and value < lo - 1e-9:
            problems.append({"field": col, "issue": f"below schema minimum {lo}"})
        if hi is not None and value > hi + 1e-6:
            problems.append({"field": col, "issue": f"above schema maximum {hi}"})
    for col in CATEGORICAL_FEATURES:
        spec = schema.get("categorical", {}).get(col)
        value = features.get(col)
        if spec is None or value is None:
            continue
        allowed = set(spec.get("allowed", []))
        if allowed and str(value) not in allowed:
            problems.append({"field": col, "issue": f"unknown category {value!r}"})
    return problems


def _predict(model: LoadedModel, records: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[float]]:
    X = pd.DataFrame(records, columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    proba = model.pipeline.predict_proba(X)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return preds, proba


def _require_model(conn: HTTPConnection) -> JSONResponse | None:
    if _get_model(conn) is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_error_payload("MODEL_NOT_LOADED", "no model is loaded; check MODEL_SOURCE/MODEL_VERSION"),
        )
    return None


# -- endpoints --------------------------------------------------------------
@app.get("/health", response_model=dict, responses={503: {"model": ErrorResponse}})
async def health(request: Request) -> JSONResponse:
    """Liveness/readiness probe exposing the active model version."""
    model = _get_model(request)
    if model is None:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=_error_payload("MODEL_NOT_LOADED", "model failed to load or is not configured"),
        )
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content={
            "status": "ok",
            "model_version": model.version,
            "loaded_at": model.loaded_at,
            "model_source": model.source,
        },
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict(body: PredictRequest, request: Request) -> PredictResponse | JSONResponse:
    """Predict whether the customer of this invoice reorders within 30 days."""
    unavailable = _require_model(request)
    if unavailable is not None:
        return unavailable
    model = _get_model(request)
    features = body.features.model_dump()
    problems = _check_against_schema(features, model.schema)
    if problems:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_error_payload("SCHEMA_VIOLATION", "input outside model schema bounds", problems),
        )
    preds, proba = _predict(model, [features])
    return PredictResponse(
        prediction=int(preds[0]), probability=float(proba[0]), model_version=model.version
    )


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    responses={400: {"model": ErrorResponse}, 422: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
)
async def predict_batch(body: BatchRequest, request: Request) -> BatchResponse | JSONResponse:
    """Batched prediction; rejects empty batches and batches above MAX_BATCH_SIZE."""
    unavailable = _require_model(request)
    if unavailable is not None:
        return unavailable
    model = _get_model(request)
    settings = get_settings()
    records = [f.model_dump() for f in body.root]
    if not records:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_error_payload("EMPTY_BATCH", "batch must contain at least one record", []),
        )
    if len(records) > settings.max_batch_size:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_error_payload(
                "BATCH_TOO_LARGE",
                f"batch size {len(records)} exceeds maximum {settings.max_batch_size}",
                [],
            ),
        )
    problems: list[dict[str, str]] = []
    for i, rec in enumerate(records):
        for p in _check_against_schema(rec, model.schema):
            problems.append({"field": f"[{i}].{p['field']}", "issue": p["issue"]})
    if problems:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=_error_payload("SCHEMA_VIOLATION", "some records violate the model schema", problems),
        )
    preds, proba = _predict(model, records)
    results = [
        PredictResponse(prediction=int(p), probability=float(q), model_version=model.version)
        for p, q in zip(preds, proba, strict=True)
    ]
    return BatchResponse(results=results, model_version=model.version, size=len(results))
