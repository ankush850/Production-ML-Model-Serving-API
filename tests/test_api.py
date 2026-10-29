"""API contract tests: /health, /predict, batch, structured errors, middleware."""

from __future__ import annotations

import io
import json

import joblib
import pytest
from fastapi.testclient import TestClient

from src.api import main as api_main
from tests.conftest import TEST_VERSION

VALID = {
    "total_quantity": 12,
    "n_line_items": 3,
    "mean_unit_price": 4.16,
    "order_value": 49.92,
    "days_since_last_order": 21.0,
    "order_hour": 10.0,
    "country": "United Kingdom",
    "month": "9",
    "day_of_week": "2",
}


@pytest.fixture
def client(loaded_model, temp_settings, tmp_path):
    """TestClient whose lifespan really loads the tiny model from the store."""
    from src.storage.s3_store import LocalModelStore

    buf = io.BytesIO()
    joblib.dump(loaded_model.pipeline, buf)
    store = LocalModelStore(root=tmp_path / "models_local")
    store.upload_version(
        TEST_VERSION,
        {
            "model.joblib": buf.getvalue(),
            "schema.json": json.dumps(loaded_model.schema).encode(),
        },
    )
    store.set_latest(TEST_VERSION)
    with TestClient(api_main.app) as c:
        assert c.app.state.model is not None
        yield c


@pytest.fixture
def client_no_model(temp_settings):
    """TestClient with NO model available - exercises the 503 paths.

    ``temp_settings`` points the local store at an empty temp dir, so the
    lifespan load fails cleanly and app.state.model stays None.
    """
    api_main.app.state.model = None
    with TestClient(api_main.app) as c:
        yield c


class TestHealth:
    def test_ok_payload(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "ok"
        assert body["model_version"] == TEST_VERSION
        assert body["model_source"] == "local"
        assert "loaded_at" in body

    def test_model_not_loaded_returns_503(self, client_no_model):
        resp = client_no_model.get("/health")
        assert resp.status_code == 503
        assert resp.json()["error_code"] == "MODEL_NOT_LOADED"


class TestPredict:
    def test_success(self, client):
        resp = client.post("/predict", json={"features": VALID})
        assert resp.status_code == 200
        body = resp.json()
        assert body["prediction"] in (0, 1)
        assert 0.0 <= body["probability"] <= 1.0
        assert body["model_version"] == TEST_VERSION

    def test_missing_fields_is_partial_input_422(self, client):
        partial = {k: v for k, v in VALID.items() if k != "order_value"}
        resp = client.post("/predict", json={"features": partial})
        assert resp.status_code == 422
        body = resp.json()
        assert body["error_code"] == "VALIDATION_ERROR"
        assert any("order_value" in d["field"] for d in body["details"])

    def test_wrong_type_422(self, client):
        bad = {**VALID, "order_hour": "not-a-number"}
        resp = client.post("/predict", json={"features": bad})
        assert resp.status_code == 422
        assert resp.json()["error_code"] == "VALIDATION_ERROR"

    def test_out_of_range_422(self, client):
        resp = client.post("/predict", json={"features": {**VALID, "order_hour": 99}})
        assert resp.status_code == 422

    def test_out_of_schema_range_400(self, client, schema_dict):
        # huge but type-valid number: caught against the model's schema.json
        resp = client.post("/predict", json={"features": {**VALID, "total_quantity": 10**9}})
        assert resp.status_code == 400
        body = resp.json()
        assert body["error_code"] == "SCHEMA_VIOLATION"
        assert body["details"][0]["field"] == "total_quantity"

    def test_unknown_category_400(self, client):
        resp = client.post("/predict", json={"features": {**VALID, "country": "Atlantis"}})
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "SCHEMA_VIOLATION"

    def test_null_category_is_allowed(self, client):
        resp = client.post("/predict", json={"features": {**VALID, "country": None}})
        assert resp.status_code == 200

    def test_empty_body_422(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422
        assert resp.json()["error_code"] == "VALIDATION_ERROR"

    def test_model_not_loaded_503(self, client_no_model):
        resp = client_no_model.post("/predict", json={"features": VALID})
        assert resp.status_code == 503
        assert resp.json()["error_code"] == "MODEL_NOT_LOADED"


class TestBatch:
    def test_success(self, client):
        resp = client.post("/predict/batch", json=[VALID, {**VALID, "month": "3"}])
        assert resp.status_code == 200
        body = resp.json()
        assert body["size"] == 2
        assert len(body["results"]) == 2
        assert body["model_version"] == TEST_VERSION

    def test_empty_batch_400(self, client):
        resp = client.post("/predict/batch", json=[])
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "EMPTY_BATCH"

    def test_batch_too_large_400(self, client, monkeypatch):
        from src.config import get_settings

        monkeypatch.setenv("MAX_BATCH_SIZE", "1")
        get_settings.cache_clear()
        resp = client.post("/predict/batch", json=[VALID, VALID])
        assert resp.status_code == 400
        assert resp.json()["error_code"] == "BATCH_TOO_LARGE"
        get_settings.cache_clear()

    def test_invalid_record_reports_indexed_field(self, client):
        resp = client.post("/predict/batch", json=[VALID, {**VALID, "country": "Atlantis"}])
        assert resp.status_code == 400
        details = resp.json()["details"]
        assert any(d["field"].startswith("[1]") for d in details)


class TestMiddleware:
    def test_process_time_and_request_id_headers(self, client):
        resp = client.get("/health")
        assert float(resp.headers["X-Process-Time"]) > 0
        assert len(resp.headers["X-Request-ID"]) >= 8

    def test_request_id_echoed_when_provided(self, client):
        resp = client.get("/health", headers={"x-request-id": "abc-123"})
        assert resp.headers["X-Request-ID"] == "abc-123"
