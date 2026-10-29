"""Shared pytest fixtures: synthetic mini data, tiny trained pipeline, moto S3.

Everything runs offline: S3 is mocked with moto and the "real" training data
used in tests is a small deterministic synthetic frame that has the SAME
columns as the Online Retail II raw export (documented as synthetic - it is
only for exercising code paths, never for reported metrics).
"""

from __future__ import annotations

import sys
from pathlib import Path

import boto3
import joblib
import numpy as np
import pandas as pd
import pytest
from moto.server import ThreadedMotoServer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.model_loader import LoadedModel  # noqa: E402
from src.data import preprocess as pp  # noqa: E402
from src.data.preprocess import build_features, build_pipeline  # noqa: E402

TEST_VERSION = "20260101T000000Z"


def make_raw_rows(n: int = 240, seed: int = 7, start: str = "2011-01-01") -> pd.DataFrame:
    """Synthesize line-item rows with the raw Online Retail II schema."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        customer = float(10000 + rng.integers(0, 20))
        if rng.random() < 0.1:
            customer = np.nan  # anonymous transaction -> real missing CustomerID
        date = pd.Timestamp(start) + pd.Timedelta(days=int(rng.integers(0, 300)), hours=int(rng.integers(0, 24)))
        for _line in range(int(rng.integers(1, 4))):
            rows.append(
                {
                    "InvoiceNo": f"{500000 + i}",
                    "StockCode": f"{rng.integers(10000, 10010)}",
                    "Description": "GIFT",
                    "Quantity": int(rng.integers(1, 30)),
                    "InvoiceDate": date,
                    "UnitPrice": round(float(rng.uniform(0.1, 20.0)), 2),
                    "CustomerID": customer,
                    "Country": rng.choice(["United Kingdom", "France", "Germany"]),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Small synthetic raw dataset with valid feature columns."""
    return make_raw_rows()


@pytest.fixture
def temp_settings(tmp_path, monkeypatch):
    """Point local model store / artifact dir at a temp dir (resets lru_cache)."""
    from src.config import get_settings

    monkeypatch.setenv("LOCAL_MODEL_DIR", str(tmp_path / "models_local"))
    monkeypatch.setenv("MODEL_SOURCE", "local")
    monkeypatch.setenv("MODEL_VERSION", "")
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path / "artifacts"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def built_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Invoice-level feature frame built from the synthetic raw data."""
    return build_features(raw_df)


@pytest.fixture
def tiny_pipeline(built_df: pd.DataFrame):
    """A real (tiny) sklearn pipeline fitted on the synthetic frame."""
    from sklearn.linear_model import LogisticRegression

    pipeline = build_pipeline(LogisticRegression(max_iter=200))
    pipeline.fit(pp.feature_frame(built_df), pp.target_vector(built_df))
    return pipeline


@pytest.fixture
def schema_dict(built_df: pd.DataFrame) -> dict:
    """Model schema built from the synthetic frame."""
    from src.training.train import build_schema

    return build_schema(built_df)


@pytest.fixture
def loaded_model(tiny_pipeline, schema_dict) -> LoadedModel:
    """A LoadedModel wrapper around the tiny pipeline."""
    return LoadedModel(
        pipeline=tiny_pipeline,
        schema=schema_dict,
        version=TEST_VERSION,
        source="local",
        loaded_at="2026-01-01T00:00:00+00:00",
    )


@pytest.fixture(scope="module")
def moto_server():
    """Module-scoped moto S3 server (endpoint_url based mocking)."""
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=0, verbose=False)
    server.start()
    host, port = server.get_host_and_port()
    yield f"http://{host}:{port}"
    server.stop()


@pytest.fixture
def s3(moto_server):
    """S3 client against the moto server with an emptied standard test bucket."""
    client = boto3.client("s3", endpoint_url=moto_server, region_name="us-east-1",
                          aws_access_key_id="test", aws_secret_access_key="test")
    try:
        client.create_bucket(Bucket="test-models")
    except client.exceptions.BucketAlreadyOwnedByYou:
        pass
    _empty_bucket(client, "test-models")  # isolate each test from prior state
    return client


def _empty_bucket(client, bucket: str) -> None:
    """Delete every object in a moto bucket so tests never see leftovers."""
    contents = client.list_objects_v2(Bucket=bucket).get("Contents", [])
    for obj in contents:
        client.delete_object(Bucket=bucket, Key=obj["Key"])


def artifact_bytes(pipeline, schema: dict) -> dict[str, bytes]:
    """Serialize a minimal artifact set (model + metadata) for store tests."""
    import io
    import json

    buf = io.BytesIO()
    joblib.dump(pipeline, buf)
    return {
        "model.joblib": buf.getvalue(),
        "metrics.json": json.dumps({"roc_auc": 0.7, "version": TEST_VERSION}).encode(),
        "schema.json": json.dumps(schema).encode(),
        "reference_stats.json": json.dumps({"numeric": {}, "categorical": {}}).encode(),
    }
