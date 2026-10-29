"""Load a versioned model pipeline + schema from S3 or the local store.

Resolution order:
1. an explicitly pinned ``MODEL_VERSION`` if set;
2. otherwise the ``LATEST`` pointer in the chosen store.

The pipeline is materialized into a local cache directory and loaded with
joblib. A small ``LoadedModel`` wrapper carries the artifacts the API needs
(pipeline, schema, version, load time, source).

.. note::
   ``joblib.load`` deserializes Python objects and can execute code, so only
   models produced by our own trusted training pipeline (our S3 bucket /
   local artifact dir) may be loaded here. Never point ``S3_BUCKET`` /
   ``LOCAL_MODEL_DIR`` at untrusted sources.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from src.config import get_settings
from src.storage.s3_store import ModelStore, get_store

logger = logging.getLogger("model_loader")

CACHE_DIR = Path("models/_serve_cache")


@dataclass
class LoadedModel:
    """In-memory model plus the metadata the API serves back."""

    pipeline: Any
    schema: dict[str, Any]
    version: str
    source: str
    loaded_at: str

    @property
    def feature_columns(self) -> list[str]:
        return [*self.schema["numeric"], *self.schema["categorical"]]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_model(store: ModelStore | None = None, version: str | None = None) -> LoadedModel:
    """Load a model version from the configured store.

    Args:
        store: model store to read from (defaults to the configured one).
        version: explicit version id; falls back to settings then LATEST.

    Returns:
        A :class:`LoadedModel` ready for prediction.

    Raises:
        FileNotFoundError: if no version/LATEST can be resolved or downloaded.
        RuntimeError: if the artifact directory is missing model/schema files.
    """
    settings = get_settings()
    store = store or get_store()
    version = version or settings.model_version or store.get_latest()

    local_dir = store.download_version(version, CACHE_DIR)
    model_path = local_dir / "model.joblib"
    schema_path = local_dir / "schema.json"
    if not model_path.exists() or not schema_path.exists():
        raise RuntimeError(f"version {version} is missing model.joblib or schema.json in {local_dir}")

    pipeline = joblib.load(model_path)
    schema = _read_json(schema_path)
    loaded_at = datetime.now(timezone.utc).isoformat()
    logger.info("loaded model version %s from %s", version, settings.model_source)
    return LoadedModel(
        pipeline=pipeline,
        schema=schema,
        version=version,
        source=settings.model_source,
        loaded_at=loaded_at,
    )
