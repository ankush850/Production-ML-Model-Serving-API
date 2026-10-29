"""Training entrypoint: time-based split, model comparison, versioned artifacts.

Running ``python -m src.training.train`` will:

1. load the raw transactions CSV, validate and engineer invoice-level features;
2. split by time (older train, newer test) - never random;
3. fit LogisticRegression / RandomForest / GradientBoosting full pipelines and
   keep the best by ROC-AUC on the held-out time slice;
4. write an immutable timestamp-versioned artifact set (model.joblib,
   metrics.json, schema.json, reference_stats.json) to the local store and,
   optionally, upload it to S3, then move the LATEST pointer to it.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.config import get_settings
from src.data import preprocess as pp
from src.data.validate import validate
from src.storage.s3_store import get_store, new_version_id

logger = logging.getLogger("train")

CANDIDATE_MODELS: dict[str, Any] = {
    "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "random_forest": RandomForestClassifier(
        n_estimators=150, random_state=42, class_weight="balanced", n_jobs=-1
    ),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=150, random_state=42),
}


def _git_commit() -> str:
    """Best-effort current git commit hash (empty if not in a repo)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - metadata only, never fail training for this
        return ""


def load_and_prepare(raw_csv: Path) -> pd.DataFrame:
    """Read raw line items, validate ranges/schema and build invoice features."""
    raw = pd.read_csv(raw_csv)
    result = validate(raw)
    if not result.ok:
        logger.warning("validation reported %d issue(s); engineering proceeds", len(result.issues))
    return pp.build_features(raw)


def time_split(df: pd.DataFrame, frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split rows so train is strictly older than test by the time column.

    Args:
        df: built feature frame containing the ``InvoiceDate`` column.
        frac: fraction of the *timeline* (not rows) reserved for testing.

    Returns:
        (train_df, test_df, cutoff) where cutoff is the first test timestamp.
    """
    times = df[pp.TIME_COL]
    cutoff = times.min() + (times.max() - times.min()) * (1 - frac)
    train = df[times < cutoff].reset_index(drop=True)
    test = df[times >= cutoff].reset_index(drop=True)
    return train, test, cutoff


def build_schema(df: pd.DataFrame) -> dict[str, Any]:
    """Describe feature names, dtypes and allowed ranges/categories."""
    schema: dict[str, Any] = {"numeric": {}, "categorical": {}}
    for col in pp.NUMERIC_FEATURES:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        schema["numeric"][col] = {
            "dtype": "float",
            "min": float(series.min()) if len(series) else None,
            "max": float(series.max()) if len(series) else None,
        }
    for col in pp.CATEGORICAL_FEATURES:
        schema["categorical"][col] = {
            "dtype": "string",
            "allowed": sorted(str(v) for v in df[col].dropna().unique()),
        }
    schema["target"] = pp.TARGET
    return schema


def build_reference_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Persist per-feature distributions used later by the drift check."""
    stats: dict[str, Any] = {"numeric": {}, "categorical": {}}
    qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    for col in pp.NUMERIC_FEATURES:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        entry: dict[str, Any] = {
            "quantiles": {f"{q:g}": float(v) for q, v in zip(qs, series.quantile(qs), strict=True)}
            if len(series)
            else {},
            "null_fraction": float(df[col].isna().mean()),
            "n": int(len(series)),
        }
        # store drift bins + expected shares computed the same way the drift
        # check reads them, so PSI on the reference window is 0 by construction
        # (low-cardinality integer features would otherwise inflate tied bins).
        if len(series):
            edges = np.unique(series.quantile(qs).to_numpy(dtype=float))
            if edges.size:
                binned = np.digitize(series.to_numpy(dtype=float), edges, right=True)
                counts = np.bincount(binned, minlength=edges.size + 1)
                entry["bins"] = [float(e) for e in edges]
                entry["expected"] = [float(c) for c in counts / counts.sum()]
        stats["numeric"][col] = entry
    for col in pp.CATEGORICAL_FEATURES:
        vc = df[col].astype("string").fillna("MISSING").value_counts(normalize=True)
        stats["categorical"][col] = {str(k): float(v) for k, v in vc.items()}
    return stats


def train_models(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[str, Any, dict[str, float]]:
    """Fit every candidate and return the best by ROC-AUC on the test slice."""
    Xtr, ytr = pp.feature_frame(train), pp.target_vector(train)
    Xte, yte = pp.feature_frame(test), pp.target_vector(test)
    scores: dict[str, float] = {}
    fitted: dict[str, Any] = {}
    for name, estimator in CANDIDATE_MODELS.items():
        pipeline = pp.build_pipeline(estimator)
        pipeline.fit(Xtr, ytr)
        proba = pipeline.predict_proba(Xte)[:, 1]
        scores[name] = float(roc_auc_score(yte, proba))
        fitted[name] = pipeline
        logger.info("trained %-22s ROC-AUC=%.4f", name, scores[name])
    best = max(scores, key=scores.get)
    return best, fitted[best], scores


def make_artifacts(pipeline: Any, metrics: dict, schema: dict, ref_stats: dict) -> dict[str, bytes]:
    """Serialize the four artifacts to a name -> bytes mapping."""
    return {
        "model.joblib": _to_bytes(pipeline),
        "metrics.json": json.dumps(metrics, indent=2).encode("utf-8"),
        "schema.json": json.dumps(schema, indent=2).encode("utf-8"),
        "reference_stats.json": json.dumps(ref_stats, indent=2).encode("utf-8"),
    }


def _to_bytes(obj: Any) -> bytes:
    import io

    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.getvalue()


def run_training(
    raw_csv: Path,
    store_source: str = "local",
    upload_s3: bool = False,
    test_frac: float = 0.2,
    version: str | None = None,
) -> dict[str, Any]:
    """Full training flow returning the metrics dict (also the new version id).

    Args:
        raw_csv: path to the raw line-items CSV.
        store_source: local store root uses settings.local_model_dir.
        upload_s3: if True also upload the version to S3 and set LATEST.
        test_frac: timeline fraction reserved for the held-out test set.
        version: force a specific version id (else generated timestamp).

    Returns:
        The metrics dict written to metrics.json.
    """
    df = load_and_prepare(raw_csv)
    train, test, cutoff = time_split(df, test_frac)
    logger.info(
        "time split: train=%d rows (< %s), test=%d rows, cutoff=%s",
        len(train), cutoff, len(test), cutoff,
    )
    if train[pp.TARGET].nunique() < 2 or test[pp.TARGET].nunique() < 2:
        raise ValueError("time split produced a class-poor split; widen the test window")

    best_name, best_model, scores = train_models(train, test)
    test_roc = scores[best_name]

    version = version or new_version_id()
    metrics = {
        "model_type": best_name,
        "candidate_roc_auc": scores,
        "roc_auc": test_roc,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_pos_rate": float(train[pp.TARGET].mean()),
        "test_pos_rate": float(test[pp.TARGET].mean()),
        "features": pp.NUMERIC_FEATURES + pp.CATEGORICAL_FEATURES,
        "sklearn_version": _sklearn_version(),
        "split_cutoff": cutoff.isoformat(),
        "git_commit": _git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "version": version,
    }
    schema = build_schema(df)
    ref_stats = build_reference_stats(train)
    artifacts = make_artifacts(best_model, metrics, schema, ref_stats)

    local_store = get_store("local")
    local_store.upload_version(version, artifacts)
    local_store.set_latest(version)
    logger.info("wrote local version %s -> %s", version, local_store.root)

    if upload_s3:
        s3_store = get_store("s3")
        s3_store.upload_version(version, artifacts)
        s3_store.set_latest(version)
        logger.info("uploaded version %s to s3://%s", version, s3_store.bucket)

    return metrics


def _sklearn_version() -> str:
    import sklearn

    return sklearn.__version__


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper around :func:`run_training`."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    s = get_settings()
    parser = argparse.ArgumentParser(description="Train and version the retention model.")
    parser.add_argument("--data", type=Path, default=s.raw_data_path, help="raw line-items CSV")
    parser.add_argument("--upload-s3", action="store_true", help="also push the version to S3")
    parser.add_argument("--test-frac", type=float, default=0.2, help="timeline fraction held out")
    parser.add_argument("--version", default=None, help="force a specific version id")
    args = parser.parse_args(argv)

    metrics = run_training(
        raw_csv=args.data,
        store_source=s.model_source,
        upload_s3=args.upload_s3,
        test_frac=args.test_frac,
        version=args.version,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
