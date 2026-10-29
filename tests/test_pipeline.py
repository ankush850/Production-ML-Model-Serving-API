"""Pipeline tests: preprocessing robustness, training artifacts, time split,
S3 versioning and rollback (moto)."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data import preprocess as pp
from src.data.preprocess import build_features, build_pipeline
from src.storage.s3_store import LocalModelStore, S3ModelStore
from src.training.train import run_training, time_split
from tests.conftest import artifact_bytes

V1 = "20260101T000000Z"
V2 = "20260102T000000Z"
VERSION_PATTERN = re.compile(r"^\d{8}T\d{6}Z$")


class TestPreprocessing:
    def test_handles_missing_values(self, raw_df):
        built = build_features(raw_df)
        X = pp.feature_frame(built)
        X.loc[X.index[:10], "mean_unit_price"] = np.nan  # inject extra NaNs
        pipeline = build_pipeline(LogisticRegression(max_iter=200))
        pipeline.fit(X, pp.target_vector(built))
        transformed = pipeline["prep"].transform(X)
        assert not np.isnan(transformed).any()

    def test_days_since_last_order_has_real_nans(self, built_df):
        # each pseudo-customer's first invoice must produce a null
        assert built_df["days_since_last_order"].isna().any()

    def test_unseen_category_does_not_break_transform(self, built_df):
        pipeline = build_pipeline(LogisticRegression(max_iter=200))
        pipeline.fit(pp.feature_frame(built_df), pp.target_vector(built_df))
        unseen = pp.feature_frame(built_df).head(3).copy()
        unseen["country"] = "Atlantis"
        probs = pipeline.predict_proba(unseen)
        assert probs.shape[0] == 3

    def test_target_is_binary(self, built_df):
        assert set(built_df[pp.TARGET].unique()) <= {0, 1}


class TestTimeSplit:
    def test_no_leakage(self, built_df):
        train, test, cutoff = time_split(built_df, frac=0.25)
        assert train[pp.TIME_COL].max() <= test[pp.TIME_COL].min()
        assert test[pp.TIME_COL].min() >= cutoff
        assert len(train) + len(test) == len(built_df)


class TestTrainingArtifacts:
    def test_run_training_produces_all_artifacts(self, raw_df, tmp_path, temp_settings):
        from src.config import get_settings

        csv = tmp_path / "raw.csv"
        raw_df.to_csv(csv, index=False)
        metrics = run_training(csv)
        version = metrics["version"]
        store = LocalModelStore(root=get_settings().local_model_dir)
        vdir = store.root / version
        for name in ("model.joblib", "metrics.json", "schema.json", "reference_stats.json"):
            assert (vdir / name).exists(), f"missing {name}"
        written = json.loads((vdir / "metrics.json").read_text())
        assert written["roc_auc"] == pytest.approx(metrics["roc_auc"])
        assert {"logistic_regression", "random_forest", "gradient_boosting"} <= set(
            written["candidate_roc_auc"]
        )
        assert VERSION_PATTERN.match(version)
        assert store.get_latest() == version
        # time split recorded in metrics and leak-free
        cutoff = pd.Timestamp(written["split_cutoff"])
        built = build_features(raw_df)
        assert built[built[pp.TIME_COL] < cutoff].shape[0] == written["n_train"]
        assert built[built[pp.TIME_COL] >= cutoff].shape[0] == written["n_test"]


class TestVersioningS3:
    def _store(self, s3):
        return S3ModelStore(bucket="test-models", prefix="pml", client=s3)

    def test_upload_list_latest_download(self, s3, tiny_pipeline, schema_dict, tmp_path):
        store = self._store(s3)
        files = artifact_bytes(tiny_pipeline, schema_dict)
        store.upload_version(V1, files)
        store.set_latest(V1)
        assert store.list_versions() == [V1]
        assert store.get_latest() == V1
        with pytest.raises(FileExistsError):
            store.upload_version(V1, files)  # versions are immutable
        out = store.download_version(V1, tmp_path / "dl")
        assert (out / "model.joblib").exists()
        assert json.loads((out / "schema.json").read_text()) == schema_dict

    def test_list_sorted_and_rollback_via_pointer(self, s3, tiny_pipeline, schema_dict):
        store = self._store(s3)
        files = artifact_bytes(tiny_pipeline, schema_dict)
        store.upload_version(V2, files)
        store.upload_version(V1, files)
        assert store.list_versions() == [V1, V2]  # sorted oldest -> newest
        store.set_latest(V2)
        # one-step rollback: just move the pointer back
        store.set_latest(V1)
        assert store.get_latest() == V1

    def test_set_latest_rejects_unknown_version(self, s3):
        store = self._store(s3)
        with pytest.raises(FileNotFoundError):
            store.set_latest(V1)

    def test_invalid_version_format_rejected(self, s3):
        store = self._store(s3)
        with pytest.raises(ValueError):
            store.upload_version("v1-final", {})

    def test_local_store_roundtrip(self, tmp_path, tiny_pipeline, schema_dict):
        store = LocalModelStore(root=tmp_path / "store")
        store.upload_version(V1, artifact_bytes(tiny_pipeline, schema_dict))
        store.set_latest(V1)
        assert store.get_latest() == V1
        out = store.download_version(V1, tmp_path / "dl")
        assert (out / "metrics.json").exists()
        with pytest.raises(FileExistsError):
            store.upload_version(V1, artifact_bytes(tiny_pipeline, schema_dict))

    def test_rollback_script_cli(self, tmp_path, tiny_pipeline, schema_dict, capsys, monkeypatch):
        from src.config import get_settings
        from scripts.rollback import main as rollback_main

        monkeypatch.setenv("LOCAL_MODEL_DIR", str(tmp_path / "store"))
        get_settings.cache_clear()
        try:
            store = LocalModelStore(root=tmp_path / "store")
            files = artifact_bytes(tiny_pipeline, schema_dict)
            store.upload_version(V1, files)
            store.upload_version(V2, files)
            store.set_latest(V2)

            assert rollback_main(["--version", V1, "--source", "local"]) == 0
            out = capsys.readouterr().out
            assert f"LATEST {V2} -> {V1}" in out
            assert store.get_latest() == V1

            # unknown version -> exit code 2, pointer unchanged
            assert rollback_main(["--version", "19990101T000000Z", "--source", "local"]) == 2
            assert store.get_latest() == V1
        finally:
            get_settings.cache_clear()
