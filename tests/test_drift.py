"""Drift tests: PSI/KS math, no-drift on same data, drift on shifted data,
and the CLI exit codes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import preprocess as pp
from src.training import drift
from src.training.train import build_reference_stats


@pytest.fixture
def reference_stats(built_df: pd.DataFrame) -> dict:
    return build_reference_stats(built_df)


class TestMetrics:
    def test_psi_zero_for_identical_shares(self):
        shares = np.full(10, 0.1)
        assert drift.psi(shares, shares) == pytest.approx(0.0, abs=1e-9)

    def test_psi_large_for_population_shift(self):
        exp = np.full(10, 0.1)
        act = np.zeros(10)
        act[9] = 1.0
        assert drift.psi(exp, act) > 2.0

    def test_ks_detects_shift(self):
        rng = np.random.default_rng(0)
        a = rng.normal(0, 1, 500)
        b = rng.normal(3, 1, 500)
        d_same, p_same = drift.ks_2samp(a, a.copy())
        d_shift, p_shift = drift.ks_2samp(a, b)
        assert d_same == pytest.approx(0.0)
        assert p_same > 0.05
        assert d_shift > 0.5
        assert p_shift < 0.05

    def test_ks_empty_inputs(self):
        assert drift.ks_2samp(np.array([]), np.array([1.0])) == (0.0, 1.0)


class TestCheckDrift:
    def test_no_drift_on_same_data(self, built_df, reference_stats):
        report = drift.check_drift(built_df, reference_stats)
        assert report["has_drift"] is False
        assert report["drifted"] == []
        for entry in report["features"].values():
            assert entry["psi"] <= drift.PSI_THRESHOLD

    def test_drift_detected_on_shifted_data(self, built_df, reference_stats):
        shifted = built_df.copy()
        shifted["total_quantity"] = shifted["total_quantity"] * 50 + 500
        shifted["country"] = "Atlantis"
        report = drift.check_drift(shifted, reference_stats)
        assert report["has_drift"] is True
        assert "total_quantity" in report["drifted"]
        assert "country" in report["drifted"]

    def test_report_structure(self, built_df, reference_stats):
        report = drift.check_drift(built_df, reference_stats)
        assert set(report) == {"threshold", "features", "drifted", "has_drift"}
        numeric_entry = report["features"]["order_value"]
        assert "psi" in numeric_entry and "ks" in numeric_entry


class TestCLI:
    def _prepare(self, tmp_path, monkeypatch, built_df, raw_df):
        from src.config import get_settings

        monkeypatch.setenv("LOCAL_MODEL_DIR", str(tmp_path / "models_local"))
        get_settings.cache_clear()
        from src.storage.s3_store import LocalModelStore

        store = LocalModelStore(root=tmp_path / "models_local")
        version = "20260105T000000Z"
        store.upload_version(
            version,
            {"reference_stats.json": json.dumps(build_reference_stats(built_df)).encode()},
        )
        store.set_latest(version)
        csv = tmp_path / "window.csv"
        raw_df.to_csv(csv, index=False)
        return version, csv

    def test_exit_zero_when_no_drift(self, tmp_path, monkeypatch, built_df, raw_df, capsys):
        version, csv = self._prepare(tmp_path, monkeypatch, built_df, raw_df)
        try:
            code = drift.main(["--data", str(csv), "--version", version, "--source", "local"])
        finally:
            from src.config import get_settings

            get_settings.cache_clear()
        assert code == 0
        assert json.loads(capsys.readouterr().out)["has_drift"] is False

    def test_exit_nonzero_on_drift(self, tmp_path, monkeypatch, built_df, raw_df, capsys):
        version, csv = self._prepare(tmp_path, monkeypatch, built_df, raw_df)
        # rewrite the window with a heavily shifted quantity distribution
        shifted = raw_df.copy()
        shifted["Quantity"] = shifted["Quantity"] * 200
        shifted["Country"] = "Atlantis"
        shifted.to_csv(csv, index=False)
        try:
            code = drift.main(["--data", str(csv), "--source", "local"])
        finally:
            from src.config import get_settings

            get_settings.cache_clear()
        assert code == 1
        assert json.loads(capsys.readouterr().out)["has_drift"] is True

    def test_missing_file_returns_2(self, tmp_path):
        assert drift.main(["--data", str(tmp_path / "nope.csv")]) == 2
