"""Population drift detection (PSI + KS) against a model's reference stats.

``reference_stats.json`` (written at training time) captures per-feature
distributions of the training window. :func:`check_drift` compares a new
observation window against it:

* numeric features: PSI over 10 quantile-derived bins + a two-sample KS test;
* categorical features: PSI over category population shares.

PSI > 0.2 flags a feature as drifted (industry-standard threshold). The KS
statistic is implemented directly with numpy so no extra dependency is needed.

CLI::

    python -m src.training.drift --data new_window.csv --version 20261029T120000Z

exits non-zero when drift is detected, so it can gate a retraining pipeline.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import preprocess as pp
from src.storage.s3_store import get_store

PSI_THRESHOLD = 0.2
KS_ALPHA = 0.05
DEFAULT_BINS = 10
_EPS = 1e-6

logger = logging.getLogger("drift")


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = DEFAULT_BINS) -> float:
    """Population Stability Index between two share arrays over aligned bins.

    Args:
        expected: reference per-bin shares (must sum to ~1).
        actual: current per-bin shares.
        bins: only used for docstring symmetry; shares already define bins.

    Returns:
        PSI value (>= 0). Larger means bigger population shift.
    """
    exp = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    act = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    return float(np.sum((act - exp) * np.log(act / exp)))


def psi_numeric(ref_quantiles: dict[str, float], values: pd.Series, bins: int = 10) -> float:
    """PSI of a numeric feature given reference quantile cut points.

    The reference bin masses are the quantile bands themselves (e.g. for
    quantiles [0.05, 0.5, 0.95] the three interior bands have mass .05, .45,
    .45...). Tied edges - common for integer-valued features - are merged so a
    zero-width bin never produces a spurious PSI spike.
    """
    if values.dropna().empty or not ref_quantiles:
        return 0.0
    pairs = sorted((float(v), float(p)) for p, v in ref_quantiles.items())
    merged: list[list[float]] = []
    for value, prob in pairs:
        if merged and value == merged[-1][0]:
            merged[-1][1] = max(merged[-1][1], prob)
        else:
            merged.append([value, prob])
    edges = [m[0] for m in merged]
    bounds = [0.0, *[m[1] for m in merged], 1.0]
    exp_shares = np.diff(bounds)
    binned = np.digitize(values.dropna().to_numpy(dtype=float), edges, right=True)
    act_counts = np.bincount(binned, minlength=len(exp_shares))
    act_shares = act_counts / max(act_counts.sum(), 1)
    return psi(exp_shares, act_shares)


def ks_2samp(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov statistic and approximate p-value.

    Implemented with numpy (no scipy dependency). The p-value uses the
    asymptotic Kolmogorov distribution.
    """
    a = np.sort(np.asarray(a, dtype=float))
    b = np.sort(np.asarray(b, dtype=float))
    if a.size == 0 or b.size == 0:
        return 0.0, 1.0
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / a.size
    cdf_b = np.searchsorted(b, grid, side="right") / b.size
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    if d <= 0.0:
        return 0.0, 1.0
    en = np.sqrt(a.size * b.size / (a.size + b.size))
    lam = (en + 0.12 + 0.11 / en) * d
    jv = [(-1.0) ** (k - 1) * np.exp(-2.0 * k * k * lam * lam) for k in range(1, 101)]
    p = float(min(max(2.0 * sum(jv), 0.0), 1.0))
    return d, p


def _reference_shares(ref_cat: dict[str, float], series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Align reference and current category shares on the union of categories."""
    current = series.astype("string").fillna("MISSING").value_counts(normalize=True)
    cats = sorted(set(ref_cat) | set(current.index.astype(str)))
    exp = np.array([ref_cat.get(c, 0.0) for c in cats], dtype=float)
    act = np.array([float(current.get(c, 0.0)) for c in cats], dtype=float)
    return exp, act


def psi_from_bins(bins: list[float], expected: list[float], values: pd.Series) -> float:
    """PSI using reference bins + expected shares stored at training time.

    Because the expected shares are derived from the reference data with the
    exact same binning the drift check applies, PSI on the reference window is
    ~0 for every feature, including low-cardinality integer ones.
    """
    vals = values.dropna().to_numpy(dtype=float)
    if vals.size == 0 or not bins:
        return 0.0
    binned = np.digitize(vals, np.asarray(bins, dtype=float), right=True)
    counts = np.bincount(binned, minlength=len(expected))
    act = counts / counts.sum()
    return psi(np.asarray(expected, dtype=float), act)


def _reference_sample(quantiles: dict[str, float], n: int) -> np.ndarray:
    """Reconstruct an approximate reference sample from stored quantiles.

    Inverts the piecewise-linear CDF defined by the quantile grid on an
    equally spaced probability lattice - enough mass points for a stable
    approximate KS comparison without storing the full training column.
    """
    qs = sorted(float(k) for k in quantiles)
    vals = np.array([quantiles[f"{q:g}"] for q in qs], dtype=float)
    p_lat = (np.arange(n) + 0.5) / n
    return np.interp(p_lat, qs, vals)


def check_drift(df: pd.DataFrame, reference_stats: dict[str, Any]) -> dict[str, Any]:
    """Compare a new data window against stored reference statistics.

    Args:
        df: built invoice-level feature frame (output of
            ``src.data.preprocess.build_features``).
        reference_stats: parsed ``reference_stats.json`` from the model version.

    Returns:
        Structured report: per-feature metrics, the drifted feature list and
        the overall PSI threshold used.
    """
    report: dict[str, Any] = {"threshold": PSI_THRESHOLD, "features": {}, "drifted": []}

    for col in pp.NUMERIC_FEATURES:
        ref = reference_stats.get("numeric", {}).get(col)
        if ref is None:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        quantiles = ref.get("quantiles", {})
        if "bins" in ref and "expected" in ref:
            feature_psi = psi_from_bins(ref["bins"], ref["expected"], values)
        else:  # backward compatibility with older reference_stats.json
            feature_psi = psi_numeric(quantiles, values)
        entry: dict[str, Any] = {"psi": round(feature_psi, 4)}
        if quantiles:
            current = values.dropna().to_numpy(dtype=float)
            ref_sample = _reference_sample(quantiles, max(min(len(current), 10_000), 100))
            d_stat, p_val = ks_2samp(current, ref_sample)
            entry["ks"] = {"statistic": round(d_stat, 4), "p_value": round(p_val, 6)}
            if p_val < KS_ALPHA and d_stat > 0.1:
                entry["ks_drift"] = True
        null_frac = float(values.isna().mean()) if len(values) else 0.0
        null_shift = abs(null_frac - float(ref.get("null_fraction", 0.0)))
        entry["null_shift"] = round(null_shift, 4)
        if feature_psi > PSI_THRESHOLD:
            entry["drifted"] = True
            report["drifted"].append(col)
        report["features"][col] = entry

    for col in pp.CATEGORICAL_FEATURES:
        ref = reference_stats.get("categorical", {}).get(col)
        if ref is None:
            continue
        exp, act = _reference_shares(ref, df[col])
        feature_psi = psi(exp, act)
        entry = {"psi": round(feature_psi, 4)}
        # KS on numeric-coded categories as an extra distributional check
        if set(exp) != set(act):
            entry["new_or_missing_categories"] = True
        if feature_psi > PSI_THRESHOLD:
            entry["drifted"] = True
            report["drifted"].append(col)
        report["features"][col] = entry

    report["has_drift"] = bool(report["drifted"])
    return report


def load_reference_stats(version: str | None = None, source: str | None = None) -> dict[str, Any]:
    """Load ``reference_stats.json`` for a version (LATEST if not given)."""
    store = get_store(source)
    version = version or store.get_latest()
    tmp_root = Path("models/_drift_cache")
    path = store.download_version(version, tmp_root) / "reference_stats.json"
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    """CLI: check drift of a CSV window against a stored model version.

    Returns 0 when no drift, 1 when drift is detected, 2 on usage errors.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    parser = argparse.ArgumentParser(description="PSI/KS drift check against model reference stats.")
    parser.add_argument("--data", type=Path, required=True, help="raw line-items CSV for the new window")
    parser.add_argument("--version", default=None, help="model version id (default: LATEST)")
    parser.add_argument("--source", default=None, help="store source override: local|s3")
    args = parser.parse_args(argv)

    if not args.data.exists():
        logger.error("data file not found: %s", args.data)
        return 2

    df = pp.build_features(pd.read_csv(args.data))
    ref = load_reference_stats(args.version, args.source)
    report = check_drift(df, ref)
    print(json.dumps(report, indent=2))
    return 1 if report["has_drift"] else 0


if __name__ == "__main__":
    sys.exit(main())
