"""Latency benchmark for the /predict endpoint.

Warm-up, then N sequential POSTs against a running service; prints p50/p95/p99
and the error count.

Usage::

    python scripts/benchmark_latency.py [--url http://localhost:8000] [--n 1000]

Start the service first (``make serve`` or the Docker container).
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from typing import Any

import httpx

PAYLOAD: dict[str, Any] = {
    "features": {
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
}


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile of a list of latencies (ms)."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def benchmark(url: str, n: int, warmup: int = 20, timeout: float = 30.0) -> dict[str, float | int]:
    """Run warm-up + N timed requests and return latency percentiles (ms)."""
    endpoint = url.rstrip("/") + "/predict"
    errors = 0
    latencies: list[float] = []
    with httpx.Client(timeout=timeout) as client:
        for i in range(warmup + n):
            start = time.perf_counter()
            try:
                resp = client.post(endpoint, json=PAYLOAD)
                resp.raise_for_status()
            except Exception:  # noqa: BLE001 - count any failed request as an error
                errors += 1
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if i >= warmup:
                latencies.append(elapsed_ms)
    return {
        "n": n,
        "errors": errors,
        "p50": _percentile(latencies, 0.50),
        "p95": _percentile(latencies, 0.95),
        "p99": _percentile(latencies, 0.99),
        "mean": statistics.fmean(latencies) if latencies else float("nan"),
        "max": max(latencies) if latencies else float("nan"),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI wrapper printing the benchmark summary."""
    parser = argparse.ArgumentParser(description="Benchmark /predict latency percentiles.")
    parser.add_argument("--url", default="http://localhost:8000", help="base URL of the service")
    parser.add_argument("--n", type=int, default=1000, help="number of measured requests")
    parser.add_argument("--warmup", type=int, default=20, help="unmeasured warm-up requests")
    args = parser.parse_args(argv)

    result = benchmark(args.url, args.n, args.warmup)
    print(
        f"requests={result['n']} errors={result['errors']}  "
        f"p50={result['p50']:.1f}ms p95={result['p95']:.1f}ms p99={result['p99']:.1f}ms "
        f"mean={result['mean']:.1f}ms max={result['max']:.1f}ms"
    )
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
