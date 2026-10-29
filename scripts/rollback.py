"""One-step rollback of the LATEST model pointer.

Usage::

    python scripts/rollback.py --version 20261028T063000Z [--source local|s3] [--list]

Validates that the target version exists, repoints LATEST and prints the
previous and new version.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow ``python scripts/rollback.py`` from the repo root, not only ``python -m``.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage.s3_store import get_store  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """CLI entry; returns 0 on success, 2 on unknown version or bad input."""
    parser = argparse.ArgumentParser(description="Roll the LATEST pointer to a prior version.")
    parser.add_argument("--version", help="target version id (YYYYMMDDTHHMMSSZ)")
    parser.add_argument("--source", default=None, choices=["local", "s3"], help="store override")
    parser.add_argument("--list", action="store_true", help="list available versions and exit")
    args = parser.parse_args(argv)

    store = get_store(args.source)
    versions = store.list_versions()

    if args.list:
        print("available versions (oldest -> newest):")
        for v in versions:
            print(f"  {v}")
        try:
            print(f"LATEST -> {store.get_latest()}")
        except FileNotFoundError:
            print("LATEST -> (unset)")
        return 0

    if not args.version:
        parser.error("--version is required unless --list is used")
        return 2

    if args.version not in versions:
        print(f"error: version {args.version} not found. available: {versions}", file=sys.stderr)
        return 2

    try:
        previous = store.get_latest()
    except FileNotFoundError:
        previous = "(unset)"

    store.set_latest(args.version)
    print(f"rollback complete: LATEST {previous} -> {args.version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
