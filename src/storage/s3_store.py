"""Timestamp-versioned model artifacts on S3 (or a local directory fallback).

Layout::

    s3://<bucket>/<prefix>/models/<YYYYMMDDTHHMMSSZ>/{model.joblib,metrics.json,
    schema.json,reference_stats.json}
    s3://<bucket>/<prefix>/models/LATEST   # pointer file holding the version id

Versions are immutable: uploading an existing version raises. ``LATEST`` is a
small pointer object, so rollback is a single one-step pointer rewrite.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import boto3
from botocore.exceptions import ClientError

from src.config import get_settings

VERSION_RE = re.compile(r"^\d{8}T\d{6}Z$")
ARTIFACT_FILES = ("model.joblib", "metrics.json", "schema.json", "reference_stats.json")
LATEST = "LATEST"


def new_version_id(dt: datetime | None = None) -> str:
    """Return a UTC timestamp version id like ``20261029T120000Z``."""
    dt = dt or datetime.now(timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _validate_version(version: str) -> None:
    if not VERSION_RE.match(version):
        raise ValueError(
            f"invalid version id {version!r}; expected UTC timestamp YYYYMMDDTHHMMSSZ"
        )


class ModelStore:
    """Store of versioned model artifacts, S3-backed or local-directory-backed.

    Use :meth:`from_settings` to build it; ``MODEL_SOURCE=local`` selects the
    local fallback directory so dev and tests work without AWS credentials.
    """

    def __init__(self, bucket: str, prefix: str, client: Any | None = None):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = client

    @property
    def models_root(self) -> str:
        return f"{self.prefix}/models"

    def key(self, *parts: str) -> str:
        return "/".join([self.models_root, *parts])

    # -- helpers -----------------------------------------------------------
    def _put(self, key: str, body: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=key, Body=body)

    def _get(self, key: str) -> bytes:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except self.client.exceptions.ClientError as e:
            raise FileNotFoundError(f"s3://{self.bucket}/{key} not found") from e

    # -- version API -------------------------------------------------------
    def upload_version(self, version: str, files: dict[str, bytes]) -> str:
        """Upload an immutable artifact set for ``version``.

        Args:
            version: timestamp id matching ``YYYYMMDDTHHMMSSZ``.
            files: mapping of artifact filename -> bytes (subset of
                :data:`ARTIFACT_FILES`).

        Returns:
            The version id.

        Raises:
            ValueError: on bad version format or unknown artifact name.
            FileExistsError: if the version already exists (never overwrite).
        """
        self._check_upload(version, files)
        if version in self.list_versions():
            raise FileExistsError(f"version {version} already exists; versions are immutable")
        self._write_version(version, files)
        return version

    def _write_version(self, version: str, files: dict[str, bytes]) -> None:
        """Persist the artifact bytes for a new version (impl-specific)."""
        raise NotImplementedError

    @staticmethod
    def _check_upload(version: str, files: dict[str, bytes]) -> None:
        """Validate version id and artifact names before an upload."""
        _validate_version(version)
        unknown = set(files) - set(ARTIFACT_FILES)
        if unknown:
            raise ValueError(f"unknown artifact names: {sorted(unknown)}")

    def list_versions(self) -> list[str]:
        """Return all version ids, sorted oldest -> newest."""
        raise NotImplementedError

    def get_latest(self) -> str:
        """Read the LATEST pointer and return its version id."""
        raise NotImplementedError

    def set_latest(self, version: str) -> None:
        """Point LATEST at ``version`` (used by deploy and rollback)."""
        _validate_version(version)
        if version not in self.list_versions():
            raise FileNotFoundError(f"cannot point LATEST at unknown version {version}")


class S3ModelStore(ModelStore):
    """S3-backed implementation."""

    def list_versions(self) -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")
        versions: set[str] = set()
        for page in paginator.paginate(
            Bucket=self.bucket, Prefix=self.models_root + "/", Delimiter="/"
        ):
            for cp in page.get("CommonPrefixes", []):
                name = cp["Prefix"].rstrip("/").split("/")[-1]
                if VERSION_RE.match(name):
                    versions.add(name)
        return sorted(versions)

    def get_latest(self) -> str:
        pointer = self._get(self.key(LATEST)).decode("utf-8").strip()
        _validate_version(pointer)
        return pointer

    def set_latest(self, version: str) -> None:
        super().set_latest(version)
        self._put(self.key(LATEST), version.encode("utf-8"))

    def _write_version(self, version: str, files: dict[str, bytes]) -> None:
        for name, body in files.items():
            self._put(self.key(version, name), body)

    def download_version(self, version: str, dest_dir: Path) -> Path:
        _validate_version(version)
        out = Path(dest_dir) / version
        out.mkdir(parents=True, exist_ok=True)
        for name in ARTIFACT_FILES:
            try:
                body = self._get(self.key(version, name))
            except FileNotFoundError:
                continue
            (out / name).write_bytes(body)
        if not any(out.iterdir()):
            raise FileNotFoundError(f"no artifacts found for version {version}")
        return out


class LocalModelStore(ModelStore):
    """Filesystem-backed store for dev/test (no AWS required).

    ``bucket`` is ignored; ``prefix`` is the root directory.
    """

    def __init__(self, root: Path):
        super().__init__(bucket="local", prefix=str(root))
        self.root = Path(root) / "models"
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def list_versions(self) -> list[str]:
        return sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and VERSION_RE.match(p.name)
        )

    def get_latest(self) -> str:
        pointer = self._path(LATEST)
        if not pointer.exists():
            raise FileNotFoundError(f"{pointer} not found; has any model been uploaded?")
        version = pointer.read_text().strip()
        _validate_version(version)
        return version

    def set_latest(self, version: str) -> None:
        super().set_latest(version)
        self._path(LATEST).write_text(version)

    def _write_version(self, version: str, files: dict[str, bytes]) -> None:
        vdir = self._path(version)
        vdir.mkdir(parents=True)
        for name, body in files.items():
            (vdir / name).write_bytes(body)

    def download_version(self, version: str, dest_dir: Path) -> Path:
        _validate_version(version)
        src = self._path(version)
        if not src.is_dir():
            raise FileNotFoundError(f"version {version} not found under {self.root}")
        out = Path(dest_dir) / version
        if out.resolve() != src.resolve():
            shutil.copytree(src, out, dirs_exist_ok=True)
        return out


def make_s3_client():
    """Build a boto3 S3 client from settings (real AWS credentials required)."""
    s = get_settings()
    return boto3.client("s3", region_name=s.aws_region)


def get_store(source: str | None = None) -> ModelStore:
    """Return the configured model store (``s3`` or ``local``)."""
    s = get_settings()
    source = (source or s.model_source).lower()
    if source == "s3":
        return S3ModelStore(bucket=s.s3_bucket, prefix=s.s3_prefix, client=make_s3_client())
    if source == "local":
        return LocalModelStore(root=s.local_model_dir)
    raise ValueError(f"unknown MODEL_SOURCE {source!r}; use 'local' or 's3'")
