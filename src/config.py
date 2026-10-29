"""Application settings loaded from environment variables / .env file.

Uses pydantic-settings so every knob (S3 bucket, model source, paths) is
declared in one place and overridable without code changes.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for pipeline, storage and API layers."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", protected_namespaces=()
    )

    # --- S3 / versioning -------------------------------------------------
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    s3_bucket: str = Field(default="ml-models-bucket", alias="S3_BUCKET")
    s3_prefix: str = Field(default="production-ml-serving", alias="S3_PREFIX")

    # --- model loading ---------------------------------------------------
    model_source: str = Field(default="local", alias="MODEL_SOURCE")  # local | s3
    model_version: str = Field(default="", alias="MODEL_VERSION")  # empty = LATEST
    local_model_dir: Path = Field(
        default=Path("models/artifacts"), alias="LOCAL_MODEL_DIR"
    )

    # --- data / artifacts --------------------------------------------------
    raw_data_path: Path = Field(default=Path("data/raw/online_retail.csv"), alias="RAW_DATA_PATH")
    artifact_dir: Path = Field(default=Path("models/artifacts"), alias="ARTIFACT_DIR")

    # --- API ---------------------------------------------------------------
    max_batch_size: int = Field(default=100, alias="MAX_BATCH_SIZE")

    # --- misc ----------------------------------------------------------------
    aws_access_key_id: str | None = Field(default=None, alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(default=None, alias="AWS_SECRET_ACCESS_KEY")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
