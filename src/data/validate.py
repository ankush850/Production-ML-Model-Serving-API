"""Schema, range and null validation for raw transaction data.

All failures raise :class:`DataValidationError` carrying a typed, structured
detail list so the API and CLIs can surface actionable messages instead of
raw tracebacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# columns that must always be fully present (no nulls allowed)
NON_NULL_COLUMNS = ("InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice")

# acceptable value ranges for the raw columns
RANGE_CHECKS: dict[str, tuple[float | None, float | None]] = {
    "Quantity": (1, None),  # negatives/zero are filtered before validation
    "UnitPrice": (0.0, None),
}

# maximum tolerated null fraction for non-critical columns
MAX_NULL_FRACTION = 0.95


@dataclass
class ValidationIssue:
    """One concrete problem found in the data."""

    column: str
    issue: str
    count: int | None = None

    def as_dict(self) -> dict[str, Any]:
        """Serialize for API/error payloads."""
        out: dict[str, Any] = {"column": self.column, "issue": self.issue}
        if self.count is not None:
            out["count"] = self.count
        return out


@dataclass
class DataSchema:
    """Required columns and their expected pandas dtypes."""

    required_columns: tuple[str, ...]
    numeric_columns: tuple[str, ...] = ()
    date_columns: tuple[str, ...] = ()
    categorical_columns: tuple[str, ...] = ()

    def categories(self) -> dict[str, set[str]]:
        """Optional per-column allow-lists, filled by callers if needed."""
        return {}


DEFAULT_SCHEMA = DataSchema(
    required_columns=(
        "InvoiceNo",
        "StockCode",
        "Description",
        "Quantity",
        "InvoiceDate",
        "UnitPrice",
        "CustomerID",
        "Country",
    ),
    numeric_columns=("Quantity", "UnitPrice"),
    date_columns=("InvoiceDate",),
    categorical_columns=("StockCode", "Description", "Country"),
)


class DataValidationError(ValueError):
    """Raised when raw data violates the expected contract."""

    def __init__(self, issues: list[ValidationIssue]):
        self.issues = issues
        detail = "; ".join(f"{i.column}: {i.issue}" for i in issues)
        super().__init__(f"data validation failed - {detail}")

    def details(self) -> list[dict[str, Any]]:
        """Structured details for API error payloads."""
        return [i.as_dict() for i in self.issues]


@dataclass
class ValidationResult:
    """Outcome of a validation run."""

    issues: list[ValidationIssue] = field(default_factory=list)
    n_rows: int = 0

    @property
    def ok(self) -> bool:
        return not self.issues

    def raise_if_invalid(self) -> None:
        """Raise :class:`DataValidationError` if any issue was found."""
        if not self.ok:
            raise DataValidationError(self.issues)


def validate(df: pd.DataFrame, schema: DataSchema = DEFAULT_SCHEMA) -> ValidationResult:
    """Validate schema, dtypes, ranges and null counts of a raw dataframe.

    Args:
        df: dataframe of raw line-item transactions.
        schema: expected columns and dtypes.

    Returns:
        ValidationResult with every issue found (empty when data is valid).
    """
    issues: list[ValidationIssue] = []

    missing_cols = [c for c in schema.required_columns if c not in df.columns]
    for col in missing_cols:
        issues.append(ValidationIssue(col, "required column missing"))

    for col in schema.numeric_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(ValidationIssue(col, "expected numeric dtype"))

    for col in schema.date_columns:
        if col in df.columns and not pd.api.types.is_datetime64_any_dtype(df[col]):
            issues.append(ValidationIssue(col, "expected datetime dtype"))

    for col in NON_NULL_COLUMNS:
        if col in df.columns:
            n_null = int(df[col].isna().sum())
            if n_null:
                issues.append(ValidationIssue(col, "nulls not allowed here", n_null))

    for col, (lo, hi) in RANGE_CHECKS.items():
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            series = df[col].dropna()
            bad = pd.Series(False, index=series.index)
            if lo is not None:
                bad |= series < lo
            if hi is not None:
                bad |= series > hi
            n_bad = int(bad.sum())
            if n_bad:
                issues.append(ValidationIssue(col, f"values outside [{lo}, {hi}]", n_bad))

    for col in schema.categorical_columns:
        if col in df.columns:
            frac = float(df[col].isna().mean()) if len(df) else 0.0
            if frac > MAX_NULL_FRACTION:
                issues.append(ValidationIssue(col, f"null fraction {frac:.2f} too high"))

    return ValidationResult(issues=issues, n_rows=len(df))
