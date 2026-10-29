"""Data validation tests: schema, dtype, range and null checks."""

from __future__ import annotations

import pytest

from src.data.validate import (
    DEFAULT_SCHEMA,
    DataValidationError,
    validate,
)


class TestSchema:
    def test_valid_frame_passes(self, raw_df):
        result = validate(raw_df)
        assert result.ok
        assert result.n_rows == len(raw_df)

    def test_missing_column_reported(self, raw_df):
        result = validate(raw_df.drop(columns=["Country"]))
        assert not result.ok
        assert any(i.column == "Country" and "missing" in i.issue for i in result.issues)

    def test_custom_schema_respected(self, raw_df):
        from src.data.validate import DataSchema

        schema = DataSchema(required_columns=("InvoiceNo", "nonexistent_col"))
        result = validate(raw_df, schema)
        assert any(i.column == "nonexistent_col" for i in result.issues)
        assert "nonexistent_col" in schema.required_columns


class TestDtypes:
    def test_numeric_column_as_string_fails(self, raw_df):
        bad = raw_df.copy()
        bad["Quantity"] = bad["Quantity"].astype(str)
        result = validate(bad)
        assert any(i.column == "Quantity" and "numeric" in i.issue for i in result.issues)

    def test_date_column_as_string_fails(self, raw_df):
        bad = raw_df.copy()
        bad["InvoiceDate"] = bad["InvoiceDate"].astype(str)
        result = validate(bad)
        assert any(i.column == "InvoiceDate" and "datetime" in i.issue for i in result.issues)


class TestNullChecks:
    @pytest.mark.parametrize("column", ["InvoiceNo", "Quantity", "InvoiceDate", "UnitPrice"])
    def test_nulls_in_required_columns_rejected(self, raw_df, column):
        bad = raw_df.copy()
        bad.loc[bad.index[:3], column] = None
        result = validate(bad)
        assert any(i.column == column and i.count == 3 for i in result.issues)

    def test_high_null_fraction_rejected_for_categoricals(self, raw_df):
        bad = raw_df.copy()
        bad.loc[:, "Country"] = None
        result = validate(bad)
        assert any(i.column == "Country" and "null fraction" in i.issue for i in result.issues)


class TestRangeChecks:
    def test_negative_quantity_rejected(self, raw_df):
        bad = raw_df.copy()
        bad.loc[bad.index[:2], "Quantity"] = -3
        result = validate(bad)
        assert any(i.column == "Quantity" and "outside" in i.issue for i in result.issues)

    def test_negative_price_rejected(self, raw_df):
        bad = raw_df.copy()
        bad.loc[bad.index[0], "UnitPrice"] = -1.0
        result = validate(bad)
        assert any(i.column == "UnitPrice" for i in result.issues)


class TestRaiseBehavior:
    def test_raise_if_invalid_raises_typed_error(self, raw_df):
        bad = raw_df.copy()
        bad.loc[bad.index[:2], "Quantity"] = -1
        result = validate(bad)
        with pytest.raises(DataValidationError) as excinfo:
            result.raise_if_invalid()
        err = excinfo.value
        details = err.details()
        assert any(d["column"] == "Quantity" and d["count"] == 2 for d in details)
        assert all({"column", "issue"} <= set(d) for d in details)

    def test_raise_if_invalid_silent_when_ok(self, raw_df):
        validate(raw_df).raise_if_invalid()

    def test_default_schema_covers_all_raw_columns(self, raw_df):
        assert set(DEFAULT_SCHEMA.required_columns) <= set(raw_df.columns)
