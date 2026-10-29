"""Pydantic request/response/error models for the serving API.

Every model carries JSON examples so /docs shows ready-to-use payloads.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, RootModel


FEATURE_EXAMPLE: dict[str, Any] = {
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


class FeatureInput(BaseModel):
    """One invoice-level feature record for prediction."""

    model_config = ConfigDict(json_schema_extra={"examples": [FEATURE_EXAMPLE]})

    total_quantity: float = Field(..., ge=1, description="Sum of item quantities on the invoice")
    n_line_items: float = Field(..., ge=1, description="Number of line items on the invoice")
    mean_unit_price: float = Field(..., ge=0, description="Mean unit price across line items")
    order_value: float = Field(..., ge=0, description="Total invoice value (quantity x price)")
    days_since_last_order: float | None = Field(
        default=None,
        ge=0,
        le=3650,
        description="Days since this customer's previous order; null for first orders",
    )
    order_hour: float = Field(..., ge=0, le=23, description="Hour of day the invoice was raised")
    country: str | None = Field(default=None, max_length=64, description="Billing country")
    month: str = Field(..., description="Month of the invoice, '1'..'12'")
    day_of_week: str = Field(..., description="Weekday of the invoice, '0' (Mon)..'6' (Sun)")


class PredictRequest(BaseModel):
    """POST /predict body."""

    model_config = ConfigDict(json_schema_extra={"examples": [FEATURE_EXAMPLE]})

    features: FeatureInput


class PredictResponse(BaseModel):
    """POST /predict response."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {"prediction": 1, "probability": 0.63, "model_version": "20261029T064500Z"}
            ]
        }
    )

    prediction: int = Field(..., description="Predicted class (0 = no reorder, 1 = reorder)")
    probability: float = Field(..., description="P(reorder within 30 days)")
    model_version: str


class BatchRequest(RootModel[list[FeatureInput]]):
    """POST /predict/batch body: a list of feature records."""

    root: list[FeatureInput]


class BatchResponse(BaseModel):
    """POST /predict/batch response."""

    model_config = ConfigDict(protected_namespaces=())

    results: list[PredictResponse]
    model_version: str
    size: int


class HealthResponse(BaseModel):
    """GET /health response."""

    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra={
            "examples": [
                {
                    "status": "ok",
                    "model_version": "20261029T064500Z",
                    "loaded_at": "2026-10-29T07:00:00+00:00",
                    "model_source": "local",
                }
            ]
        }
    )

    status: str
    model_version: str
    loaded_at: str
    model_source: str


class ErrorDetail(BaseModel):
    field: str
    issue: str


class ErrorResponse(BaseModel):
    """Structured error envelope returned for 400/404/422/500/503."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "error_code": "VALIDATION_ERROR",
                    "message": "request body failed validation",
                    "details": [{"field": "order_hour", "issue": "ensure this value is less than or equal to 23"}],
                }
            ]
        }
    )

    error_code: str
    message: str
    details: list[ErrorDetail] = Field(default_factory=list)
