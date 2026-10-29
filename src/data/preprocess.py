"""Feature engineering and the sklearn preprocessing/model Pipeline.

Task (derived from the data, nothing fabricated): invoice-level *retention*
prediction - given a customer order, will the same customer place another order
within the next 30 days? This yields a real binary target, a genuine datetime
split column, and natural missing values (a customer's first order has no
``days_since_last_order``; Country can be null).

A single sklearn ``Pipeline`` (ColumnTransformer + estimator) is saved as one
artifact so training and serving can never skew.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "reorder_30d"
TIME_COL = "InvoiceDate"

NUMERIC_FEATURES = [
    "total_quantity",
    "n_line_items",
    "mean_unit_price",
    "order_value",
    "days_since_last_order",  # genuinely missing for a customer's first order
    "order_hour",
]
CATEGORICAL_FEATURES = ["country", "month", "day_of_week"]

REORDER_WINDOW_DAYS = 30


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate line-item transactions into invoice-level modeling features.

    Args:
        raw: line items with InvoiceNo, StockCode, Quantity, InvoiceDate,
            UnitPrice, CustomerID, Country.

    Returns:
        One row per invoice with engineered features plus the ``reorder_30d``
        target and the ``InvoiceDate`` time column used for splitting.
    """
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)

    # drop cancellations and non-positive quantities/prices (returns, errors)
    df = df[~df["InvoiceNo"].str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    # anonymous rows (null CustomerID) become their own pseudo-customer so the
    # "days since last order" feature stays null - i.e. real missing data
    df["CustomerID"] = df["CustomerID"].astype("object")
    anon_mask = df["CustomerID"].isna()
    df.loc[anon_mask, "CustomerID"] = (
        "ANON_" + df.loc[anon_mask, "InvoiceNo"].astype(str)
    )

    # order value = sum(quantity * price) per line item
    df["_line_value"] = df["Quantity"] * df["UnitPrice"]

    agg = (
        df.groupby(["CustomerID", "InvoiceNo"], as_index=False)
        .agg(
            total_quantity=("Quantity", "sum"),
            n_line_items=("StockCode", "count"),
            mean_unit_price=("UnitPrice", "mean"),
            order_value=("_line_value", "sum"),
            InvoiceDate=("InvoiceDate", "max"),
            country=("Country", "first"),
        )
    )

    agg = agg.sort_values(["CustomerID", "InvoiceDate"]).reset_index(drop=True)

    prev = agg.groupby("CustomerID")["InvoiceDate"].shift(1)
    nxt = agg.groupby("CustomerID")["InvoiceDate"].shift(-1)
    agg["days_since_last_order"] = (agg["InvoiceDate"] - prev).dt.days
    agg["days_to_next"] = (nxt - agg["InvoiceDate"]).dt.days

    # target: another order strictly within the window; unknown (no next order)
    # counts as "did not reorder" (0) - conservative and leak-free
    agg[TARGET] = (agg["days_to_next"] <= REORDER_WINDOW_DAYS).astype(int)

    agg["month"] = agg["InvoiceDate"].dt.month.astype(str)
    agg["day_of_week"] = agg["InvoiceDate"].dt.dayofweek.astype(str)
    agg["order_hour"] = agg["InvoiceDate"].dt.hour.astype(float)
    agg["country"] = agg["country"].astype("string")

    features = ["CustomerID", *NUMERIC_FEATURES, *CATEGORICAL_FEATURES, TIME_COL, TARGET]
    return agg[features].reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: median-impute + scale numerics, one-hot categoricals.

    Unknown categories at serving time are ignored (handled by
    ``handle_unknown='ignore'``), and null categoricals are imputed to a
    literal ``MISSING`` token before encoding.
    """
    numeric = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def build_pipeline(estimator) -> Pipeline:
    """Wrap a fitted-to-be estimator together with the preprocessor."""
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return just the modeling feature columns from a built dataframe."""
    return df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()


def target_vector(df: pd.DataFrame) -> np.ndarray:
    """Return the binary target as a numpy array."""
    return df[TARGET].to_numpy().astype(int)
