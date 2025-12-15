
"""Shared feature engineering utilities for fraud detection models."""

from __future__ import annotations

from typing import Final, Iterable, List

import pandas as pd


RAW_COLUMNS: Final[List[str]] = [
    "user_id",
    "amount",
    "timestamp",
    "account_balance",
    "previous_fraudulent_activity",
    "daily_transaction_count",
    "avg_transaction_amount_7d",
    "failed_transaction_count_7d",
    "card_type",
    "card_age",
    "risk_score",
    "is_weekend",
]

FEATURE_COLUMNS: Final[List[str]] = [
    "user_id",
    "amount",
    "account_balance",
    "previous_fraudulent_activity",
    "daily_transaction_count",
    "avg_transaction_amount_7d",
    "failed_transaction_count_7d",
    "card_type",
    "card_age",
    "risk_score",
    "is_weekend",
    "txn_hour",
    "txn_day_of_week",
    "txn_month",
]


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(sorted(missing))}")


def transform_raw_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw transaction fields into model-ready features.

    Parameters
    ----------
    df:
        DataFrame containing at least the columns listed in :data:`RAW_COLUMNS`.

    Returns
    -------
    pd.DataFrame
        DataFrame with feature columns defined in :data:`FEATURE_COLUMNS`.
    """
    _validate_columns(df, RAW_COLUMNS)

    features = df.copy()
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True, errors="coerce")
    if features["timestamp"].isna().any():
        raise ValueError("Timestamp column contains invalid values.")

    features["txn_hour"] = features["timestamp"].dt.hour.astype("int16")
    features["txn_day_of_week"] = features["timestamp"].dt.dayofweek.astype("int16")
    features["txn_month"] = features["timestamp"].dt.month.astype("int16")

    features = features.drop(columns=["timestamp"])

    # Ensure consistent column order
    features = features.reindex(columns=FEATURE_COLUMNS)
    return features


__all__ = ["RAW_COLUMNS", "FEATURE_COLUMNS", "transform_raw_transactions"]
