"""Transfer-specific feature engineering for fraud detection.

This module provides feature engineering tailored for money transfer fraud detection
using ONLY behavioral and transactional data (no device/IP/authentication features).
"""

from __future__ import annotations

from typing import Final, List

import pandas as pd
import numpy as np


# Raw input columns from Kafka event
TRANSFER_RAW_COLUMNS: Final[List[str]] = [
    "transaction_id",
    "user_id",
    "transaction_amount",
    "transaction_type",
    "merchant_category",
    "card_type",
    "card_age_months",
    "account_balance_before",
    "avg_transaction_amount_7d",
    "transaction_count_24h",
    "transaction_count_7d",
    "previous_fraud_flag",
    "is_new_receiver",
    "is_weekend",
    "timestamp",
]

# Model feature columns after transformation
# Note: We use RATIOS instead of raw amounts for scale-independence
TRANSFER_FEATURE_COLUMNS: Final[List[str]] = [
    # Ratio features (scale-independent)
    "amount_to_avg_ratio",
    "balance_drain_ratio",
    "velocity_24h",
    "velocity_7d",
    "velocity_burst",
    # Card features
    "card_age_months",
    "card_type",
    # Risk indicators
    "is_new_receiver",
    "is_new_card",
    "is_round_amount",
    "is_large_amount",
    "is_off_hours",
    "previous_fraud_flag",
    "is_weekend",
    # Time features
    "txn_hour",
    "txn_day_of_week",
]


def _validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    """Validate that all required columns are present."""
    missing = set(required) - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(sorted(missing))}")


def compute_amount_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Compute amount-based ratio features.
    
    Features:
    - amount_to_avg_ratio: Transaction amount relative to user's 7-day average
    - balance_drain_ratio: Portion of account balance being transferred
    """
    result = df.copy()
    
    # Amount to average ratio (with safeguard for zero average)
    avg_safe = result["avg_transaction_amount_7d"].replace(0, np.nan).fillna(1.0)
    result["amount_to_avg_ratio"] = result["transaction_amount"] / avg_safe
    
    # Cap extreme values for numerical stability
    result["amount_to_avg_ratio"] = result["amount_to_avg_ratio"].clip(upper=100.0)
    
    # Balance drain ratio (with safeguard for zero balance)
    balance_safe = result["account_balance_before"].replace(0, np.nan).fillna(1.0)
    result["balance_drain_ratio"] = result["transaction_amount"] / balance_safe
    
    # Cap at 1.0 (transfer cannot exceed balance in normal cases)
    result["balance_drain_ratio"] = result["balance_drain_ratio"].clip(upper=1.0)
    
    return result


def compute_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute velocity-based fraud indicators.
    
    Features:
    - velocity_24h: Hourly transaction rate over last 24 hours
    - velocity_7d: Hourly transaction rate over last 7 days
    - velocity_burst: Ratio of 24h velocity to 7d velocity (detects burst activity)
    """
    result = df.copy()
    
    # Hourly rates
    result["velocity_24h"] = result["transaction_count_24h"] / 24.0
    result["velocity_7d"] = result["transaction_count_7d"] / 168.0  # 7 days * 24 hours
    
    # Burst detection (safeguard against division by zero)
    velocity_7d_safe = result["velocity_7d"].replace(0, np.nan).fillna(0.01)
    result["velocity_burst"] = result["velocity_24h"] / velocity_7d_safe
    
    # Cap extreme burst values
    result["velocity_burst"] = result["velocity_burst"].clip(upper=50.0)
    
    return result


def compute_risk_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute binary risk indicator features.
    
    Features:
    - is_new_card: Card is less than 3 months old
    - is_round_amount: Transaction is a round number (common in fraud)
    - is_large_amount: Transaction exceeds high-value threshold
    - is_off_hours: Transaction occurs during unusual hours (midnight-6am, after 10pm)
    """
    result = df.copy()
    
    # Card maturity risk
    result["is_new_card"] = (result["card_age_months"] < 3).astype(int)
    
    # Round amount detection (multiples of 100)
    result["is_round_amount"] = (result["transaction_amount"] % 100 == 0).astype(int)
    
    # Large amount threshold (configurable, default 5000)
    LARGE_AMOUNT_THRESHOLD = 5000.0
    result["is_large_amount"] = (result["transaction_amount"] > LARGE_AMOUNT_THRESHOLD).astype(int)
    
    # Off-hours detection
    if "txn_hour" in result.columns:
        result["is_off_hours"] = ((result["txn_hour"] < 6) | (result["txn_hour"] > 22)).astype(int)
    else:
        result["is_off_hours"] = 0  # Default if timestamp not yet processed
    
    return result


def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract time-based features from timestamp.
    
    Features:
    - txn_hour: Hour of transaction (0-23)
    - txn_day_of_week: Day of week (0=Monday, 6=Sunday)
    """
    result = df.copy()
    
    # Parse timestamp if not already datetime
    if "timestamp" in result.columns:
        result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")
        
        if result["timestamp"].isna().any():
            raise ValueError("Timestamp column contains invalid values.")
        
        result["txn_hour"] = result["timestamp"].dt.hour.astype("int16")
        result["txn_day_of_week"] = result["timestamp"].dt.dayofweek.astype("int16")
    
    return result


def transform_transfer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw transfer transaction data into model-ready features.
    
    This is the main entry point for feature engineering. It applies all
    transformations in the correct order and returns only the model features.
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction data containing columns from TRANSFER_RAW_COLUMNS.
    
    Returns
    -------
    pd.DataFrame
        Transformed features matching TRANSFER_FEATURE_COLUMNS.
    
    Raises
    ------
    KeyError
        If required columns are missing from input.
    ValueError
        If timestamp contains invalid values.
    """
    _validate_columns(df, TRANSFER_RAW_COLUMNS)
    
    # Apply transformations in order
    features = df.copy()
    features = compute_time_features(features)
    features = compute_amount_ratios(features)
    features = compute_velocity_features(features)
    features = compute_risk_indicators(features)
    
    # Convert boolean columns to int if needed
    bool_columns = ["previous_fraud_flag", "is_new_receiver", "is_weekend"]
    for col in bool_columns:
        if col in features.columns:
            features[col] = features[col].astype(int)
    
    # Select only model features and ensure consistent order
    model_features = features.reindex(columns=TRANSFER_FEATURE_COLUMNS)
    
    return model_features


def transform_single_transaction(transaction: dict) -> pd.DataFrame:
    """Transform a single transaction dict into model features.
    
    This is a convenience method for inference on single transactions
    from Kafka events.
    
    Parameters
    ----------
    transaction : dict
        Transaction data as a dictionary.
    
    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with model features.
    """
    df = pd.DataFrame([transaction])
    return transform_transfer_features(df)


__all__ = [
    "TRANSFER_RAW_COLUMNS",
    "TRANSFER_FEATURE_COLUMNS",
    "transform_transfer_features",
    "transform_single_transaction",
]
