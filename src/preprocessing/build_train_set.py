"""Prepare processed training data from the synthetic fraud subset."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR.parent) not in sys.path:
    sys.path.append(str(SCRIPT_DIR.parent))

from preprocessing.model_features import FEATURE_COLUMNS, RAW_COLUMNS, transform_raw_transactions


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = PROJECT_ROOT / "data" / "synthetic_fraud_subset.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"


def load_raw() -> pd.DataFrame:
    """Load the raw CSV and normalise column names."""
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_PATH}")

    df = pd.read_csv(RAW_PATH)
    df.rename(columns={"transaction_amount": "amount"}, inplace=True)

    missing = [col for col in RAW_COLUMNS + ["fraud_label"] if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    df["fraud_label"] = df["fraud_label"].astype(int)
    return df


def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Derive model features and append the fraud label."""
    features = transform_raw_transactions(df_raw[RAW_COLUMNS])
    features["is_fraud"] = df_raw["fraud_label"].astype(int)
    return features


def main() -> None:
    """Entry point for preprocessing the training dataset."""
    df_raw = load_raw()
    print(f"Raw shape: {df_raw.shape}")
    print(f"Columns: {df_raw.columns.tolist()}")
    print(f"Fraud rate: {df_raw['fraud_label'].mean():.5f}")

    df_processed = preprocess(df_raw)
    print(f"Processed columns: {df_processed.columns.tolist()}")
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_parquet(PROCESSED_PATH, index=False)
    print(f"Saved processed features to {PROCESSED_PATH} ({df_processed.shape})")


if __name__ == "__main__":
    main()
