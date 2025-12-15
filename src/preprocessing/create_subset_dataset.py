"""Create a reduced fraud dataset with selected columns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = PROJECT_ROOT / "data" / "synthetic_fraud_dataset.csv"
TARGET_PATH = PROJECT_ROOT / "data" / "synthetic_fraud_subset.csv"

SELECTED_COLUMNS = {
    "User_ID": "user_id",
    "Transaction_Amount": "transaction_amount",
    "Timestamp": "timestamp",
    "Account_Balance": "account_balance",
    "Previous_Fraudulent_Activity": "previous_fraudulent_activity",
    "Daily_Transaction_Count": "daily_transaction_count",
    "Avg_Transaction_Amount_7d": "avg_transaction_amount_7d",
    "Failed_Transaction_Count_7d": "failed_transaction_count_7d",
    "Card_Type": "card_type",
    "Card_Age": "card_age",
    "Risk_Score": "risk_score",
    "Is_Weekend": "is_weekend",
    "Fraud_Label": "fraud_label",
}


def main() -> None:
    """Load the full dataset, select/rename columns, and persist a subset CSV."""
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(f"Source dataset not found at {SOURCE_PATH}")

    df = pd.read_csv(SOURCE_PATH, usecols=list(SELECTED_COLUMNS.keys()))
    df.rename(columns=SELECTED_COLUMNS, inplace=True)

    TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TARGET_PATH, index=False)
    print(f"Saved subset dataset to {TARGET_PATH} with shape {df.shape}")


if __name__ == "__main__":
    main()
