"""Generate realistic synthetic fraud data for model training.

This script generates synthetic transaction data with realistic fraud patterns
where fraud is correlated with:
- High balance drain ratio (>70%)
- Unusually high amounts compared to user average
- New receivers
- Off-hours transactions (midnight-6am)
- New cards (<3 months)
- High velocity bursts
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "realistic_fraud_dataset.csv"


def generate_normal_transaction() -> dict:
    """Generate a normal (non-fraudulent) transaction.
    
    Normal transactions can occasionally have suspicious-looking features
    (e.g., higher amounts, off-hours) but with LOW balance drain ratio.
    """
    account_balance = np.random.uniform(1000, 500000)  # Higher range
    avg_amount_7d = np.random.uniform(50, 500)
    
    # Normal transactions: 0.3x - 10x of average (occasionally higher)
    # Key: balance drain is always LOW for normal transactions
    amount_multiplier = np.random.choice(
        [np.random.uniform(0.3, 3.0),    # 80%: typical range
         np.random.uniform(3.0, 10.0)],   # 20%: occasional larger purchases
        p=[0.80, 0.20]
    )
    amount = avg_amount_7d * amount_multiplier
    amount = min(amount, account_balance * 0.30)  # Max 30% of balance (key differentiator!)
    
    # Transaction counts - allow more variety
    txn_count_24h = np.random.choice(
        [np.random.randint(0, 5),   # 70%: typical
         np.random.randint(5, 12)],  # 30%: busy day
        p=[0.70, 0.30]
    )
    txn_count_7d = txn_count_24h + np.random.randint(0, 20)
    
    # Card age: weighted towards established, but allow new cards
    card_age_months = np.random.choice(
        range(0, 120),
        p=np.array([1/(i+3) for i in range(120)]) / sum([1/(i+3) for i in range(120)])
    )
    
    # Allow some off-hours for normal transactions (shift workers, travelers)
    hour_probs = np.array([
        0.02, 0.02, 0.02, 0.02, 0.02, 0.03,  # 0-5: some activity
        0.04, 0.05, 0.06, 0.07, 0.08, 0.08,  # 6-11
        0.08, 0.08, 0.07, 0.06, 0.05, 0.05,  # 12-17
        0.05, 0.04, 0.03, 0.02, 0.02, 0.02   # 18-23
    ])
    hour = np.random.choice(range(24), p=hour_probs / hour_probs.sum())
    
    return {
        "transaction_amount": round(amount, 2),
        "account_balance_before": round(account_balance, 2),
        "avg_transaction_amount_7d": round(avg_amount_7d, 2),
        "transaction_count_24h": int(txn_count_24h),
        "transaction_count_7d": int(txn_count_7d),
        "card_age_months": int(card_age_months),
        "card_type": np.random.choice(["DEBIT", "CREDIT"], p=[0.7, 0.3]),
        "is_new_receiver": np.random.choice([0, 1], p=[0.60, 0.40]),  # 40% new receivers is normal
        "previous_fraud_flag": np.random.choice([0, 1], p=[0.98, 0.02]),
        "is_weekend": np.random.choice([0, 1], p=[0.71, 0.29]),
        "txn_hour": hour,
        "fraud_label": 0
    }


def generate_fraudulent_transaction() -> dict:
    """Generate a fraudulent transaction with suspicious patterns."""
    fraud_type = np.random.choice([
        "balance_drain",
        "amount_spike", 
        "velocity_burst",
        "new_card_fraud",
        "combined"
    ], p=[0.25, 0.25, 0.15, 0.15, 0.20])
    
    account_balance = np.random.uniform(5000, 500000)
    avg_amount_7d = np.random.uniform(50, 300)
    
    if fraud_type == "balance_drain":
        # High balance drain: 70-98% of account
        drain_ratio = np.random.uniform(0.70, 0.98)
        amount = account_balance * drain_ratio
        txn_count_24h = np.random.randint(1, 5)
        card_age_months = np.random.randint(3, 60)
        
    elif fraud_type == "amount_spike":
        # Amount way above average: 10x - 100x
        spike_ratio = np.random.uniform(10, 100)
        amount = avg_amount_7d * spike_ratio
        amount = min(amount, account_balance * 0.95)
        txn_count_24h = np.random.randint(0, 3)
        card_age_months = np.random.randint(3, 60)
        
    elif fraud_type == "velocity_burst":
        # Many transactions in short time
        amount = avg_amount_7d * np.random.uniform(2, 8)
        amount = min(amount, account_balance * 0.6)
        txn_count_24h = np.random.randint(10, 30)  # Unusual burst
        card_age_months = np.random.randint(3, 60)
        
    elif fraud_type == "new_card_fraud":
        # Fraud on newly issued cards
        amount = np.random.uniform(1000, 20000)
        amount = min(amount, account_balance * 0.8)
        txn_count_24h = np.random.randint(1, 8)
        card_age_months = np.random.randint(0, 3)  # New card
        
    else:  # combined - worst case scenario
        drain_ratio = np.random.uniform(0.80, 0.98)
        amount = account_balance * drain_ratio
        txn_count_24h = np.random.randint(5, 20)
        card_age_months = np.random.randint(0, 6)
    
    txn_count_7d = txn_count_24h + np.random.randint(0, 10)
    
    # Fraudsters prefer off-hours
    off_hours_probs = np.array([
        0.08, 0.10, 0.12, 0.12, 0.10, 0.06,  # 0-5 (high)
        0.03, 0.02, 0.02, 0.02, 0.02, 0.02,  # 6-11 (low)
        0.02, 0.02, 0.02, 0.02, 0.02, 0.02,  # 12-17 (low)
        0.02, 0.03, 0.04, 0.05, 0.06, 0.07   # 18-23 (rising)
    ])
    hour = np.random.choice(range(24), p=off_hours_probs / off_hours_probs.sum())
    
    return {
        "transaction_amount": round(amount, 2),
        "account_balance_before": round(account_balance, 2),
        "avg_transaction_amount_7d": round(avg_amount_7d, 2),
        "transaction_count_24h": txn_count_24h,
        "transaction_count_7d": txn_count_7d,
        "card_age_months": int(card_age_months),
        "card_type": np.random.choice(["DEBIT", "CREDIT"], p=[0.6, 0.4]),
        "is_new_receiver": np.random.choice([0, 1], p=[0.35, 0.65]),  # Often new receiver
        "previous_fraud_flag": np.random.choice([0, 1], p=[0.70, 0.30]),  # Higher previous fraud
        "is_weekend": np.random.choice([0, 1], p=[0.50, 0.50]),
        "txn_hour": hour,
        "fraud_label": 1
    }


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features to the dataset."""
    result = df.copy()
    
    # Amount ratios
    avg_safe = result["avg_transaction_amount_7d"].replace(0, 1.0)
    result["amount_to_avg_ratio"] = (result["transaction_amount"] / avg_safe).clip(upper=100.0)
    
    balance_safe = result["account_balance_before"].replace(0, 1.0)
    result["balance_drain_ratio"] = (result["transaction_amount"] / balance_safe).clip(upper=1.0)
    
    # Velocity features
    result["velocity_24h"] = result["transaction_count_24h"] / 24.0
    result["velocity_7d"] = result["transaction_count_7d"] / 168.0
    velocity_7d_safe = result["velocity_7d"].replace(0, 0.01)
    result["velocity_burst"] = (result["velocity_24h"] / velocity_7d_safe).clip(upper=50.0)
    
    # Binary indicators
    result["is_new_card"] = (result["card_age_months"] < 3).astype(int)
    result["is_round_amount"] = (result["transaction_amount"] % 100 == 0).astype(int)
    result["is_large_amount"] = (result["transaction_amount"] > 5000).astype(int)
    result["is_off_hours"] = ((result["txn_hour"] < 6) | (result["txn_hour"] > 22)).astype(int)
    result["txn_day_of_week"] = np.random.randint(0, 7, size=len(result))
    
    return result


def generate_dataset(
    n_samples: int = 30000,
    fraud_rate: float = 0.15
) -> pd.DataFrame:
    """Generate a complete dataset with realistic fraud patterns.
    
    Parameters
    ----------
    n_samples : int
        Total number of samples to generate.
    fraud_rate : float
        Proportion of fraudulent transactions (default 15%).
    
    Returns
    -------
    pd.DataFrame
        Generated dataset with fraud labels.
    """
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud
    
    print(f"Generating {n_samples:,} transactions...")
    print(f"  - Normal: {n_normal:,} ({100*(1-fraud_rate):.1f}%)")
    print(f"  - Fraud: {n_fraud:,} ({100*fraud_rate:.1f}%)")
    
    # Generate transactions
    normal_txns = [generate_normal_transaction() for _ in range(n_normal)]
    fraud_txns = [generate_fraudulent_transaction() for _ in range(n_fraud)]
    
    # Combine and shuffle
    all_txns = normal_txns + fraud_txns
    random.shuffle(all_txns)
    
    # Create DataFrame
    df = pd.DataFrame(all_txns)
    
    # Add IDs
    df["transaction_id"] = [f"TXN_{i:06d}" for i in range(len(df))]
    df["user_id"] = [f"USER_{np.random.randint(1000, 9999)}" for _ in range(len(df))]
    
    # Add derived features
    df = add_derived_features(df)
    
    print(f"\nDataset generated successfully!")
    print(f"Fraud rate: {df['fraud_label'].mean():.2%}")
    
    return df


def main():
    """Generate and save realistic fraud dataset."""
    # Generate 30,000 samples with 15% fraud rate
    df = generate_dataset(n_samples=30000, fraud_rate=0.15)
    
    # Display feature statistics by fraud label
    print("\n" + "="*60)
    print("Feature Statistics by Fraud Label:")
    print("="*60)
    
    numeric_cols = [
        "transaction_amount", "balance_drain_ratio", "amount_to_avg_ratio",
        "velocity_burst", "transaction_count_24h", "card_age_months"
    ]
    
    for col in numeric_cols:
        normal_mean = df[df["fraud_label"] == 0][col].mean()
        fraud_mean = df[df["fraud_label"] == 1][col].mean()
        print(f"{col:30s} | Normal: {normal_mean:10.2f} | Fraud: {fraud_mean:10.2f}")
    
    # Save dataset
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nDataset saved to: {OUTPUT_PATH}")
    
    return df


if __name__ == "__main__":
    main()
