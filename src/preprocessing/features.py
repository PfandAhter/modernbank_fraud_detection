"""Feature extraction utilities for fraud detection preprocessing."""

from __future__ import annotations

from typing import Final

import pandas as pd


ROLLING_WINDOW_30D: Final[str] = "30D"
ROLLING_WINDOW_7D: Final[str] = "7D"


def extract_user_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add user-centric behavioral features to a transactions dataframe.

    Parameters
    ----------
    df:
        Transaction records containing at least ``user_id``, ``amount``, and ``timestamp``.

    Returns
    -------
    pd.DataFrame
        A copy of ``df`` sorted to the original order and augmented with:

        * ``amount_zscore``: Z-score of ``amount`` relative to the user's global mean/std.
        * ``user_mean_30d``: Rolling 30-day mean transaction amount per user.
        * ``user_std_30d``: Rolling 30-day standard deviation of transaction amount per user.
        * ``txn_count_7d``: Rolling 7-day count of transactions per user.
        * ``hours_since_last_tx``: Hours since the user's previous transaction.
    """
    required_columns = {"user_id", "amount", "timestamp"}
    missing = required_columns.difference(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Dataframe is missing required columns: {missing_str}")

    working = df.copy()
    working["_orig_order"] = range(len(working))
    working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")

    if working["timestamp"].isna().any():
        raise ValueError("Timestamp column contains non-convertible values.")

    working.sort_values(["user_id", "timestamp"], inplace=True, kind="mergesort")

    user_groups = working.groupby("user_id", group_keys=False)

    user_mean = user_groups["amount"].transform("mean")
    user_std = user_groups["amount"].transform("std").replace(0.0, pd.NA)
    zscore = (working["amount"] - user_mean) / user_std
    working["amount_zscore"] = zscore.fillna(0.0)

    rolling_30d = user_groups.rolling(ROLLING_WINDOW_30D, on="timestamp")["amount"]
    working["user_mean_30d"] = (
        rolling_30d.mean().reset_index(level=0, drop=True).to_numpy()
    )
    working["user_std_30d"] = (
        rolling_30d.std().reset_index(level=0, drop=True).to_numpy()
    )

    rolling_7d = user_groups.rolling(ROLLING_WINDOW_7D, on="timestamp")["amount"]
    working["txn_count_7d"] = (
        rolling_7d.count().reset_index(level=0, drop=True).to_numpy()
    )

    time_since_last = user_groups["timestamp"].diff().dt.total_seconds() / 3600.0
    working["hours_since_last_tx"] = time_since_last.fillna(-1.0)

    working.sort_values("_orig_order", inplace=True, kind="mergesort")
    working.drop(columns="_orig_order", inplace=True)

    return working
