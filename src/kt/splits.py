"""Chronological split within each user.

Riiid timestamps are milliseconds since each user's own first event, so a
single global cutoff is meaningless; the honest protocol (standard in the
knowledge-tracing literature) is: for every user, earliest (1 - test_frac)
of their interactions are train, the rest are test. Predicting the future
from the past, per learner.
"""

import pandas as pd


def add_split_flag(df: pd.DataFrame, test_frac: float = 0.2) -> pd.DataFrame:
    df = df.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    rank = df.groupby("user_id").cumcount()
    size = df.groupby("user_id")["user_id"].transform("size")
    cut = ((1.0 - test_frac) * size).astype(int).clip(lower=1)
    df["is_test"] = rank >= cut
    return df


def per_user_temporal_split(
    df: pd.DataFrame, test_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    flagged = add_split_flag(df, test_frac)
    return flagged[~flagged["is_test"]].copy(), flagged[flagged["is_test"]].copy()
