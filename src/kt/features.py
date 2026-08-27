"""Hand-engineered features for the logistic-regression baseline.

The leakage discipline: user_trailing_acc at row t uses answers strictly
before t (cumsum minus current), and item_rate comes from train rows only.
"""

import numpy as np
import pandas as pd

from kt.baselines import smoothed_rates


def add_trailing_user_stats(
    df: pd.DataFrame, alpha: float = 5.0, prior: float = 0.65
) -> pd.DataFrame:
    df = df.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    grp = df.groupby("user_id")["answered_correctly"]
    prior_correct = grp.cumsum() - df["answered_correctly"]  # exclude current row
    attempts = grp.cumcount()
    df["user_trailing_acc"] = (
        (prior_correct + alpha * prior) / (attempts + alpha)
    ).astype("float32")
    df["user_attempts"] = attempts.astype("int32")
    return df


def build_feature_frame(
    df: pd.DataFrame, train: pd.DataFrame, questions: pd.DataFrame
) -> pd.DataFrame:
    prior = float(train["answered_correctly"].mean())
    item_rate = smoothed_rates(train, "content_id", alpha=5.0, prior=prior)
    part = questions.set_index("question_id")["part"]

    out = pd.DataFrame(index=df.index)
    out["user_trailing_acc"] = df["user_trailing_acc"]
    out["log_attempts"] = np.log1p(df["user_attempts"])
    out["item_rate"] = df["content_id"].map(item_rate).fillna(prior)
    out["prior_elapsed_log"] = np.log1p(
        df["prior_question_elapsed_time"].fillna(0.0).clip(lower=0)
    )
    out["had_explanation"] = (
        df["prior_question_had_explanation"].fillna(False).astype("float32")
    )
    out["part"] = df["content_id"].map(part).fillna(0).astype("float32")
    return out
