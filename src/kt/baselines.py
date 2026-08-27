"""Counting baseline: no learning, just smoothed historical rates.

If a neural network cannot beat this, the network is noise. alpha is
Laplace-style smoothing pulling rare users/questions toward the prior.
"""

import pandas as pd


def smoothed_rates(
    train: pd.DataFrame, key: str, alpha: float, prior: float
) -> pd.Series:
    g = train.groupby(key)["answered_correctly"].agg(["sum", "count"])
    return (g["sum"] + alpha * prior) / (g["count"] + alpha)


def heuristic_predict(
    train: pd.DataFrame, test: pd.DataFrame, alpha: float = 5.0
) -> pd.Series:
    prior = float(train["answered_correctly"].mean())
    user_rate = smoothed_rates(train, "user_id", alpha, prior)
    item_rate = smoothed_rates(train, "content_id", alpha, prior)
    pu = test["user_id"].map(user_rate).fillna(prior)
    pq = test["content_id"].map(item_rate).fillna(prior)
    return (pu + pq) / 2.0
