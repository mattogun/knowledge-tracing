import pandas as pd
import pytest

from kt.baselines import heuristic_predict, smoothed_rates


def test_smoothed_rates_hand_example():
    train = pd.DataFrame({"user_id": [1, 1, 1], "answered_correctly": [1, 1, 0]})
    rates = smoothed_rates(train, "user_id", alpha=2.0, prior=0.5)
    # (sum + alpha*prior) / (count + alpha) = (2 + 1) / (3 + 2)
    assert rates[1] == pytest.approx(0.6)


def test_unseen_user_falls_back_to_train_prior():
    train = pd.DataFrame(
        {"user_id": [1, 1], "content_id": [7, 7], "answered_correctly": [1, 0]}
    )
    test = pd.DataFrame({"user_id": [99], "content_id": [8]})
    preds = heuristic_predict(train, test)
    assert preds.iloc[0] == pytest.approx(0.5)  # train prior is 0.5


def test_question_stats_come_from_train_only():
    train = pd.DataFrame(
        {"user_id": [1, 2], "content_id": [7, 7], "answered_correctly": [1, 1]}
    )
    # question 7 is always-correct in train; test outcomes must not affect it
    test = pd.DataFrame({"user_id": [3, 4], "content_id": [7, 7]})
    preds = heuristic_predict(train, test, alpha=0.0)
    assert (preds > 0.9).all()
