import pandas as pd
import pytest

from kt.features import add_trailing_user_stats


def test_trailing_accuracy_excludes_current_row():
    # One user answering [right, wrong, right]. With alpha=2, prior=0.5:
    # row0: no history -> (0 + 1) / (0 + 2) = 0.5
    # row1: history [1] -> (1 + 1) / (1 + 2) = 0.6667
    # row2: history [1,0] -> (1 + 1) / (2 + 2) = 0.5
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1],
            "timestamp": [10, 20, 30],
            "answered_correctly": [1, 0, 1],
        }
    )
    out = add_trailing_user_stats(df, alpha=2.0, prior=0.5)
    assert out["user_trailing_acc"].tolist() == pytest.approx([0.5, 0.6667, 0.5], abs=1e-3)
    assert out["user_attempts"].tolist() == [0, 1, 2]
