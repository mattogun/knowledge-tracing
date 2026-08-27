import pandas as pd

from kt.splits import add_split_flag, per_user_temporal_split


def _df():
    # user 1: 5 rows, user 2: 2 rows, user 3: 1 row
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 1, 1, 2, 2, 3],
            "timestamp": [10, 20, 30, 40, 50, 5, 15, 7],
            "answered_correctly": [1, 0, 1, 1, 0, 1, 0, 1],
        }
    )


def test_no_user_time_leakage():
    train, test = per_user_temporal_split(_df(), test_frac=0.2)
    for uid in test["user_id"].unique():
        tr_max = train.loc[train["user_id"] == uid, "timestamp"].max()
        te_min = test.loc[test["user_id"] == uid, "timestamp"].min()
        assert tr_max < te_min  # every train row strictly before every test row


def test_single_row_users_stay_in_train():
    train, test = per_user_temporal_split(_df(), test_frac=0.2)
    assert 3 in train["user_id"].values
    assert 3 not in test["user_id"].values


def test_split_flag_matches_split():
    df = add_split_flag(_df(), test_frac=0.2)
    train, test = per_user_temporal_split(_df(), test_frac=0.2)
    assert df["is_test"].sum() == len(test)
    assert len(df) == len(train) + len(test)
