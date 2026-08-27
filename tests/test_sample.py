import pandas as pd

from kt.sample import filter_chunk, keep_user


def _chunk():
    return pd.DataFrame(
        {
            "timestamp": [0, 100, 0, 50, 200],
            "user_id": [1, 1, 2, 2, 1],
            "content_id": [10, 11, 10, 12, 13],
            "content_type_id": [0, 0, 0, 1, 0],
            "answered_correctly": [1, 0, 1, -1, 1],
            "prior_question_elapsed_time": [None, 300.0, None, None, 200.0],
            "prior_question_had_explanation": [None, True, None, None, False],
        }
    )


def test_filter_chunk_keeps_only_answered_questions():
    out = filter_chunk(_chunk())
    assert (out["content_type_id"] == 0).all()
    assert set(out["answered_correctly"].unique()) <= {0, 1}
    assert len(out) == 4  # the lecture row (content_type_id=1) dropped


def test_keep_user_is_deterministic_partition():
    kept = [u for u in range(1000) if keep_user(u, mod=20)]
    assert len(kept) > 0
    assert kept == [u for u in range(1000) if keep_user(u, mod=20)]  # stable
    assert all(u % 20 == 0 for u in kept)
