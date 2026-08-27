import pandas as pd

from kt.dataset import KTSequences


def _df():
    return pd.DataFrame(
        {
            "user_id": [1] * 5 + [2] * 2,
            "timestamp": [10, 20, 30, 40, 50, 5, 15],
            "content_id": [3, 4, 5, 6, 7, 3, 4],
            "answered_correctly": [1, 0, 1, 1, 0, 1, 1],
            "is_test": [False, False, False, False, True, False, True],
        }
    )


def test_shapes_and_padding():
    ds = KTSequences(_df(), max_len=8)
    items, _resps, _nxt, _labels, loss_mask, _test_mask = ds[0]
    assert len(items) == 8
    assert loss_mask.sum() > 0
    assert (items[int(loss_mask.sum().item()):] == 0).all()  # padded tail is 0


def test_labels_align_with_next_item():
    # user 1 history: q3(right) q4(wrong) q5(right) q6(right) q7(wrong)
    # first sample predicts positions 1..4 from history 0..3
    ds = KTSequences(_df(), max_len=8)
    items, resps, nxt, labels, _loss_mask, test_mask = ds[0]
    assert items[0].item() == 4 and resps[0].item() == 2  # q3+1, right
    assert nxt[0].item() == 5 and labels[0].item() == 0.0  # predicts q4: wrong
    # position of q7 (a test row) is flagged in test_mask
    assert test_mask.sum().item() == 1


def test_single_interaction_user_is_skipped():
    df = pd.DataFrame(
        {
            "user_id": [9],
            "timestamp": [1],
            "content_id": [2],
            "answered_correctly": [1],
            "is_test": [False],
        }
    )
    assert len(KTSequences(df, max_len=8)) == 0  # nothing to predict
