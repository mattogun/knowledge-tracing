"""Sliding-window sequence dataset for DKT/SAKT.

Encoding: item ids are content_id + 1 and responses are correctness + 1,
because 0 is reserved for padding in both embeddings.

Each window of a user's history yields: inputs at positions 0..n-1, the
queried next question at 1..n, its label, a loss mask over real positions,
and a test mask marking positions whose label belongs to the test split.
Training uses loss_mask * (1 - test_mask); evaluation uses test_mask only,
so test answers never contribute to a gradient.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class KTSequences(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int = 200):
        df = df.sort_values(["user_id", "timestamp"], kind="stable")
        self.max_len = max_len
        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for _, g in df.groupby("user_id", sort=False):
            items = g["content_id"].to_numpy(np.int64) + 1
            resps = g["answered_correctly"].to_numpy(np.int64) + 1
            is_test = g["is_test"].to_numpy(bool)
            for start in range(0, max(len(items) - 1, 0), max_len):
                window = slice(start, start + max_len + 1)
                if len(items[window]) < 2:
                    continue
                self.samples.append((items[window], resps[window], is_test[window]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        items_w, resps_w, test_w = self.samples[i]
        n = len(items_w) - 1
        length = self.max_len

        items = np.zeros(length, np.int64)
        resps = np.zeros(length, np.int64)
        nxt = np.zeros(length, np.int64)
        labels = np.zeros(length, np.float32)
        loss_mask = np.zeros(length, np.float32)
        test_mask = np.zeros(length, np.float32)

        items[:n] = items_w[:-1]
        resps[:n] = resps_w[:-1]
        nxt[:n] = items_w[1:]
        labels[:n] = (resps_w[1:] == 2).astype(np.float32)
        loss_mask[:n] = 1.0
        test_mask[:n] = test_w[1:].astype(np.float32)

        return tuple(
            torch.from_numpy(a) for a in (items, resps, nxt, labels, loss_mask, test_mask)
        )
