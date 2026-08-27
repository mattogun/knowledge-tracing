"""DKT-style LSTM: read (question, response) history, query the next question.

At step t the LSTM has consumed interactions 0..t; its hidden state is
concatenated with the embedding of question t+1 to score P(correct at t+1).
"""

import torch
from torch import nn


class DKT(nn.Module):
    def __init__(self, n_items: int, d: int = 64, hidden: int = 128):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, d, padding_idx=0)
        self.resp_emb = nn.Embedding(3, d, padding_idx=0)  # 0 pad, 1 wrong, 2 right
        self.lstm = nn.LSTM(d, hidden, batch_first=True)
        self.head = nn.Linear(hidden + d, 1)

    def forward(
        self, items: torch.Tensor, resps: torch.Tensor, next_items: torch.Tensor
    ) -> torch.Tensor:
        x = self.item_emb(items) + self.resp_emb(resps)
        h, _ = self.lstm(x)
        q = self.item_emb(next_items)
        return self.head(torch.cat([h, q], dim=-1)).squeeze(-1)
