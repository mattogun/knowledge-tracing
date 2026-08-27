"""Train DKT. Locally: --mod 200 for a tiny smoke slice. Kaggle: default sample."""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from kt.dataset import KTSequences
from kt.models.dkt import DKT
from kt.splits import add_split_flag

N_ITEMS = 13523  # question content_ids are 0..13522


def evaluate(model, loader, device) -> float:
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for items, resps, nxt, lab, _loss_mask, test_mask in loader:
            logits = model(items.to(device), resps.to(device), nxt.to(device))
            keep = test_mask.bool()
            probs.append(torch.sigmoid(logits.cpu())[keep].numpy())
            labels.append(lab[keep].numpy())
    return roc_auc_score(np.concatenate(labels), np.concatenate(probs))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/sample.parquet")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--mod", type=int, default=1, help="extra user subsampling for smoke runs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(args.parquet)
    if args.mod > 1:
        df = df[df["user_id"] % args.mod == 0]
    df = add_split_flag(df, test_frac=0.2)

    ds = KTSequences(df)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=2)
    model = DKT(n_items=N_ITEMS).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.BCEWithLogitsLoss(reduction="none")

    for epoch in range(args.epochs):
        model.train()
        for items, resps, nxt, lab, loss_mask, test_mask in loader:
            opt.zero_grad()
            logits = model(items.to(device), resps.to(device), nxt.to(device))
            train_mask = (loss_mask * (1 - test_mask)).to(device)
            raw = lossf(logits, lab.to(device))
            loss = (raw * train_mask).sum() / train_mask.sum().clamp(min=1)
            loss.backward()
            opt.step()
        auc = evaluate(model, loader, device)
        print(f"epoch={epoch} dkt: test_auc={auc:.4f}")

    torch.save(model.state_dict(), "dkt.pt")


if __name__ == "__main__":
    main()
