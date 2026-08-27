"""First honest number: heuristic AUC on the sample."""

import pandas as pd
from sklearn.metrics import roc_auc_score

from kt.baselines import heuristic_predict
from kt.splits import per_user_temporal_split


def main() -> None:
    df = pd.read_parquet("data/sample.parquet")
    train, test = per_user_temporal_split(df, test_frac=0.2)
    preds = heuristic_predict(train, test)
    auc = roc_auc_score(test["answered_correctly"], preds)
    print(f"heuristic: test_auc={auc:.4f} train_rows={len(train)} test_rows={len(test)}")


if __name__ == "__main__":
    main()
