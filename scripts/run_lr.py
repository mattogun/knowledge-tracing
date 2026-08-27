"""Second number: logistic regression on engineered features."""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from kt.features import add_trailing_user_stats, build_feature_frame
from kt.splits import add_split_flag


def main() -> None:
    df = pd.read_parquet("data/sample.parquet")
    questions = pd.read_csv("data/raw/questions.csv")
    df = add_split_flag(df, test_frac=0.2)
    df = add_trailing_user_stats(df)
    train, test = df[~df["is_test"]], df[df["is_test"]]

    x_train = build_feature_frame(train, train, questions)
    x_test = build_feature_frame(test, train, questions)  # train-only stats
    scaler = StandardScaler().fit(x_train)
    model = LogisticRegression(max_iter=1000)
    model.fit(scaler.transform(x_train), train["answered_correctly"])
    preds = model.predict_proba(scaler.transform(x_test))[:, 1]
    auc = roc_auc_score(test["answered_correctly"], preds)
    print(f"logreg: test_auc={auc:.4f}")
    print(dict(zip(x_train.columns, model.coef_[0].round(3))))


if __name__ == "__main__":
    main()
