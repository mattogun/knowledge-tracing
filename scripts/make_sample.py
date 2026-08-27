"""Build data/sample.parquet: all rows for the deterministic 1/20 user slice."""

import argparse

import pandas as pd

from kt.sample import keep_user, load_riiid_chunks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/raw/train.csv")
    ap.add_argument("--out", default="data/sample.parquet")
    ap.add_argument("--mod", type=int, default=20)
    args = ap.parse_args()

    parts = []
    for chunk in load_riiid_chunks(args.csv):
        parts.append(chunk[chunk["user_id"].map(lambda u: keep_user(u, args.mod))])
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values(["user_id", "timestamp"], kind="stable").reset_index(drop=True)
    df.to_parquet(args.out)
    print(f"rows={len(df)} users={df['user_id'].nunique()} -> {args.out}")


if __name__ == "__main__":
    main()
