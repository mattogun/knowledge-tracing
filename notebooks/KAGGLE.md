# Running training on Kaggle

1. kaggle.com -> Create -> Notebook. Settings: Accelerator = GPU (P100 or T4).
2. Add data: search "Riiid Answer Correctness Prediction" (competition data).
3. Cells:

       !git clone https://github.com/mattogun/knowledge-tracing.git
       %cd knowledge-tracing
       !pip install -e . -q
       !python scripts/make_sample.py --csv /kaggle/input/riiid-test-answer-prediction/train.csv --out sample.parquet --mod 20
       !python scripts/train_dkt.py --parquet sample.parquet --epochs 3

4. Copy the final `dkt: test_auc=` line into the README results table, and
   note the exact command + Kaggle GPU type alongside it.
