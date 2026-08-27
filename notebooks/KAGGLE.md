# Running training on Kaggle

1. kaggle.com -> Create -> Notebook. Right panel: Session options ->
   Accelerator = GPU (P100 or T4).
2. Right panel: Input -> **+ Add Input** -> search "Riiid Answer Correctness
   Prediction" -> add the **competition** (trophy icon), not a user dataset.
   Without this step nothing else works.
3. First cell, and do not proceed until it lists train.csv:

       !ls /kaggle/input/riiid-test-answer-prediction

4. Then, each in its own cell (absolute path in the cd, so re-running a cell
   never nests clones):

       !git clone https://github.com/mattogun/knowledge-tracing.git
       %cd /kaggle/working/knowledge-tracing
       !pip install -e . -q
       !python scripts/make_sample.py --csv /kaggle/input/riiid-test-answer-prediction/train.csv --out sample.parquet --mod 20
       !python scripts/train_dkt.py --parquet sample.parquet --epochs 3

5. Copy the final `dkt: test_auc=` line into the README results table, and
   note the exact command plus Kaggle GPU type alongside it.
