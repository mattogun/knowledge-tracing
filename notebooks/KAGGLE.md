# Running training on Kaggle

1. kaggle.com -> Create -> Notebook.
2. Right panel FIRST (each of these restarts the session and wipes
   /kaggle/working, so do them before running anything):
   - Session options -> Accelerator = **GPU T4 x2** (NOT the P100: Kaggle's
     current PyTorch build has dropped Pascal sm_60 kernels, so the P100
     fails with "no kernel image is available")
   - Input -> + Add Input -> filter to **Competitions** -> add
     "Riiid Answer Correctness Prediction"
3. One cell, all absolute paths, safe to re-run:

       !rm -rf /kaggle/working/knowledge-tracing
       !git clone -q https://github.com/mattogun/knowledge-tracing.git /kaggle/working/knowledge-tracing
       !pip install -q -e /kaggle/working/knowledge-tracing
       !python /kaggle/working/knowledge-tracing/scripts/make_sample.py --csv /kaggle/input/competitions/riiid-test-answer-prediction/train.csv --out /kaggle/working/sample.parquet --mod 20
       !python /kaggle/working/knowledge-tracing/scripts/train_dkt.py --parquet /kaggle/working/sample.parquet --epochs 3

4. Copy the final `dkt: test_auc=` line into the README results table with
   the GPU type.
