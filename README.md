# Knowledge Tracing on Riiid (101M interactions)

![ci](https://github.com/mattogun/knowledge-tracing/actions/workflows/ci.yml/badge.svg)

Predicting whether a learner answers the next question correctly, from their
interaction history. Models: heuristic baseline -> logistic regression ->
DKT (LSTM) -> SAKT (transformer, in progress).

## Results

| Model | Test AUC | Notes |
|---|---|---|
| Heuristic (smoothed user + item rates) | 0.7372 | 1/20 user sample: 4.93M rows, 19,639 users, 992K test rows, CPU |
| Logistic regression (6 leak-free features) | 0.7384 | same split; item_rate and user_trailing_acc dominate the weights |
| DKT (LSTM, 3 epochs) | **0.7468** | Kaggle T4, same sample and split; AUC still rising at epoch 3 |

DKT learning curve (test AUC by epoch): 0.6967 -> 0.7390 -> 0.7468. The model
is below the counting baseline after one epoch and clear of every baseline by
three; the curve had not plateaued, so longer training is scheduled next.

Split: chronological per user (first 80% train, last 20% test). All question
statistics computed on train rows only. Numbers are reproducible via
`scripts/` on the Kaggle `riiid-test-answer-prediction` data.

## Next steps

Roughly in order:

1. **Train DKT longer.** The learning curve had not plateaued at epoch 3;
   run to ~10 epochs with early stopping on test AUC.
2. **SAKT.** Self-attentive knowledge tracing from the paper, same split and
   sample, so the comparison against DKT is apples to apples.
3. **Kaggle late submission.** Score the best model against the 3,400-team
   leaderboard to get an honest percentile, not just a local AUC.
4. **Optimizer comparison.** Re-run DKT/SAKT with a modern optimizer (Muon)
   against AdamW, same budget, and report the difference either way.
5. **Full-data run.** Everything so far uses a deterministic 1/20 user
   sample; the pipeline is built to scale to the full 101M rows once the
   architecture choices settle.
