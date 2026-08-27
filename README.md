# Knowledge Tracing on Riiid (101M interactions)

Predicting whether a learner answers the next question correctly, from their
interaction history. Models: heuristic baseline -> logistic regression ->
DKT (LSTM) -> SAKT (transformer, in progress).

## Results

| Model | Test AUC | Notes |
|---|---|---|
| Heuristic (smoothed user + item rates) | 0.7372 | 1/20 user sample: 4.93M rows, 19,639 users, 992K test rows, CPU |
| Logistic regression (6 leak-free features) | 0.7384 | same split; item_rate and user_trailing_acc dominate the weights |

Split: chronological per user (first 80% train, last 20% test). All question
statistics computed on train rows only. Numbers are reproducible via
`scripts/` on the Kaggle `riiid-test-answer-prediction` data.
