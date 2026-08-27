# Knowledge Tracing on Riiid (101M interactions)

Predicting whether a learner answers the next question correctly, from their
interaction history. Models: heuristic baseline -> logistic regression ->
DKT (LSTM) -> SAKT (transformer, in progress).

## Results

| Model | Test AUC | Notes |
|---|---|---|
| (results land here as runs complete) | | |

Split: chronological per user (first 80% train, last 20% test). All question
statistics computed on train rows only. Numbers are reproducible via
`scripts/` on the Kaggle `riiid-test-answer-prediction` data.
