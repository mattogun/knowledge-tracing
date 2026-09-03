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

## Where the data comes from

One source: the **Riiid Answer Correctness Prediction** competition dataset
on Kaggle (Riiid Labs, 2020). It is the interaction log of Santa TOEIC, a
Korean English-test tutoring app: every question a learner answered, in
order, with whether they got it right. The competition is closed, but the
data is still downloadable from Kaggle with a free account, and Kaggle
notebooks can mount it directly (see `notebooks/KAGGLE.md`). It is not
redistributed here; `data/` is git-ignored.

### The raw files

| File | Size | What it is |
|---|---|---|
| `train.csv` | 5.8 GB, 101,230,332 rows | one row per interaction: user, content, answer, timing |
| `questions.csv` | 13,523 rows | question metadata: bundle, correct answer, TOEIC part (1 to 7), skill tags |
| `lectures.csv` | 418 rows | lecture metadata (watched, not answered) |

The first two rows of `train.csv`, verbatim:

```
row_id,timestamp,user_id,content_id,content_type_id,task_container_id,user_answer,answered_correctly,prior_question_elapsed_time,prior_question_had_explanation
0,0,115,5692,0,1,3,1,,
1,56943,115,5716,0,2,2,1,37000.0,False
```

`timestamp` is milliseconds since **that user's** first event, not wall-clock
time; the longest history in the sample spans 973 days. This is why a single
global time cutoff would be meaningless and the split below is per user.

### What the pipeline keeps

`src/kt/sample.py` streams `train.csv` in 2M-row chunks (it does not fit in
memory on a laptop) and keeps seven columns. Lecture rows
(`content_type_id = 1`, which carry `answered_correctly = -1`) are dropped:
they are events, not questions, and there is no label to predict.

Then a **deterministic 1/20 user sample**: a user is kept iff
`user_id % 20 == 0`. Sampling is by user, never by row, because sequence
models need whole histories, and the modulus rule means the same 19,639
users come out on every machine with no seed to lose. Everything in the
results table was computed on this slice. It looks like this:

| | value |
|---|---|
| rows (question answers only) | 4,925,665 |
| users | 19,639 |
| overall accuracy (the base rate any model has to beat) | 0.655 |
| interactions per user: median / mean / max | 40 / 251 / 14,068 |
| users with fewer than 10 interactions | 133 |
| users with 1,000 or more | 1,154 |

A real user's first six rows after filtering (user 24600, the lowest id in
the slice):

| timestamp | content_id | answered_correctly | prior_question_elapsed_time | prior_question_had_explanation |
|---|---|---|---|---|
| 0 | 7900 | 1 | | |
| 25379 | 7876 | 0 | 24000.0 | False |
| 50137 | 175 | 1 | 23000.0 | False |
| 70181 | 1278 | 1 | 22000.0 | False |
| 148601 | 2064 | 0 | 18000.0 | False |
| 148601 | 2065 | 1 | 18000.0 | False |

The last two rows share a timestamp: they are one bundle, served together,
which is a leakage trap for any feature that peeks within a timestamp.

### The split, and where the 992K test rows come from

`src/kt/splits.py`: within each user, sorted by timestamp, the earliest 80%
of interactions are train and the last 20% are test. User 24600 has 50
rows, so 40 train and 10 test. Across the slice that gives 3,933,851 train
rows and 991,814 test rows, the "992K held-out answers" every AUC in the
results table is scored on. Question statistics (item rates, the smoothing
prior) are computed on train rows only; the trailing user accuracy at row
t uses answers strictly before t. Fifteen tests under `tests/` pin these
rules so a refactor cannot quietly reintroduce future information.

### How it reaches the models

`src/kt/dataset.py` turns each user's history into sliding windows of 200
steps. Item ids and responses are shifted by one so that 0 is padding;
each position predicts the next question's correctness, and a test mask
marks positions whose label is in the test split. Training multiplies the
loss by `loss_mask * (1 - test_mask)`, so test answers never contribute a
gradient, and evaluation reads `test_mask` only.

Rebuild the slice yourself:

```
python scripts/make_sample.py --csv data/raw/train.csv --out data/sample.parquet --mod 20
```

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
