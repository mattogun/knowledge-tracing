"""Chunked loading of Riiid train.csv and deterministic user-level sampling.

Sampling is by user, never by row: sequence models need whole histories.
keep_user(u, mod) keeps users where u % mod == 0 -- a stable ~1/mod slice
that is identical on every machine, so local numbers reproduce anywhere.
"""

from collections.abc import Iterator

import pandas as pd

USECOLS = [
    "timestamp",
    "user_id",
    "content_id",
    "content_type_id",
    "answered_correctly",
    "prior_question_elapsed_time",
    "prior_question_had_explanation",
]

DTYPES = {
    "timestamp": "int64",
    "user_id": "int64",
    "content_id": "int16",
    "content_type_id": "int8",
    "answered_correctly": "int8",
    "prior_question_elapsed_time": "float32",
}


def keep_user(user_id: int, mod: int) -> bool:
    return user_id % mod == 0


def filter_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    out = chunk[(chunk["content_type_id"] == 0) & (chunk["answered_correctly"] >= 0)]
    return out.copy()


def load_riiid_chunks(csv_path: str, chunksize: int = 2_000_000) -> Iterator[pd.DataFrame]:
    for chunk in pd.read_csv(
        csv_path, usecols=USECOLS, dtype=DTYPES, chunksize=chunksize
    ):
        yield filter_chunk(chunk)
