from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_train_valid_test_split(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    seed: int = 42,
    group_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a classification dataset into train, validation, and test sets.

    ``group_col`` keeps every row sharing a group value in the same split.
    Use it whenever rows are not independent — repeated customers, threads,
    documents, or near-duplicate phrasings. Splitting those at random puts
    near-copies of training rows into test and reports a score the model has
    not earned. Grouped splits cannot be stratified, so class balance is
    approximate.
    """
    if label_col not in df.columns:
        raise ValueError(f"label_col not found: {label_col}")
    if group_col is not None and group_col not in df.columns:
        raise ValueError(f"group_col not found: {group_col}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 <= validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

    if group_col is not None:
        return _grouped_split(df, group_col, test_size, validation_size, seed)

    train_valid, test = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=seed,
    )
    if validation_size == 0:
        return train_valid.reset_index(drop=True), pd.DataFrame(), test.reset_index(drop=True)

    # validation_size is interpreted as a fraction of the original dataset.
    relative_valid = validation_size / (1.0 - test_size)
    train, valid = train_test_split(
        train_valid,
        test_size=relative_valid,
        stratify=train_valid[label_col],
        random_state=seed,
    )
    return train.reset_index(drop=True), valid.reset_index(drop=True), test.reset_index(drop=True)


def _grouped_split(
    df: pd.DataFrame,
    group_col: str,
    test_size: float,
    validation_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by whole groups so no group spans two splits."""
    groups = df[group_col].astype(str).unique()
    if len(groups) < 2:
        raise ValueError(
            f"Need at least 2 distinct '{group_col}' values to split by group; got {len(groups)}."
        )

    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(groups))
    n_test = max(1, round(len(shuffled) * test_size))
    n_valid = round(len(shuffled) * validation_size) if validation_size else 0
    if n_test + n_valid >= len(shuffled):
        raise ValueError(
            f"Splitting {len(shuffled)} '{group_col}' values by "
            f"test_size={test_size} and validation_size={validation_size} leaves no groups "
            "for training."
        )

    test_groups = set(shuffled[:n_test])
    valid_groups = set(shuffled[n_test : n_test + n_valid])

    keys = df[group_col].astype(str)
    test = df[keys.isin(test_groups)]
    valid = df[keys.isin(valid_groups)]
    train = df[~keys.isin(test_groups | valid_groups)]
    return (
        train.reset_index(drop=True),
        valid.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def duplicate_text_report(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Return the repeated text values and how often each occurs.

    Duplicates split at random land in both train and test, which inflates the
    reported score. Run this before splitting; feed the result to ``group_col``
    or drop the repeats.
    """
    if text_col not in df.columns:
        raise ValueError(f"text_col not found: {text_col}")
    counts = df[text_col].value_counts()
    repeated = counts[counts > 1]
    return pd.DataFrame({text_col: repeated.index, "count": repeated.to_numpy()}).reset_index(
        drop=True
    )
