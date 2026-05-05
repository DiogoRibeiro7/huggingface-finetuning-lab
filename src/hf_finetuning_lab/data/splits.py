from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_train_valid_test_split(
    df: pd.DataFrame,
    label_col: str,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a classification dataset into train, validation, and test sets."""
    if label_col not in df.columns:
        raise ValueError(f"label_col not found: {label_col}")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 <= validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1.")

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
