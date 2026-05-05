from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV or JSONL dataset into a DataFrame."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {data_path}")

    suffix = data_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(data_path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(data_path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported file extension '{suffix}'. Use CSV or JSONL.")


def validate_text_classification_frame(
    df: pd.DataFrame,
    text_col: str,
    label_col: str,
) -> None:
    """Validate that a DataFrame can be used for text classification."""
    missing = [col for col in [text_col, label_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if df[text_col].isna().any():
        raise ValueError(f"Column '{text_col}' contains missing values.")
    if df[label_col].isna().any():
        raise ValueError(f"Column '{label_col}' contains missing values.")
    if df[label_col].nunique() < 2:
        raise ValueError("Text classification requires at least two labels.")


def build_label_mapping(labels: pd.Series) -> tuple[dict[str, int], dict[int, str]]:
    """Create deterministic label-to-ID and ID-to-label mappings."""
    unique_labels = sorted(str(label) for label in labels.unique())
    label2id = {label: idx for idx, label in enumerate(unique_labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def encode_labels(df: pd.DataFrame, label_col: str, label2id: dict[str, int]) -> pd.DataFrame:
    """Add an integer `label_id` column from string labels."""
    output = df.copy()
    output["label_id"] = output[label_col].map(lambda value: label2id[str(value)])
    return output


def to_hf_dataset(df: pd.DataFrame):
    """Convert a DataFrame into a Hugging Face Dataset.

    The import is local so lightweight tests do not require importing the datasets package.
    """
    try:
        from datasets import Dataset
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError("Install the `datasets` package to use Hugging Face datasets.") from exc
    return Dataset.from_pandas(df, preserve_index=False)
