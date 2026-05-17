"""Load and normalize Hugging Face Hub text-classification datasets.

The loader maps each preset's native field names onto our canonical ``text`` /
``label`` schema, resolves integer label IDs to human-readable names, and lets
callers cap rows per split so smoke runs stay quick. ``load_hub_dataset`` is
the only function that calls into ``datasets.load_dataset``; the rest of the
module works on already-loaded ``Dataset`` / ``DatasetDict`` objects so tests
can exercise the normalization path without hitting the network.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover - typing only
    from datasets import Dataset, DatasetDict


@dataclass(slots=True, frozen=True)
class HubDatasetConfig:
    """Preset describing how to normalize one Hub text-classification dataset."""

    name: str
    text_field: str = "text"
    label_field: str = "label"
    config: str | None = None
    train_split: str = "train"
    test_split: str = "test"
    validation_split: str | None = None
    label_names: tuple[str, ...] | None = None
    description: str = ""

    def splits(self) -> dict[str, str]:
        """Return the canonical-name -> source-split mapping for this preset."""
        out: dict[str, str] = {"train": self.train_split, "test": self.test_split}
        if self.validation_split is not None:
            out["validation"] = self.validation_split
        return out


HUB_PRESETS: dict[str, HubDatasetConfig] = {
    "ag_news": HubDatasetConfig(
        name="ag_news",
        label_names=("World", "Sports", "Business", "Sci/Tech"),
        description="AG News topic classification (4 classes).",
    ),
    "imdb": HubDatasetConfig(
        name="imdb",
        label_names=("negative", "positive"),
        description="IMDb sentiment classification (2 classes).",
    ),
    "banking77": HubDatasetConfig(
        name="banking77",
        description="Banking customer-intent classification (77 fine-grained classes).",
    ),
    "tweet_eval_sentiment": HubDatasetConfig(
        name="tweet_eval",
        config="sentiment",
        train_split="train",
        test_split="test",
        validation_split="validation",
        label_names=("negative", "neutral", "positive"),
        description="TweetEval sentiment subtask (3 classes).",
    ),
}


def list_hub_presets() -> list[str]:
    """Return the names of all built-in Hub presets, sorted."""
    return sorted(HUB_PRESETS)


def _resolve_label_names(dataset: Dataset, cfg: HubDatasetConfig) -> list[str] | None:
    """Pick the most informative label-name list available for this dataset."""
    if cfg.label_names is not None:
        return list(cfg.label_names)
    feature = dataset.features.get(cfg.label_field) if hasattr(dataset, "features") else None
    if feature is not None and hasattr(feature, "names"):
        return list(feature.names)
    return None


def normalize_hub_dataset(
    dataset: Dataset,
    cfg: HubDatasetConfig,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Normalize a single ``Dataset`` split into a ``text`` / ``label`` DataFrame.

    The returned frame has ``text`` (str), ``label`` (human-readable label name
    when available, else the raw value as a string) and ``label_id`` (int)
    columns. ``max_rows`` caps the output so smoke runs stay quick.
    """
    if cfg.text_field not in dataset.column_names:
        raise ValueError(
            f"Dataset '{cfg.name}' is missing expected text field '{cfg.text_field}'. "
            f"Available columns: {list(dataset.column_names)}"
        )
    if cfg.label_field not in dataset.column_names:
        raise ValueError(
            f"Dataset '{cfg.name}' is missing expected label field '{cfg.label_field}'. "
            f"Available columns: {list(dataset.column_names)}"
        )

    label_names = _resolve_label_names(dataset, cfg)
    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows must be positive.")
        dataset = dataset.select(range(min(max_rows, len(dataset))))

    df = pd.DataFrame(
        {
            "text": [str(value) for value in dataset[cfg.text_field]],
            "label_id": list(dataset[cfg.label_field]),
        }
    )

    if label_names is not None:
        def _name_for(label_id: Any) -> str:
            try:
                idx = int(label_id)
            except (TypeError, ValueError):
                return str(label_id)
            if 0 <= idx < len(label_names):
                return label_names[idx]
            return str(label_id)

        df["label"] = df["label_id"].map(_name_for)
        df["label_id"] = df["label_id"].astype(int)
    else:
        df["label"] = df["label_id"].astype(str)
        # Keep numeric label_id if it was numeric; else encode by sorted unique.
        unique = sorted(df["label"].unique())
        mapping = {value: idx for idx, value in enumerate(unique)}
        df["label_id"] = df["label"].map(mapping).astype(int)

    return df[["text", "label", "label_id"]]


def normalize_hub_dataset_dict(
    dataset_dict: DatasetDict | Mapping[str, Dataset],
    cfg: HubDatasetConfig,
    max_rows_per_split: int | None = None,
    splits: Iterable[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Normalize each requested split into a DataFrame keyed by canonical split name."""
    split_map = cfg.splits()
    if splits is not None:
        split_map = {k: v for k, v in split_map.items() if k in set(splits)}
    output: dict[str, pd.DataFrame] = {}
    for canonical, source in split_map.items():
        if source not in dataset_dict:
            continue
        output[canonical] = normalize_hub_dataset(
            dataset_dict[source], cfg, max_rows=max_rows_per_split
        )
    if not output:
        raise ValueError(
            f"None of the requested splits ({list(split_map.values())}) were found in the "
            f"dataset. Available splits: {sorted(dataset_dict)}"
        )
    return output


def load_hub_dataset(
    name_or_config: str | HubDatasetConfig,
    max_rows_per_split: int | None = None,
    splits: Iterable[str] | None = None,
    **load_dataset_kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Download a Hub dataset and return canonical-split-name -> DataFrame.

    ``name_or_config`` is either a preset name (see :func:`list_hub_presets`)
    or a fully-specified :class:`HubDatasetConfig`. Additional keyword
    arguments are forwarded to ``datasets.load_dataset``.

    This is the only function in the module that requires network access.
    """
    if isinstance(name_or_config, str):
        if name_or_config not in HUB_PRESETS:
            raise KeyError(
                f"Unknown Hub preset '{name_or_config}'. Known presets: {list_hub_presets()}"
            )
        cfg = HUB_PRESETS[name_or_config]
    else:
        cfg = name_or_config

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - depends on optional runtime
        raise ImportError(
            "Install the `datasets` package to load Hugging Face Hub datasets."
        ) from exc

    if cfg.config is not None:
        dataset_dict = load_dataset(cfg.name, cfg.config, **load_dataset_kwargs)
    else:
        dataset_dict = load_dataset(cfg.name, **load_dataset_kwargs)
    return normalize_hub_dataset_dict(dataset_dict, cfg, max_rows_per_split=max_rows_per_split, splits=splits)


def write_hub_dataset_csv(
    name_or_config: str | HubDatasetConfig,
    output_dir: Any,
    max_rows_per_split: int | None = None,
) -> dict[str, Any]:
    """Download a Hub dataset and write one CSV per split into ``output_dir``."""
    from pathlib import Path

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    frames = load_hub_dataset(name_or_config, max_rows_per_split=max_rows_per_split)
    paths: dict[str, Path] = {}
    name = name_or_config if isinstance(name_or_config, str) else name_or_config.name
    for split, frame in frames.items():
        path = destination / f"{name}_{split}.csv"
        frame.to_csv(path, index=False)
        paths[split] = path
    return paths
