from __future__ import annotations

import pandas as pd
import pytest

from hf_finetuning_lab.data.hub import (
    HUB_PRESETS,
    HubDatasetConfig,
    list_hub_presets,
    normalize_hub_dataset,
    normalize_hub_dataset_dict,
)

datasets = pytest.importorskip("datasets")


def _fake_dataset_dict(label_names: list[str] | None = None) -> datasets.DatasetDict:
    """Construct a tiny in-memory DatasetDict mimicking a Hub text-classification dataset."""
    train_df = pd.DataFrame(
        {
            "text": ["alpha account", "beta billing", "gamma billing"],
            "label": [0, 1, 1],
        }
    )
    test_df = pd.DataFrame(
        {
            "text": ["delta account", "epsilon billing"],
            "label": [0, 1],
        }
    )
    if label_names is not None:
        features = datasets.Features(
            {
                "text": datasets.Value("string"),
                "label": datasets.ClassLabel(names=label_names),
            }
        )
        train = datasets.Dataset.from_pandas(train_df, features=features, preserve_index=False)
        test = datasets.Dataset.from_pandas(test_df, features=features, preserve_index=False)
    else:
        train = datasets.Dataset.from_pandas(train_df, preserve_index=False)
        test = datasets.Dataset.from_pandas(test_df, preserve_index=False)
    return datasets.DatasetDict({"train": train, "test": test})


def test_presets_listed_alphabetically() -> None:
    names = list_hub_presets()
    assert names == sorted(names)
    assert {"ag_news", "imdb", "banking77", "tweet_eval_sentiment"}.issubset(names)


def test_preset_split_mapping_includes_validation_when_set() -> None:
    cfg = HUB_PRESETS["tweet_eval_sentiment"]
    assert cfg.splits() == {"train": "train", "validation": "validation", "test": "test"}

    cfg_no_val = HUB_PRESETS["ag_news"]
    assert "validation" not in cfg_no_val.splits()


def test_normalize_uses_classlabel_names() -> None:
    ds = _fake_dataset_dict(label_names=["account", "billing"]).get("train")
    cfg = HubDatasetConfig(name="fake")
    frame = normalize_hub_dataset(ds, cfg)
    assert set(frame.columns) == {"text", "label", "label_id"}
    assert frame.loc[0, "label"] == "account"
    assert frame.loc[1, "label"] == "billing"
    assert frame["label_id"].tolist() == [0, 1, 1]


def test_normalize_uses_explicit_label_names_override() -> None:
    ds = _fake_dataset_dict().get("train")  # no ClassLabel feature
    cfg = HubDatasetConfig(name="fake", label_names=("acct", "bill"))
    frame = normalize_hub_dataset(ds, cfg)
    assert frame["label"].tolist() == ["acct", "bill", "bill"]


def test_normalize_falls_back_to_string_labels() -> None:
    ds = _fake_dataset_dict().get("train")
    cfg = HubDatasetConfig(name="fake")
    frame = normalize_hub_dataset(ds, cfg)
    assert frame["label"].tolist() == ["0", "1", "1"]
    assert frame["label_id"].tolist() == [0, 1, 1]


def test_normalize_caps_rows() -> None:
    ds = _fake_dataset_dict(label_names=["a", "b"]).get("train")
    cfg = HubDatasetConfig(name="fake")
    frame = normalize_hub_dataset(ds, cfg, max_rows=2)
    assert len(frame) == 2


def test_normalize_rejects_missing_text_field() -> None:
    ds = _fake_dataset_dict(label_names=["a", "b"]).get("train")
    cfg = HubDatasetConfig(name="fake", text_field="not_a_column")
    with pytest.raises(ValueError, match="text field"):
        normalize_hub_dataset(ds, cfg)


def test_normalize_dict_returns_canonical_split_names() -> None:
    ds_dict = _fake_dataset_dict(label_names=["a", "b"])
    cfg = HubDatasetConfig(name="fake")
    frames = normalize_hub_dataset_dict(ds_dict, cfg)
    assert set(frames) == {"train", "test"}
    assert len(frames["train"]) == 3
    assert len(frames["test"]) == 2


def test_normalize_dict_respects_splits_filter() -> None:
    ds_dict = _fake_dataset_dict(label_names=["a", "b"])
    cfg = HubDatasetConfig(name="fake")
    frames = normalize_hub_dataset_dict(ds_dict, cfg, splits=["test"])
    assert set(frames) == {"test"}


def test_normalize_dict_errors_when_no_splits_match() -> None:
    ds_dict = _fake_dataset_dict(label_names=["a", "b"])
    cfg = HubDatasetConfig(name="fake", train_split="missing", test_split="also_missing")
    with pytest.raises(ValueError, match="None of the requested splits"):
        normalize_hub_dataset_dict(ds_dict, cfg)
