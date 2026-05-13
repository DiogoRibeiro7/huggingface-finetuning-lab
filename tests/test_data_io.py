import pandas as pd
import pytest

from hf_finetuning_lab.data.io import (
    build_label_mapping,
    encode_labels,
    validate_text_classification_frame,
)


def test_validate_text_classification_frame_accepts_valid_data() -> None:
    df = pd.DataFrame({"text": ["a", "b"], "label": ["x", "y"]})
    validate_text_classification_frame(df, "text", "label")


def test_validate_text_classification_frame_rejects_missing_column() -> None:
    df = pd.DataFrame({"text": ["a", "b"]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_text_classification_frame(df, "text", "label")


def test_build_label_mapping_is_deterministic() -> None:
    labels = pd.Series(["billing", "account", "billing"])
    label2id, id2label = build_label_mapping(labels)
    assert label2id == {"account": 0, "billing": 1}
    assert id2label == {0: "account", 1: "billing"}


def test_encode_labels_adds_label_id() -> None:
    df = pd.DataFrame({"label": ["a", "b"]})
    encoded = encode_labels(df, "label", {"a": 0, "b": 1})
    assert encoded["label_id"].tolist() == [0, 1]
