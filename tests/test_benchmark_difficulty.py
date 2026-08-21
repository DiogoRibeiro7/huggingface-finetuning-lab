"""Difficulty knobs on the synthetic benchmark, and leakage-aware splitting."""

from __future__ import annotations

import pandas as pd
import pytest

from hf_finetuning_lab.data.splits import (
    duplicate_text_report,
    stratified_train_valid_test_split,
)
from hf_finetuning_lab.sample_data import generate_support_ticket_data


def test_default_generation_is_unchanged_in_shape() -> None:
    frame = generate_support_ticket_data(rows=50, seed=1)

    assert list(frame.columns) == ["id", "text", "label", "template_family"]
    assert len(frame) == 50


def test_template_family_groups_rows_from_the_same_phrasing() -> None:
    frame = generate_support_ticket_data(rows=200, seed=1)

    # Every family belongs to exactly one class.
    per_family = frame.groupby("template_family")["label"].nunique()
    assert set(per_family.unique()) == {1}


def test_label_noise_moves_labels_off_their_template_class() -> None:
    clean = generate_support_ticket_data(rows=400, seed=3)
    noisy = generate_support_ticket_data(rows=400, seed=3, label_noise=0.3)

    def mismatch_rate(frame: pd.DataFrame) -> float:
        family_class = frame["template_family"].str.split(":").str[0]
        return float((family_class != frame["label"]).mean())

    assert mismatch_rate(clean) == 0.0
    assert 0.2 < mismatch_rate(noisy) < 0.4


def test_ambiguity_blends_in_another_class_phrasing() -> None:
    plain = generate_support_ticket_data(rows=200, seed=5)
    blended = generate_support_ticket_data(rows=200, seed=5, ambiguity=0.5)

    assert blended["text"].str.len().mean() > plain["text"].str.len().mean()


@pytest.mark.parametrize("kwargs", [{"label_noise": 1.0}, {"ambiguity": 1.5}])
def test_invalid_difficulty_settings_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        generate_support_ticket_data(rows=10, **kwargs)


def test_grouped_split_keeps_a_group_within_one_split() -> None:
    """A near-duplicate phrasing in both train and test inflates the score."""
    frame = generate_support_ticket_data(rows=400, seed=7)

    train, valid, test = stratified_train_valid_test_split(
        frame, label_col="label", group_col="template_family"
    )

    for name, part in (("valid", valid), ("test", test)):
        overlap = set(train["template_family"]) & set(part["template_family"])
        assert not overlap, f"train and {name} share families: {sorted(overlap)}"


def test_grouped_split_covers_every_row() -> None:
    frame = generate_support_ticket_data(rows=300, seed=11)

    train, valid, test = stratified_train_valid_test_split(
        frame, label_col="label", group_col="template_family"
    )

    assert len(train) + len(valid) + len(test) == len(frame)


def test_grouped_split_rejects_a_missing_column() -> None:
    frame = generate_support_ticket_data(rows=20, seed=1)

    with pytest.raises(ValueError, match="group_col not found"):
        stratified_train_valid_test_split(frame, label_col="label", group_col="thread_id")


def test_grouped_split_rejects_too_few_groups() -> None:
    frame = pd.DataFrame({"text": ["a", "b"], "label": ["x", "y"], "thread": ["t1", "t1"]})

    with pytest.raises(ValueError, match="at least 2 distinct"):
        stratified_train_valid_test_split(frame, label_col="label", group_col="thread")


def test_duplicate_report_lists_repeated_text() -> None:
    frame = pd.DataFrame(
        {"text": ["same", "same", "same", "unique"], "label": ["a", "a", "b", "b"]}
    )

    report = duplicate_text_report(frame, "text")

    assert report.to_dict(orient="records") == [{"text": "same", "count": 3}]


def test_duplicate_report_is_empty_when_rows_are_distinct() -> None:
    frame = generate_support_ticket_data(rows=5, seed=1)
    frame["text"] = [f"unique text {i}" for i in range(len(frame))]

    assert duplicate_text_report(frame, "text").empty
