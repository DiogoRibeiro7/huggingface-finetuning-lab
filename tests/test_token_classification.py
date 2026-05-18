from __future__ import annotations

import json
from pathlib import Path

import pytest

from hf_finetuning_lab.token_classification import (
    NERExample,
    align_word_labels_to_subwords,
    extract_entities,
    generate_synthetic_ner_data,
    sequence_tagging_report,
    validate_ner_dataset,
    write_synthetic_ner_jsonl,
)


def test_ner_example_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        NERExample(tokens=["a", "b"], labels=["O"])


def test_generate_synthetic_data_is_well_formed() -> None:
    examples = generate_synthetic_ner_data(n_examples=20, seed=0)
    assert len(examples) == 20
    validate_ner_dataset(examples)
    # At least one PER, ORG, and LOC should appear across 20 templates.
    label_set: set[str] = set()
    for example in examples:
        label_set.update(example.labels)
    assert any(label.endswith("-PER") for label in label_set)
    assert any(label.endswith("-ORG") for label in label_set)
    assert any(label.endswith("-LOC") for label in label_set)


def test_validate_ner_dataset_rejects_bad_label() -> None:
    examples = [NERExample(tokens=["x"], labels=["NOT_BIO"])]
    with pytest.raises(ValueError):
        validate_ner_dataset(examples)


def test_validate_ner_dataset_rejects_empty() -> None:
    with pytest.raises(ValueError):
        validate_ner_dataset([])


def test_write_synthetic_ner_jsonl_round_trips(tmp_path: Path) -> None:
    output = tmp_path / "ner.jsonl"
    path = write_synthetic_ner_jsonl(output, n_examples=5, seed=1)
    assert path.exists()
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    first = json.loads(lines[0])
    assert "tokens" in first and "labels" in first


def test_extract_entities_basic() -> None:
    labels = ["O", "B-PER", "I-PER", "O", "B-ORG", "O"]
    spans = extract_entities(labels)
    assert spans == [("PER", 1, 3), ("ORG", 4, 5)]


def test_extract_entities_handles_stray_i() -> None:
    labels = ["I-PER", "I-PER", "O", "I-ORG"]
    spans = extract_entities(labels)
    assert spans == [("PER", 0, 2), ("ORG", 3, 4)]


def test_extract_entities_all_o() -> None:
    assert extract_entities(["O", "O", "O"]) == []


def test_extract_entities_adjacent_b_starts_new_span() -> None:
    labels = ["B-PER", "B-PER", "O"]
    spans = extract_entities(labels)
    assert spans == [("PER", 0, 1), ("PER", 1, 2)]


def test_sequence_tagging_report_perfect_predictions() -> None:
    true_seq = [["B-PER", "I-PER", "O"], ["O", "B-LOC"]]
    report = sequence_tagging_report(true_seq, true_seq)
    assert report.loc["PER", "f1"] == pytest.approx(1.0)
    assert report.loc["LOC", "f1"] == pytest.approx(1.0)
    assert report.loc["micro avg", "f1"] == pytest.approx(1.0)


def test_sequence_tagging_report_counts_missed_entities() -> None:
    true_seq = [["B-PER", "I-PER", "O"]]
    pred_seq = [["O", "O", "O"]]
    report = sequence_tagging_report(true_seq, pred_seq)
    assert report.loc["PER", "support"] == 1
    assert report.loc["PER", "recall"] == 0.0
    assert report.loc["PER", "precision"] == 0.0


def test_sequence_tagging_report_partial_match_counts_as_miss() -> None:
    # Boundary mismatch (single token instead of two) should not be a hit.
    true_seq = [["B-PER", "I-PER"]]
    pred_seq = [["B-PER", "O"]]
    report = sequence_tagging_report(true_seq, pred_seq)
    assert report.loc["PER", "f1"] == pytest.approx(0.0)


def test_sequence_tagging_report_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        sequence_tagging_report([["O"]], [["O"], ["O"]])


def test_align_first_strategy_marks_continuation_subwords_special() -> None:
    word_labels = ["B-PER", "O"]
    word_ids = [None, 0, 0, 1, None]
    aligned = align_word_labels_to_subwords(word_labels, word_ids, strategy="first")
    assert aligned == ["O", "B-PER", "O", "O", "O"]


def test_align_all_strategy_propagates_with_bio_consistency() -> None:
    word_labels = ["B-PER", "O"]
    word_ids = [0, 0, 1]
    aligned = align_word_labels_to_subwords(word_labels, word_ids, strategy="all")
    assert aligned == ["B-PER", "I-PER", "O"]


def test_align_rejects_bad_strategy() -> None:
    with pytest.raises(ValueError):
        align_word_labels_to_subwords(["O"], [0], strategy="nonsense")


def test_align_rejects_word_id_out_of_range() -> None:
    with pytest.raises(ValueError):
        align_word_labels_to_subwords(["O"], [0, 1], strategy="first")
