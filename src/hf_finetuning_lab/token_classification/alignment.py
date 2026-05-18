"""Align word-level NER labels onto subword tokens produced by an HF tokenizer."""

from __future__ import annotations

from collections.abc import Sequence


def align_word_labels_to_subwords(
    word_labels: Sequence[str],
    word_ids: Sequence[int | None],
    strategy: str = "first",
    special_label: str = "O",
) -> list[str]:
    """Map word-level BIO labels onto subword positions.

    ``word_ids`` mirrors the output of ``tokenizer(...).word_ids()`` from
    Hugging Face fast tokenizers: one entry per subword, equal to the source
    word index or ``None`` for special tokens (``[CLS]``, ``[SEP]``, padding).

    Strategies:
      - ``"first"``: the first subword of a word keeps the original label;
        every later subword of the same word receives ``special_label``
        (typically ``"O"`` or ``"X"``). This is the standard CoNLL convention.
      - ``"all"``: every subword of a word inherits the word's label, with
        ``B-<TYPE>`` rewritten to ``I-<TYPE>`` after the first subword so the
        BIO scheme remains consistent.

    Special-token positions (``word_ids[i] is None``) always receive
    ``special_label``.
    """
    if strategy not in {"first", "all"}:
        raise ValueError(f"Unknown alignment strategy '{strategy}'. Use 'first' or 'all'.")

    aligned: list[str] = []
    last_word: int | None = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(special_label)
            last_word = None
            continue
        if word_id < 0 or word_id >= len(word_labels):
            raise ValueError(
                f"word_id={word_id} is out of bounds for word_labels of length {len(word_labels)}."
            )
        label = word_labels[word_id]
        if word_id != last_word:
            aligned.append(label)
        else:
            if strategy == "first":
                aligned.append(special_label)
            else:  # all
                # Convert B-<TYPE> to I-<TYPE> when propagating to a later subword.
                if label.startswith("B-"):
                    aligned.append("I-" + label[2:])
                else:
                    aligned.append(label)
        last_word = word_id
    return aligned
