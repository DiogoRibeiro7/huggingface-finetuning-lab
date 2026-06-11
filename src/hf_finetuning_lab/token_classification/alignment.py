"""Align word-level NER labels onto subword tokens produced by an HF tokenizer."""

from __future__ import annotations

from collections.abc import Sequence


def align_word_labels_to_subwords(
    word_labels: Sequence[str],
    word_ids: Sequence[int | None],
    strategy: str = "first",
    ignore_index: int = -100,
) -> list[int | str]:
    """Map word-level BIO labels onto subword positions.

    ``word_ids`` mirrors the output of ``tokenizer(...).word_ids()`` from
    Hugging Face fast tokenizers: one entry per subword, equal to the source
    word index or ``None`` for special tokens (``[CLS]``, ``[SEP]``, padding).

    Strategies:
      - ``"first"``: the first subword of a word keeps the original label;
        every later subword of the same word is masked with ``ignore_index``.
        This is the standard Hugging Face convention — masked positions are
        skipped by the cross-entropy loss (``ignore_index=-100``) rather than
        being taught a contradictory ``"O"`` label.
      - ``"all"``: every subword of a word inherits the word's label, with
        ``B-<TYPE>`` rewritten to ``I-<TYPE>`` after the first subword so the
        BIO scheme remains consistent.

    Special-token positions (``word_ids[i] is None``) are always masked with
    ``ignore_index``. The returned list therefore mixes label strings (for
    supervised positions) and the integer ``ignore_index`` (for masked ones).
    """
    if strategy not in {"first", "all"}:
        raise ValueError(f"Unknown alignment strategy '{strategy}'. Use 'first' or 'all'.")

    aligned: list[int | str] = []
    last_word: int | None = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(ignore_index)
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
                aligned.append(ignore_index)
            else:  # all
                # Convert B-<TYPE> to I-<TYPE> when propagating to a later subword.
                if label.startswith("B-"):
                    aligned.append("I-" + label[2:])
                else:
                    aligned.append(label)
        last_word = word_id
    return aligned
