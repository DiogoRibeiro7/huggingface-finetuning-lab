from __future__ import annotations

from typing import Any


def load_tokenizer(model_name: str):
    """Load a Hugging Face tokenizer by model name."""
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install `transformers` to load tokenizers.") from exc
    return AutoTokenizer.from_pretrained(model_name)


def tokenize_dataset(dataset: Any, tokenizer: Any, text_col: str, max_length: int) -> Any:
    """Tokenize a Hugging Face Dataset for text classification."""
    if max_length <= 0:
        raise ValueError("max_length must be positive.")

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        encoding = tokenizer(
            batch[text_col],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        return dict(encoding)

    return dataset.map(_tokenize, batched=True)
