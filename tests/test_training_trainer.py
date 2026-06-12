from __future__ import annotations

from pathlib import Path

import pandas as pd

from hf_finetuning_lab.training.trainer import (
    _ensure_optimizer_mode_compatibility,
    _write_heldout_test_split,
)


class _BaseOptimizer:
    """Minimal stand-in for a plain torch optimizer without train/eval hooks."""


class _WrappedOptimizer:
    """Accelerate-style wrapper exposing the wrapped optimizer via `.optimizer`."""

    def __init__(self) -> None:
        self.optimizer = _BaseOptimizer()


def test_ensure_optimizer_mode_compatibility_adds_noop_hooks() -> None:
    wrapped = _WrappedOptimizer()

    _ensure_optimizer_mode_compatibility(wrapped)

    assert callable(wrapped.train)
    assert callable(wrapped.eval)
    assert wrapped.train() is None
    assert wrapped.eval() is None


def test_ensure_optimizer_mode_compatibility_preserves_existing_hooks() -> None:
    class _ModeAwareOptimizer:
        def __init__(self) -> None:
            self.called: list[str] = []

        def train(self) -> None:
            self.called.append("train")

        def eval(self) -> None:
            self.called.append("eval")

    class _ModeAwareWrappedOptimizer:
        def __init__(self) -> None:
            self.optimizer = _ModeAwareOptimizer()

        def train(self) -> str:
            self.optimizer.train()
            return "wrapped-train"

        def eval(self) -> str:
            self.optimizer.eval()
            return "wrapped-eval"

    wrapped = _ModeAwareWrappedOptimizer()

    _ensure_optimizer_mode_compatibility(wrapped)

    assert wrapped.train() == "wrapped-train"
    assert wrapped.eval() == "wrapped-eval"
    assert wrapped.optimizer.called == ["train", "eval"]


def test_write_heldout_test_split_persists_rows(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "text": ["alpha", "beta"],
            "label": ["account", "billing"],
            "label_id": [0, 1],
        }
    )

    written = _write_heldout_test_split(frame, tmp_path)

    assert written == tmp_path / "heldout_test.csv"
    restored = pd.read_csv(written)
    assert restored.to_dict(orient="records") == frame.to_dict(orient="records")
