from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from hf_finetuning_lab.config import TrainingConfig
from hf_finetuning_lab.training.trainer import (
    CompatibleTrainer,
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


def _heldout_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text": ["alpha", "beta"],
            "label": ["account", "billing"],
            "label_id": [0, 1],
        }
    )


def test_write_heldout_test_split_writes_a_manifest_by_default(tmp_path: Path) -> None:
    frame = _heldout_frame()

    written = _write_heldout_test_split(frame, tmp_path, TrainingConfig())

    assert written == tmp_path / "heldout_manifest.json"
    payload = json.loads(written.read_text(encoding="utf-8"))
    assert payload["n_rows"] == 2
    # The evaluation text must not travel with the model directory.
    assert not (tmp_path / "heldout_test.csv").exists()


def test_write_heldout_test_split_can_persist_rows(tmp_path: Path) -> None:
    frame = _heldout_frame()

    _write_heldout_test_split(frame, tmp_path, TrainingConfig(persist_heldout_rows=True))

    restored = pd.read_csv(tmp_path / "heldout_test.csv")
    assert restored.to_dict(orient="records") == frame.to_dict(orient="records")


def test_compatible_trainer_create_optimizer_matches_base_contract() -> None:
    """Transformers >=5 passes the model positionally and uses the return value."""
    seen: dict[str, object] = {}

    class _BaseTrainer:
        def create_optimizer(self, model=None):
            seen["model"] = model
            self.optimizer = _WrappedOptimizer()
            return self.optimizer

    class _Trainer(CompatibleTrainer, _BaseTrainer):
        pass

    trainer = _Trainer()
    sentinel = object()

    optimizer = trainer.create_optimizer(sentinel)

    assert seen["model"] is sentinel
    assert optimizer is trainer.optimizer
    # The mode shims are installed on the returned wrapper.
    assert optimizer.train() is None
    assert optimizer.eval() is None
