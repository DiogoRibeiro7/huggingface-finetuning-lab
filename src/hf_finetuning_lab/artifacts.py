"""Stable model-artifact layout and verification helpers.

A model directory produced by ``hf-lab train`` should contain everything
required to reload the model, reproduce evaluation, and operate it in
production. :func:`verify_artifact` walks a directory and reports which
elements of that contract are present, which are missing, and which are
optional-but-recommended.

This module is intentionally framework-agnostic — it only inspects the file
layout, not the bytes — so it can run in any environment without importing
heavy ML dependencies.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

CheckStatus = Literal["ok", "missing", "warning"]


REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
)

ALTERNATIVE_REQUIRED_FILES: tuple[tuple[str, ...], ...] = (
    # Weights — either safetensors or PyTorch pickle is acceptable.
    ("model.safetensors", "pytorch_model.bin"),
    # Tokenizer — fast tokenizer or slow vocab file is acceptable.
    ("tokenizer.json", "vocab.txt", "vocab.json"),
)

RECOMMENDED_FILES: tuple[str, ...] = (
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model_card.md",
    "metrics.json",
)


@dataclass(slots=True)
class ArtifactCheck:
    """One file / group of files that was inspected."""

    name: str
    status: CheckStatus
    detail: str = ""


@dataclass(slots=True)
class ArtifactReport:
    """Aggregate output of :func:`verify_artifact`."""

    path: Path
    checks: list[ArtifactCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """True when no required check failed (warnings are allowed)."""
        return all(check.status != "missing" for check in self.checks)

    @property
    def missing(self) -> list[str]:
        return [check.name for check in self.checks if check.status == "missing"]

    @property
    def warnings(self) -> list[str]:
        return [check.name for check in self.checks if check.status == "warning"]


def _present(path: Path, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if (path / name).is_file():
            return name
    return None


def verify_artifact(model_dir: str | Path) -> ArtifactReport:
    """Inspect ``model_dir`` and report on the stable artifact contract.

    The function never raises on a missing or malformed artifact — the caller
    decides what to do based on :attr:`ArtifactReport.ok`, missing names, and
    warnings. Passing a non-directory raises ``FileNotFoundError`` so the
    caller surfaces the problem instead of silently passing.
    """
    path = Path(model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Model directory not found: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    report = ArtifactReport(path=path)

    for name in REQUIRED_FILES:
        present = (path / name).is_file()
        report.checks.append(
            ArtifactCheck(
                name=name,
                status="ok" if present else "missing",
                detail="present" if present else f"required file `{name}` is missing",
            )
        )

    for group in ALTERNATIVE_REQUIRED_FILES:
        match = _present(path, group)
        label = " | ".join(group)
        if match is not None:
            report.checks.append(
                ArtifactCheck(name=label, status="ok", detail=f"present as `{match}`")
            )
        else:
            report.checks.append(
                ArtifactCheck(
                    name=label,
                    status="missing",
                    detail=f"none of {list(group)} present",
                )
            )

    for name in RECOMMENDED_FILES:
        present = (path / name).is_file()
        report.checks.append(
            ArtifactCheck(
                name=name,
                status="ok" if present else "warning",
                detail="present" if present else f"recommended file `{name}` is absent",
            )
        )

    return report
