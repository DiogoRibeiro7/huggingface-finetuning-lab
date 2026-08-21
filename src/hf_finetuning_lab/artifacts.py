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

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

CheckStatus = Literal["ok", "missing", "warning"]


REQUIRED_FILES: tuple[str, ...] = (
    "config.json",
    "label_mapping.json",
    "training_config.json",
    "heldout_manifest.json",
)

ALTERNATIVE_REQUIRED_FILES: tuple[tuple[str, ...], ...] = (
    # Weights — safetensors or PyTorch pickle, single-file or sharded. A
    # sharded checkpoint ships an index manifest instead of one weight file.
    (
        "model.safetensors",
        "pytorch_model.bin",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ),
    # Tokenizer — fast tokenizer, WordPiece/BPE vocab, or a SentencePiece model.
    (
        "tokenizer.json",
        "vocab.txt",
        "vocab.json",
        "spiece.model",
        "tokenizer.model",
        "sentencepiece.bpe.model",
    ),
)

RECOMMENDED_FILES: tuple[str, ...] = (
    "tokenizer_config.json",
    "special_tokens_map.json",
    "model_card.md",
    "test_metrics.json",
    "preprocessing.json",
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


def verify_artifact(model_dir: str | Path, deep: bool = False) -> ArtifactReport:
    """Inspect ``model_dir`` and report on the stable artifact contract.

    The function never raises on a missing or malformed artifact — the caller
    decides what to do based on :attr:`ArtifactReport.ok`, missing names, and
    warnings. Passing a non-directory raises ``FileNotFoundError`` so the
    caller surfaces the problem instead of silently passing.

    The default pass inspects the file layout only, so it is fast and needs no
    ML dependencies — but it cannot tell a real checkpoint from a directory of
    placeholder files. ``deep=True`` additionally parses the JSON, checks the
    label space against the model config, and asks transformers to load the
    tokenizer and model, which is what catches a corrupt or mismatched
    artifact before it reaches production.
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

    if deep:
        report.checks.extend(_deep_checks(path))

    return report


def _check(name: str, ok: bool, detail: str, *, warn_only: bool = False) -> ArtifactCheck:
    status: CheckStatus = "ok" if ok else ("warning" if warn_only else "missing")
    return ArtifactCheck(name=name, status=status, detail=detail)


def _load_json(path: Path) -> tuple[Any | None, str]:
    try:
        return json.loads(path.read_text(encoding="utf-8")), "parsed"
    except FileNotFoundError:
        return None, "file is absent"
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return None, f"is not valid JSON: {exc}"


def _weight_checks(path: Path) -> list[ArtifactCheck]:
    weights = [
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix in {".safetensors", ".bin"}
    ]
    if not weights:
        return [_check("deep:weights", False, "no weight file found")]
    empty = sorted(w.name for w in weights if w.stat().st_size == 0)
    return [
        _check(
            "deep:weights",
            not empty,
            f"{len(weights)} weight file(s)" if not empty else f"empty weight file(s): {empty}",
        )
    ]


def _label_space_checks(path: Path) -> list[ArtifactCheck]:
    checks: list[ArtifactCheck] = []
    config, config_detail = _load_json(path / "config.json")
    checks.append(_check("deep:config.json", config is not None, f"config.json {config_detail}"))

    mapping, mapping_detail = _load_json(path / "label_mapping.json")
    checks.append(
        _check("deep:label_mapping.json", mapping is not None, f"label_mapping.json {mapping_detail}")
    )
    if not isinstance(config, dict) or not isinstance(mapping, dict):
        return checks

    label2id = mapping.get("label2id")
    if not isinstance(label2id, dict):
        checks.append(_check("deep:label_space", False, "label_mapping.json has no label2id object"))
        return checks

    config_labels = config.get("id2label")
    if isinstance(config_labels, dict):
        matches = len(config_labels) == len(label2id)
        checks.append(
            _check(
                "deep:label_space",
                matches,
                f"{len(label2id)} labels in mapping, {len(config_labels)} in config"
                + ("" if matches else " — the classification head does not match the label space"),
            )
        )
    else:
        checks.append(
            _check("deep:label_space", True, "config declares no id2label; nothing to compare", warn_only=True)
        )
    return checks


def _metrics_checks(path: Path) -> list[ArtifactCheck]:
    metrics_path = path / "test_metrics.json"
    if not metrics_path.is_file():
        return [_check("deep:test_metrics.json", True, "absent", warn_only=True)]
    payload, detail = _load_json(metrics_path)
    if not isinstance(payload, dict):
        return [_check("deep:test_metrics.json", False, f"test_metrics.json {detail}")]
    bad = sorted(
        name
        for name, value in payload.items()
        if isinstance(value, float) and not math.isfinite(value)
    )
    return [
        _check(
            "deep:test_metrics.json",
            not bad,
            "all values finite" if not bad else f"non-finite metric values: {bad}",
        )
    ]


def _loadability_checks(path: Path) -> list[ArtifactCheck]:
    """Ask transformers to actually load the artifact."""
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return [
            _check(
                "deep:loadable",
                True,
                "transformers is not installed; skipped the load check",
                warn_only=True,
            )
        ]

    checks: list[ArtifactCheck] = []
    try:
        AutoTokenizer.from_pretrained(str(path))
        checks.append(_check("deep:tokenizer_loads", True, "tokenizer loaded"))
    except Exception as exc:
        checks.append(_check("deep:tokenizer_loads", False, f"tokenizer failed to load: {exc}"))

    try:
        model = AutoModelForSequenceClassification.from_pretrained(str(path))
        n_labels = int(getattr(model.config, "num_labels", 0))
        checks.append(_check("deep:model_loads", True, f"model loaded with {n_labels} labels"))
    except Exception as exc:
        checks.append(_check("deep:model_loads", False, f"model failed to load: {exc}"))
    return checks


def _deep_checks(path: Path) -> list[ArtifactCheck]:
    checks: list[ArtifactCheck] = []
    checks.extend(_weight_checks(path))
    checks.extend(_label_space_checks(path))
    checks.extend(_metrics_checks(path))
    checks.extend(_loadability_checks(path))
    return checks
