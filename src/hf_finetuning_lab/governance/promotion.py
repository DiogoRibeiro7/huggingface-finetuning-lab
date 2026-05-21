"""Composable model-promotion gate.

A *promotion gate* answers a single question: is this model ready to ship?
The answer is a structured report — one row per criterion, plus a derived
verdict — so deployment pipelines can fail fast on a single missing check
and humans can read the same report on a PR.

This module deliberately ships only the data model and rendering helpers.
The actual checks (bootstrap CIs, calibration, subgroup ratios, drift,
artifact verification, governance presence) live in their respective
modules and are composed by the caller (typically
``notebooks/10_promotion_gate.ipynb`` or a deployment script).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

CriterionStatus = Literal["pass", "fail", "skip"]


@dataclass(slots=True)
class PromotionCriterion:
    """One gate check with its observed value and threshold."""

    name: str
    status: CriterionStatus
    detail: str = ""
    value: float | str | None = None
    threshold: float | str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PromotionReport:
    """All criteria evaluated for one candidate model."""

    run_id: str
    model_name: str
    criteria: list[PromotionCriterion] = field(default_factory=list)
    notes: str | None = None
    generated_at_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"),
    )

    @property
    def failed(self) -> list[PromotionCriterion]:
        return [c for c in self.criteria if c.status == "fail"]

    @property
    def skipped(self) -> list[PromotionCriterion]:
        return [c for c in self.criteria if c.status == "skip"]

    @property
    def passed(self) -> list[PromotionCriterion]:
        return [c for c in self.criteria if c.status == "pass"]

    @property
    def should_promote(self) -> bool:
        """True when no criterion failed. Skipped criteria do not block."""
        return not self.failed

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "generated_at_utc": self.generated_at_utc,
            "notes": self.notes,
            "should_promote": self.should_promote,
            "criteria": [c.to_dict() for c in self.criteria],
        }


def threshold_criterion(
    name: str,
    value: float,
    threshold: float,
    *,
    direction: Literal["ge", "le"],
    detail_unit: str = "",
) -> PromotionCriterion:
    """Build a numeric pass/fail criterion.

    ``direction='ge'`` requires ``value >= threshold``; ``direction='le'``
    requires ``value <= threshold``. ``detail_unit`` is appended to the
    rendered detail string (e.g. ``"%"`` or ``" macro F1"``).
    """
    if direction == "ge":
        ok = value >= threshold
        comparator = ">="
    elif direction == "le":
        ok = value <= threshold
        comparator = "<="
    else:
        raise ValueError(f"Unknown direction '{direction}'. Use 'ge' or 'le'.")
    detail = f"{value:.4f} {comparator} {threshold:.4f}{detail_unit}"
    return PromotionCriterion(
        name=name,
        status="pass" if ok else "fail",
        detail=detail,
        value=float(value),
        threshold=float(threshold),
    )


def boolean_criterion(name: str, ok: bool, detail: str = "") -> PromotionCriterion:
    """Build a binary pass/fail criterion."""
    return PromotionCriterion(
        name=name,
        status="pass" if ok else "fail",
        detail=detail,
    )


def skipped_criterion(name: str, detail: str = "") -> PromotionCriterion:
    """Build a criterion that is intentionally not evaluated."""
    return PromotionCriterion(name=name, status="skip", detail=detail)


def write_promotion_report(report: PromotionReport, output_path: str | Path) -> Path:
    """Render ``report`` as Markdown next to a JSON sidecar."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    verdict_word = "PROMOTE" if report.should_promote else "BLOCK"
    rows = [
        "| status | criterion | value | threshold | detail |",
        "| --- | --- | --- | --- | --- |",
    ]
    for criterion in report.criteria:
        value = "—" if criterion.value is None else str(criterion.value)
        threshold = "—" if criterion.threshold is None else str(criterion.threshold)
        rows.append(
            f"| {criterion.status.upper()} | `{criterion.name}` | {value} | {threshold} | {criterion.detail} |"
        )
    body = f"""# Promotion Report: {report.run_id}

## Verdict

**{verdict_word}** — `should_promote = {report.should_promote}`.

- **Model:** `{report.model_name}`
- **Generated (UTC):** `{report.generated_at_utc}`
- **Failed:** {len(report.failed)} | **Passed:** {len(report.passed)} | **Skipped:** {len(report.skipped)}

## Criteria

{chr(10).join(rows)}

## Notes

{report.notes or "_No notes._"}
"""
    destination.write_text(body.strip() + "\n", encoding="utf-8")
    json_path = destination.with_suffix(".json")
    json_path.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return destination


def aggregate_reports(reports: Iterable[PromotionReport]) -> list[dict[str, Any]]:
    """Flatten multiple promotion reports into rows for comparison."""
    rows: list[dict[str, Any]] = []
    for report in reports:
        rows.append(
            {
                "run_id": report.run_id,
                "model_name": report.model_name,
                "should_promote": report.should_promote,
                "n_failed": len(report.failed),
                "n_passed": len(report.passed),
                "n_skipped": len(report.skipped),
                "generated_at_utc": report.generated_at_utc,
            }
        )
    return rows
