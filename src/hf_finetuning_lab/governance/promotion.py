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
#: ``required`` criteria gate promotion: each must be evaluated and pass.
#: ``advisory`` criteria are reported but never block.
CriterionSeverity = Literal["required", "advisory"]


@dataclass(slots=True)
class PromotionCriterion:
    """One gate check with its observed value and threshold."""

    name: str
    status: CriterionStatus
    detail: str = ""
    value: float | str | None = None
    threshold: float | str | None = None
    severity: CriterionSeverity = "required"

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
    def required(self) -> list[PromotionCriterion]:
        return [c for c in self.criteria if c.severity == "required"]

    @property
    def blocking_reasons(self) -> list[str]:
        """Why promotion is blocked, empty when the model may ship.

        Absent evidence is not evidence of readiness, so a report with no
        required criteria, or with a required criterion that was never
        evaluated, blocks rather than promotes.
        """
        required = self.required
        if not required:
            return ["no required criteria were evaluated"]
        reasons = [f"required criterion '{c.name}' failed" for c in required if c.status == "fail"]
        reasons += [
            f"required criterion '{c.name}' was not evaluated"
            for c in required
            if c.status == "skip"
        ]
        return reasons

    @property
    def should_promote(self) -> bool:
        """True only when every required criterion was evaluated and passed."""
        return not self.blocking_reasons

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model_name": self.model_name,
            "generated_at_utc": self.generated_at_utc,
            "notes": self.notes,
            "should_promote": self.should_promote,
            "blocking_reasons": self.blocking_reasons,
            "criteria": [c.to_dict() for c in self.criteria],
        }


def threshold_criterion(
    name: str,
    value: float,
    threshold: float,
    *,
    direction: Literal["ge", "le"],
    detail_unit: str = "",
    severity: CriterionSeverity = "required",
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
        severity=severity,
    )


def boolean_criterion(
    name: str,
    ok: bool,
    detail: str = "",
    *,
    severity: CriterionSeverity = "required",
) -> PromotionCriterion:
    """Build a binary pass/fail criterion."""
    return PromotionCriterion(
        name=name,
        status="pass" if ok else "fail",
        detail=detail,
        severity=severity,
    )


def skipped_criterion(
    name: str,
    detail: str = "",
    *,
    severity: CriterionSeverity = "required",
) -> PromotionCriterion:
    """Build a criterion that is intentionally not evaluated.

    A skipped criterion still blocks promotion unless it is marked
    ``advisory``: a check that did not run has produced no evidence.
    """
    return PromotionCriterion(name=name, status="skip", detail=detail, severity=severity)


def write_promotion_report(report: PromotionReport, output_path: str | Path) -> Path:
    """Render ``report`` as Markdown next to a JSON sidecar."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    verdict_word = "PROMOTE" if report.should_promote else "BLOCK"
    reasons = report.blocking_reasons
    blocking_block = (
        "**Blocked by:**\n" + "\n".join(f"- {reason}" for reason in reasons)
        if reasons
        else "_Nothing blocking._"
    )
    rows = [
        "| status | criterion | severity | value | threshold | detail |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for criterion in report.criteria:
        value = "—" if criterion.value is None else str(criterion.value)
        threshold = "—" if criterion.threshold is None else str(criterion.threshold)
        rows.append(
            f"| {criterion.status.upper()} | `{criterion.name}` | {criterion.severity} "
            f"| {value} | {threshold} | {criterion.detail} |"
        )
    body = f"""# Promotion Report: {report.run_id}

## Verdict

**{verdict_word}** — `should_promote = {report.should_promote}`.

- **Model:** `{report.model_name}`
- **Generated (UTC):** `{report.generated_at_utc}`
- **Failed:** {len(report.failed)} | **Passed:** {len(report.passed)} | **Skipped:** {len(report.skipped)}

{blocking_block}

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
                "n_required": len(report.required),
                "generated_at_utc": report.generated_at_utc,
            }
        )
    return rows
