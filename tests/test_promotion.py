from __future__ import annotations

import json
from pathlib import Path

import pytest

from hf_finetuning_lab.governance import (
    PromotionCriterion,
    PromotionReport,
    aggregate_reports,
    boolean_criterion,
    skipped_criterion,
    threshold_criterion,
    write_promotion_report,
)


def test_threshold_criterion_ge_passes_when_value_above() -> None:
    c = threshold_criterion("macro_f1_ci_low", 0.71, 0.65, direction="ge")
    assert c.status == "pass"
    assert c.value == pytest.approx(0.71)
    assert c.threshold == pytest.approx(0.65)
    assert ">=" in c.detail


def test_threshold_criterion_ge_fails_when_value_below() -> None:
    c = threshold_criterion("macro_f1_ci_low", 0.50, 0.65, direction="ge")
    assert c.status == "fail"


def test_threshold_criterion_le_passes_when_value_below() -> None:
    c = threshold_criterion("ece", 0.10, 0.15, direction="le")
    assert c.status == "pass"
    assert "<=" in c.detail


def test_threshold_criterion_ge_passes_on_boundary_equality() -> None:
    # The gate uses inclusive comparison (>=): value == threshold must pass.
    # Guards against a regression flipping >= to >.
    c = threshold_criterion("macro_f1_ci_low", 0.65, 0.65, direction="ge")
    assert c.status == "pass"


def test_threshold_criterion_le_passes_on_boundary_equality() -> None:
    # The gate uses inclusive comparison (<=): value == threshold must pass.
    c = threshold_criterion("ece", 0.15, 0.15, direction="le")
    assert c.status == "pass"


def test_threshold_criterion_rejects_unknown_direction() -> None:
    with pytest.raises(ValueError):
        threshold_criterion("x", 1.0, 1.0, direction="eq")  # type: ignore[arg-type]


def test_boolean_criterion_marks_status() -> None:
    assert boolean_criterion("artifact_ok", ok=True, detail="all green").status == "pass"
    assert boolean_criterion("artifact_ok", ok=False, detail="missing files").status == "fail"


def test_skipped_criterion_marks_status() -> None:
    c = skipped_criterion("gpu_smoke", detail="no GPU available in CI")
    assert c.status == "skip"


def test_promotion_report_promotes_when_every_required_check_passed() -> None:
    report = PromotionReport(
        run_id="r1",
        model_name="tfidf-logreg",
        criteria=[
            boolean_criterion("a", True),
            boolean_criterion("b", True),
            skipped_criterion("c", severity="advisory"),
        ],
    )
    assert report.should_promote
    assert report.blocking_reasons == []
    assert report.failed == []
    assert len(report.passed) == 2
    assert len(report.skipped) == 1


def test_promotion_report_blocks_on_failure() -> None:
    report = PromotionReport(
        run_id="r2",
        model_name="tfidf-logreg",
        criteria=[
            boolean_criterion("a", True),
            boolean_criterion("b", False, "calibration too far off"),
        ],
    )
    assert not report.should_promote
    assert report.failed[0].name == "b"


def test_write_promotion_report_emits_md_and_json(tmp_path: Path) -> None:
    report = PromotionReport(
        run_id="promotion-demo-1",
        model_name="tfidf-logreg",
        criteria=[
            threshold_criterion("macro_f1_ci_low", 0.71, 0.65, direction="ge"),
            threshold_criterion("ece", 0.20, 0.15, direction="le"),
            boolean_criterion("artifact_ok", True, "all required files present"),
            skipped_criterion("gpu_smoke", "no GPU available"),
        ],
        notes="Test report.",
    )
    md_path = write_promotion_report(report, tmp_path / "promotion.md")
    json_path = md_path.with_suffix(".json")
    text = md_path.read_text(encoding="utf-8")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert "BLOCK" in text  # ece criterion fails
    assert "promotion-demo-1" in text
    assert "`macro_f1_ci_low`" in text
    assert "`gpu_smoke`" in text
    assert payload["should_promote"] is False
    assert any(c["name"] == "ece" and c["status"] == "fail" for c in payload["criteria"])


def test_write_promotion_report_emits_promote_verdict_when_clean(tmp_path: Path) -> None:
    report = PromotionReport(
        run_id="promotion-clean",
        model_name="tfidf-logreg",
        criteria=[boolean_criterion("artifact_ok", True)],
    )
    text = write_promotion_report(report, tmp_path / "promotion.md").read_text(encoding="utf-8")
    assert "PROMOTE" in text


def test_aggregate_reports_flattens_summary() -> None:
    a = PromotionReport(
        run_id="a",
        model_name="m1",
        criteria=[boolean_criterion("x", True)],
    )
    b = PromotionReport(
        run_id="b",
        model_name="m2",
        criteria=[boolean_criterion("x", False)],
    )
    rows = aggregate_reports([a, b])
    assert {row["run_id"] for row in rows} == {"a", "b"}
    promote_map = {row["run_id"]: row["should_promote"] for row in rows}
    assert promote_map == {"a": True, "b": False}


def test_promotion_criterion_to_dict_round_trip() -> None:
    c = PromotionCriterion(
        name="ece", status="pass", detail="ok", value=0.05, threshold=0.15
    )
    payload = c.to_dict()
    assert payload["status"] == "pass"
    assert payload["value"] == pytest.approx(0.05)


def test_empty_report_blocks_promotion() -> None:
    """No evidence is not evidence of readiness."""
    report = PromotionReport(run_id="r0", model_name="tfidf-logreg", criteria=[])

    assert not report.should_promote
    assert report.blocking_reasons == ["no required criteria were evaluated"]


def test_skipped_required_criterion_blocks_promotion() -> None:
    report = PromotionReport(
        run_id="r1",
        model_name="tfidf-logreg",
        criteria=[boolean_criterion("a", True), skipped_criterion("gpu_smoke")],
    )

    assert not report.should_promote
    assert report.blocking_reasons == ["required criterion 'gpu_smoke' was not evaluated"]


def test_advisory_criteria_alone_cannot_authorize_promotion() -> None:
    report = PromotionReport(
        run_id="r1",
        model_name="tfidf-logreg",
        criteria=[
            boolean_criterion("nice_to_have", True, severity="advisory"),
            boolean_criterion("also_advisory", True, severity="advisory"),
        ],
    )

    assert not report.should_promote
    assert report.blocking_reasons == ["no required criteria were evaluated"]


def test_failing_advisory_criterion_does_not_block() -> None:
    report = PromotionReport(
        run_id="r1",
        model_name="tfidf-logreg",
        criteria=[
            boolean_criterion("macro_f1", True),
            boolean_criterion("nice_to_have", False, severity="advisory"),
        ],
    )

    assert report.should_promote
    assert len(report.failed) == 1


def test_threshold_criterion_accepts_advisory_severity() -> None:
    criterion = threshold_criterion(
        "latency_ms", 120.0, 100.0, direction="le", severity="advisory"
    )

    assert criterion.status == "fail"
    assert criterion.severity == "advisory"


def test_report_records_blocking_reasons_in_its_payload() -> None:
    report = PromotionReport(
        run_id="r1",
        model_name="tfidf-logreg",
        criteria=[boolean_criterion("macro_f1", False)],
    )

    payload = report.to_dict()

    assert payload["should_promote"] is False
    assert payload["blocking_reasons"] == ["required criterion 'macro_f1' failed"]
