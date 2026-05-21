from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from hf_finetuning_lab.serving.api import create_app
from hf_finetuning_lab.serving.logging import REQUEST_LOGGER_NAME


class FakePredictor:
    """In-memory predictor used to keep serving tests off the HF stack."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.fail_warm_up: bool = False
        self.fail_predict: bool = False

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        self.calls.append(list(texts))
        if self.fail_predict:
            raise RuntimeError("forced predict failure")
        if self.fail_warm_up and texts == ["warm up"]:
            raise RuntimeError("forced warm-up failure")
        return [{"label": "billing", "score": 0.42} for _ in texts]


def _build_client(
    tmp_path: Path,
    *,
    predictor: FakePredictor | None = None,
    warm_up_texts: tuple[str, ...] | None = ("warm up",),
    model_version: str | None = "test-1",
) -> tuple[TestClient, FakePredictor]:
    predictor = predictor or FakePredictor()
    app = create_app(
        model_dir=tmp_path,
        predictor_factory=lambda _path: predictor,
        warm_up_texts=warm_up_texts,
        model_version=model_version,
    )
    return TestClient(app), predictor


def test_liveness_returns_ok(tmp_path: Path) -> None:
    client, _ = _build_client(tmp_path)
    with client:
        for path in ["/health", "/health/live"]:
            response = client.get(path)
            assert response.status_code == 200
            payload = response.json()
            assert payload["status"] == "ok"
            assert payload["model_dir"] == str(tmp_path)
            assert payload["model_version"] == "test-1"


def test_readiness_returns_ok_after_warm_up(tmp_path: Path) -> None:
    client, predictor = _build_client(tmp_path)
    with client:
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
        assert predictor.calls and predictor.calls[0] == ["warm up"]


def test_readiness_503_when_predictor_factory_raises(tmp_path: Path) -> None:
    def broken_factory(_path: Path) -> FakePredictor:
        raise RuntimeError("model artefact missing")

    app = create_app(
        model_dir=tmp_path,
        predictor_factory=broken_factory,
        warm_up_texts=None,
        model_version=None,
    )
    with TestClient(app) as client:
        response = client.get("/health/ready")
        assert response.status_code == 503
        body = response.json()
        assert body["detail"]["status"] == "not_ready"
        assert "model artefact missing" in body["detail"]["startup_error"]


def test_readiness_503_when_warm_up_fails(tmp_path: Path) -> None:
    predictor = FakePredictor()
    predictor.fail_warm_up = True
    client, _ = _build_client(tmp_path, predictor=predictor)
    with client:
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["detail"]["status"] == "warming"


def test_predict_returns_predictions_and_model_version(tmp_path: Path) -> None:
    client, predictor = _build_client(tmp_path)
    with client:
        response = client.post("/predict", json={"texts": ["hello", "world"]})
        assert response.status_code == 200
        payload = response.json()
        assert len(payload["predictions"]) == 2
        assert payload["predictions"][0]["label"] == "billing"
        assert payload["model_version"] == "test-1"
        # First call is the warm-up; the second is the user request.
        assert predictor.calls == [["warm up"], ["hello", "world"]]


def test_predict_rejects_empty_texts(tmp_path: Path) -> None:
    client, _ = _build_client(tmp_path)
    with client:
        response = client.post("/predict", json={"texts": []})
        assert response.status_code == 422


def test_request_logger_emits_structured_payload(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    client, _ = _build_client(tmp_path)
    logger = logging.getLogger(REQUEST_LOGGER_NAME)
    logger.propagate = True  # let caplog capture it
    try:
        with caplog.at_level(logging.INFO, logger=REQUEST_LOGGER_NAME):
            with client:
                response = client.get("/health/live")
                assert response.status_code == 200
        request_records = [
            json.loads(record.getMessage())
            for record in caplog.records
            if record.name == REQUEST_LOGGER_NAME
        ]
        assert request_records, "expected at least one request log record"
        live = next(rec for rec in request_records if rec["path"] == "/health/live")
        assert live["method"] == "GET"
        assert live["status"] == 200
        assert live["model_version"] == "test-1"
        assert live["latency_ms"] >= 0
    finally:
        logger.propagate = False


def test_predict_503_when_predictor_is_missing(tmp_path: Path) -> None:
    def broken_factory(_path: Path) -> FakePredictor:
        raise RuntimeError("nope")

    app = create_app(
        model_dir=tmp_path,
        predictor_factory=broken_factory,
        warm_up_texts=None,
    )
    with TestClient(app) as client:
        response = client.post("/predict", json={"texts": ["x"]})
        assert response.status_code == 503
        assert response.json()["detail"] == "predictor not ready"
