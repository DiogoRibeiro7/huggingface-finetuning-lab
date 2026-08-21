"""FastAPI app for the Hugging Face Fine-Tuning Lab.

The app exposes a small set of endpoints designed for production-shaped
deployment: liveness/readiness probes, a structured request log, optional
Prometheus metrics, and a warm-up step that primes the model on startup so
the first user request does not pay the cold-start latency.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from hf_finetuning_lab import __version__
from hf_finetuning_lab.serving.config import ServingConfig
from hf_finetuning_lab.serving.logging import StructuredRequestLogger, get_request_logger


class TextClassifierProtocol(Protocol):
    """Minimal predictor contract the serving layer relies on."""

    def predict(self, texts: list[str]) -> list[dict[str, Any]]: ...


PredictorFactory = Callable[[Path], TextClassifierProtocol]


#: Maximum number of texts accepted in a single /predict call. Bounds the
#: work a single request can queue onto the model (resource-exhaustion guard).
MAX_PREDICT_BATCH = 256


class PredictRequest(BaseModel):
    """Prediction request payload."""

    texts: list[str] = Field(
        min_length=1,
        max_length=MAX_PREDICT_BATCH,
        description="Texts to classify.",
    )


class PredictResponse(BaseModel):
    """Prediction response payload."""

    predictions: list[dict[str, Any]]
    model_version: str | None = None


class HealthResponse(BaseModel):
    """Standard health-check payload."""

    status: str
    model_dir: str
    model_version: str | None = None


def _default_predictor_factory(model_dir: Path) -> TextClassifierProtocol:
    """Build the production predictor. Imported lazily so tests stay light."""
    from hf_finetuning_lab.inference.predictor import TextClassificationPredictor

    return TextClassificationPredictor(model_dir=model_dir)


def create_app(
    model_dir: str | Path,
    *,
    predictor_factory: PredictorFactory | None = None,
    warm_up_texts: Sequence[str] | None = ("warm up",),
    model_version: str | None = None,
    enable_metrics: bool = False,
    max_chars_per_text: int = 20_000,
) -> FastAPI:
    """Create the serving app.

    Parameters
    ----------
    model_dir:
        Local model directory passed to the predictor.
    predictor_factory:
        Optional callable that builds the predictor. Defaults to
        :class:`TextClassificationPredictor`. Tests can inject a fake to
        avoid importing transformers.
    warm_up_texts:
        Texts used to prime the model on startup. Set to ``None`` to skip.
    model_version:
        Optional version string echoed in health responses and request logs.
    enable_metrics:
        Mount a Prometheus ``/metrics`` endpoint and instrument requests.
        Requires the ``metrics`` extra (``poetry install --extras metrics``).
    max_chars_per_text:
        Reject any request carrying a longer single text than this.
    """
    model_path = Path(model_dir)
    factory = predictor_factory or _default_predictor_factory
    logger = get_request_logger()
    if not logger.handlers:
        # Default to a single stdout handler so structured logs are visible
        # without configuring logging from outside.
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            predictor = factory(model_path)
            app.state.predictor = predictor
            # Warm-up is a best-effort latency optimisation, not a readiness
            # gate: once the predictor loads the app can serve, so readiness is
            # keyed on predictor availability and a failed or skipped warm-up
            # must not pin the app in a permanent "warming" state.
            if warm_up_texts:
                try:
                    predictor.predict(list(warm_up_texts))
                except Exception:
                    logger.exception("model warm-up failed; serving without warm-up")
            app.state.warmed_up = True
            app.state.startup_error = None
        except Exception as exc:
            app.state.predictor = None
            app.state.warmed_up = False
            # Keep the detail in the server log only: the readiness payload
            # is reachable by any caller and repr(exc) can carry local paths,
            # configuration values, or library internals.
            app.state.startup_error = "model failed to load"
            logger.exception("predictor initialisation failed: %r", exc)
        yield

    app = FastAPI(
        title="Hugging Face Fine-Tuning Lab API",
        version=__version__,
        lifespan=lifespan,
    )
    app.state.model_dir = str(model_path)
    app.state.model_version = model_version
    app.state.max_chars_per_text = max_chars_per_text
    app.state.predictor = None
    app.state.warmed_up = False
    app.state.startup_error = None

    app.add_middleware(StructuredRequestLogger, model_version=model_version)

    if enable_metrics:
        from hf_finetuning_lab.serving.metrics import install_metrics

        install_metrics(app)

    @app.get("/health", response_model=HealthResponse)
    @app.get("/health/live", response_model=HealthResponse)
    def health_live(request: Request) -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_dir=request.app.state.model_dir,
            model_version=request.app.state.model_version,
        )

    @app.get("/health/ready", response_model=HealthResponse)
    def health_ready(request: Request) -> HealthResponse:
        state = request.app.state
        if state.predictor is None:
            raise HTTPException(status_code=503, detail={
                "status": "not_ready",
                "startup_error": state.startup_error,
            })
        if not state.warmed_up:
            raise HTTPException(status_code=503, detail={
                "status": "warming",
                "startup_error": state.startup_error,
            })
        return HealthResponse(
            status="ready",
            model_dir=state.model_dir,
            model_version=state.model_version,
        )

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        predictor: TextClassifierProtocol | None = request.app.state.predictor
        if predictor is None:
            raise HTTPException(status_code=503, detail="predictor not ready")
        limit = request.app.state.max_chars_per_text
        oversized = next((len(text) for text in payload.texts if len(text) > limit), None)
        if oversized is not None:
            # The batch cap alone bounds the number of texts, not their size.
            raise HTTPException(
                status_code=413,
                detail=f"text of {oversized} characters exceeds the {limit} character limit",
            )
        predictions = predictor.predict(payload.texts)
        return PredictResponse(
            predictions=predictions,
            model_version=request.app.state.model_version,
        )

    return app


def create_app_from_config(
    config: ServingConfig,
    *,
    predictor_factory: PredictorFactory | None = None,
) -> FastAPI:
    """Build the app from a resolved :class:`ServingConfig`.

    A Hub source is downloaded at its pinned revision before the app is built,
    so a failure to fetch surfaces at startup rather than on the first request.
    """
    model_dir = config.model_dir
    version = config.model_version
    if model_dir is None:
        from hf_finetuning_lab.publishing import pull_model

        assert config.model_repo_id is not None  # guaranteed by ServingConfig
        model_dir = pull_model(config.model_repo_id, revision=config.model_revision)
        if version is None:
            # Make what is actually being served observable in /health.
            version = f"{config.model_repo_id}@{config.model_revision}"

    return create_app(
        model_dir=model_dir,
        predictor_factory=predictor_factory,
        model_version=version,
        enable_metrics=config.enable_metrics,
        max_chars_per_text=config.max_chars_per_text,
    )
