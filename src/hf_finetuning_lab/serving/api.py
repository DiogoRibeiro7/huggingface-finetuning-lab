from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel, Field

from hf_finetuning_lab.inference.predictor import TextClassificationPredictor


class PredictRequest(BaseModel):
    """Prediction request payload."""

    texts: list[str] = Field(min_length=1, description="Texts to classify.")


class PredictResponse(BaseModel):
    """Prediction response payload."""

    predictions: list[dict]


def create_app(model_dir: str | Path) -> FastAPI:
    """Create a FastAPI application bound to a local model directory."""
    app = FastAPI(title="Hugging Face Fine-Tuning Lab API", version="0.1.0")
    model_path = Path(model_dir)

    @lru_cache(maxsize=1)
    def get_predictor() -> TextClassificationPredictor:
        return TextClassificationPredictor(model_dir=model_path)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model_dir": str(model_path)}

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        predictions = get_predictor().predict(payload.texts)
        return PredictResponse(predictions=predictions)

    return app
