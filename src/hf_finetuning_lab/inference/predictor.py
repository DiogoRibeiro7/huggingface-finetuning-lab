from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hf_finetuning_lab.data.io import load_table


class TextClassificationPredictor:
    """Local Hugging Face text-classification predictor."""

    def __init__(self, model_dir: str | Path, device: int = -1) -> None:
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {self.model_dir}")
        self.device = device
        self.pipeline = self._load_pipeline()
        self.id2label = self._load_id2label()

    def _load_pipeline(self) -> Any:
        try:
            from transformers import pipeline
        except ImportError as exc:  # pragma: no cover
            raise ImportError("Install `transformers` to run inference.") from exc
        return pipeline("text-classification", model=str(self.model_dir), tokenizer=str(self.model_dir), device=self.device, top_k=None)

    def _load_id2label(self) -> dict[int, str]:
        mapping_path = self.model_dir / "label_mapping.json"
        if mapping_path.exists():
            payload = json.loads(mapping_path.read_text(encoding="utf-8"))
            return {int(k): v for k, v in payload["id2label"].items()}
        return {}

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        """Predict labels and probabilities for a list of texts."""
        if not texts:
            return []
        raw_outputs = self.pipeline(texts)
        predictions: list[dict[str, Any]] = []
        for text, outputs in zip(texts, raw_outputs, strict=True):
            # The pipeline returns a list of dictionaries per example when top_k=None.
            sorted_outputs = sorted(outputs, key=lambda item: item["score"], reverse=True)
            best = sorted_outputs[0]
            probs = {f"prob_{item['label']}": float(item["score"]) for item in sorted_outputs}
            predictions.append(
                {
                    "text": text,
                    "predicted_label": str(best["label"]),
                    "confidence": float(best["score"]),
                    **probs,
                }
            )
        return predictions


def batch_predict(
    model_dir: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    text_col: str = "text",
    device: int = -1,
) -> Path:
    """Run batch inference from a local CSV/JSONL file and write predictions."""
    df = load_table(input_path)
    if text_col not in df.columns:
        raise ValueError(f"Missing text column: {text_col}")
    predictor = TextClassificationPredictor(model_dir=model_dir, device=device)
    predictions = predictor.predict(df[text_col].astype(str).tolist())
    output = pd.DataFrame(predictions)
    output.insert(0, "row_id", np.arange(len(output)))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    return destination
