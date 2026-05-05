from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from hf_finetuning_lab.data.io import build_label_mapping, load_table, validate_text_classification_frame
from hf_finetuning_lab.evaluation.metrics import compute_classification_metrics, confusion_matrix_frame
from hf_finetuning_lab.inference.predictor import TextClassificationPredictor


def evaluate_model(
    model_dir: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    text_col: str = "text",
    label_col: str = "label",
) -> Path:
    """Evaluate a local text-classification model on a labelled dataset."""
    df = load_table(input_path)
    validate_text_classification_frame(df, text_col=text_col, label_col=label_col)

    label2id, id2label = build_label_mapping(df[label_col])
    predictor = TextClassificationPredictor(model_dir=model_dir)
    predictions = predictor.predict(df[text_col].astype(str).tolist())
    pred_df = pd.DataFrame(predictions)

    y_true_labels = df[label_col].astype(str).tolist()
    y_pred_labels = pred_df["predicted_label"].astype(str).tolist()

    # For models that use LABEL_0 style outputs, map IDs back if possible.
    normalized_pred_labels = []
    for label in y_pred_labels:
        if label.startswith("LABEL_"):
            idx = int(label.replace("LABEL_", ""))
            normalized_pred_labels.append(id2label.get(idx, label))
        else:
            normalized_pred_labels.append(label)

    y_true = np.array([label2id[label] for label in y_true_labels])
    y_pred = np.array([label2id.get(label, -1) for label in normalized_pred_labels])
    metrics = compute_classification_metrics(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix_frame(y_true, y_pred, id2label)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm.to_dict(),
    }
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return destination
