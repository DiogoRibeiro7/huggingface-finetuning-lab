# Data Model

## Input dataset

Required columns:

| Column | Type | Description |
|---|---:|---|
| text | string | Raw text input. |
| label | string or integer | Class label. |

Optional columns may be preserved but are not required by the core training pipeline.

## Prediction output

| Column | Type | Description |
|---|---:|---|
| row_id | integer | Row number from input data. |
| text | string | Input text. |
| predicted_label | string | Predicted class label. |
| predicted_label_id | integer | Predicted class ID. |
| confidence | float | Maximum predicted class probability. |
| prob_* | float | Per-class probabilities. |

## Model artifact directory

A trained model directory contains:

```text
config.json
tokenizer files
model weights
label_mapping.json
training_config.json
model_card.md
```
