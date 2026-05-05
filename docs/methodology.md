# Methodology

This project implements a standard transformer fine-tuning workflow for text classification.

## Data

Input data must contain a text column and a label column. The sample generator creates synthetic customer-support tickets with labels such as account, billing, technical, delivery, cancellation, and security.

## Tokenization

Texts are tokenized with the tokenizer that corresponds to the selected model. Padding and truncation are applied during preprocessing.

## Fine-tuning

The default workflow uses Hugging Face `Trainer` with `AutoModelForSequenceClassification`. The model is trained on a stratified training split and evaluated on validation/test splits.

## LoRA

LoRA fine-tuning is optional. It uses PEFT adapters to train a smaller set of parameters. This is useful for constrained compute settings.

## Evaluation

The project reports:

- Accuracy
- Precision
- Recall
- F1
- ROC AUC for binary classification when available
- Confusion matrix
- Per-class report

## Serving

The FastAPI service loads a local model directory and exposes `/health` and `/predict` endpoints.
