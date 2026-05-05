# Coding Agent Prompts

## Prompt 1 — Add real Hugging Face dataset support

Implement a CLI command `load-hub-dataset` that downloads a named Hugging Face dataset, selects a split, maps text and label columns, and writes a local CSV.

## Prompt 2 — Add calibration diagnostics

Add expected calibration error, reliability diagrams, and temperature scaling for text-classification logits.

## Prompt 3 — Add model comparison

Train several base models with the same split and produce an experiment comparison report.

## Prompt 4 — Add subgroup metrics

Given one or more metadata columns, compute per-group precision, recall, F1, and error gaps.

## Prompt 5 — Add token classification

Extend the repo to support named entity recognition using Hugging Face token-classification models.
