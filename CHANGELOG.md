# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added
- Repository professionalization baseline: CI quality gates, contributor workflow, and release automation.
- `hf_finetuning_lab.experiments` module: run IDs, dataset hashing, run-record persistence, and run-comparison DataFrame.
- `per_class_report` helper in `hf_finetuning_lab.evaluation.metrics`.
- `notebooks/02_experiment_management.ipynb`: repeated TF-IDF + LogReg runs with persisted records, side-by-side comparison, per-class report, and confusion-matrix heatmap.
- `hf_finetuning_lab.evaluation.robust` module: reliability curves, expected calibration error, bootstrap confidence intervals, threshold optimization, subgroup metrics, and label-share PSI drift.
- `notebooks/03_robust_evaluation.ipynb`: reliability diagram, threshold sweep, bootstrap CIs, subgroup table, and drift visualization on the synthetic support-ticket task.
- `hf_finetuning_lab.data.hub` module: `HubDatasetConfig`, `HUB_PRESETS` (AG News, IMDb, Banking77, TweetEval sentiment), `load_hub_dataset`, `normalize_hub_dataset_dict`, and `write_hub_dataset_csv`.
- CLI commands `hf-lab list-hub-datasets` and `hf-lab fetch-hub-dataset` for downloading Hub presets to local CSV.
- `notebooks/04_hub_datasets.ipynb`: preset registry walkthrough, offline mock-DatasetDict normalization, opt-in real Hub download, and a TF-IDF baseline on the normalized schema.

## [0.1.0] - 2026-05-05

### Added
- End-to-end Hugging Face text-classification workbench with CLI, evaluation, serving, and tests.
