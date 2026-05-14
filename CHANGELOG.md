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

## [0.1.0] - 2026-05-05

### Added
- End-to-end Hugging Face text-classification workbench with CLI, evaluation, serving, and tests.
