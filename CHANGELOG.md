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
- `hf_finetuning_lab.token_classification` module: NER schema (`NERExample`, synthetic data generation, JSONL writer, validator), subword alignment (`align_word_labels_to_subwords` with `first` and `all` strategies), and entity-level metrics (`extract_entities`, `sequence_tagging_report`).
- `notebooks/05_token_classification.ipynb`: synthetic CoNLL-style NER, label/entity distribution, subword alignment demo, per-token logistic-regression baseline, and entity-level micro/macro P/R/F1.
- `hf_finetuning_lab.retrieval` module: `EmbeddingIndex` (cosine search over L2-normalised embeddings), `IndexEntry`, `l2_normalize`, plus retrieval metrics `recall_at_k`, `mean_reciprocal_rank`, `ndcg_at_k`, `retrieval_report`.
- `notebooks/06_semantic_search.ipynb`: synthetic FAQ corpus, TF-IDF embedding index, cosine retrieval with Recall@k / MRR / nDCG@k, error inspection, and an opt-in sentence-transformer comparison.
- `hf_finetuning_lab.governance` module: `DatasetCard` / `DatasetColumn` / `DatasetSplit` + `write_dataset_card`, `task_limitations` and `write_task_model_card` for text-classification, token-classification, and retrieval, and `ReproducibilityRecord` + `capture_environment` + `write_reproducibility_checklist` (Markdown + JSON sidecar with environment, seed, dataset hash, and git commit metadata).
- `notebooks/07_governance_template.ipynb`: end-to-end governance walkthrough — trains a small baseline, writes a dataset card with split-level label distributions, a task-specific model card, and a reproducibility checklist tying together run ID, dataset hash, environment snapshot, and metrics.
- `hf_finetuning_lab.serving` deployment hardening: `create_app` now accepts a `predictor_factory` (lazy/injectable predictor), runs model warm-up on startup, and exposes `/health/live` + `/health/ready` (with 503 + diagnostic payload when the predictor cannot load). `StructuredRequestLogger` emits one JSON log line per request; `install_metrics(app)` mounts a Prometheus `/metrics` endpoint when `prometheus-client` is installed.
- `docker-compose.yml` at the repo root plus a `HEALTHCHECK` in the Dockerfile wired to `/health/ready` so orchestrators only route traffic to healthy instances.
- `notebooks/08_serving_hardening.ipynb`: drives the hardened API offline via `TestClient` + a fake predictor; demonstrates warm-up evidence, structured logs, a 503 readiness failure, and the optional Prometheus metrics endpoint.

## [0.1.0] - 2026-05-05

### Added
- End-to-end Hugging Face text-classification workbench with CLI, evaluation, serving, and tests.
