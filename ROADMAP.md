# Roadmap

## v0.1 — Text-classification fine-tuning workbench

- [x] Synthetic support-ticket text classification data.
- [x] CSV/JSONL loading utilities.
- [x] Hugging Face Dataset conversion.
- [x] Tokenization with `AutoTokenizer`.
- [x] Transformer fine-tuning with `Trainer`.
- [x] Optional PEFT/LoRA configuration.
- [x] Evaluation metrics.
- [x] Batch inference.
- [x] FastAPI serving.
- [x] Model-card generation.
- [x] CLI workflow.
- [x] Notebook workflow.
- [x] Tests for non-GPU components.

## v0.2 — Stronger experiment management

- [x] Add explicit run IDs (`hf_finetuning_lab.experiments.make_run_id`).
- [x] Save full training configuration as JSON (`RunRecord.params` persisted via `save_run`).
- [x] Save dataset hashes (`hash_dataframe`).
- [x] Add experiment comparison tables (`runs_to_frame`, surfaced in `notebooks/02_experiment_management.ipynb`).
- [x] Add support for repeated train/eval runs (notebook 02 sweep).
- [x] Add confusion-matrix and per-class error reports (`per_class_report`, `confusion_matrix_frame`).

## v0.3 — Real Hugging Face datasets

- [x] Add Hub dataset CLI commands (`list-hub-datasets`, `fetch-hub-dataset`).
- [x] Add split mapping for Hub datasets (`HubDatasetConfig.splits`).
- [x] Add label-name normalization (uses `ClassLabel.names` with explicit preset override).
- [x] Add support for AG News, IMDb, Banking77, and TweetEval sentiment via `HUB_PRESETS`.
- All demoed in `notebooks/04_hub_datasets.ipynb` (offline-by-default with opt-in real download).

## v0.4 — Robust evaluation

- [x] Add calibration metrics (`expected_calibration_error`, `reliability_curve`).
- [x] Add threshold optimization (`find_best_threshold`).
- [x] Add subgroup metrics (`subgroup_metrics`).
- [x] Add prediction drift comparison between two datasets (`prediction_share_drift`, PSI).
- [x] Add bootstrap confidence intervals (`bootstrap_metric`).
- All demoed in `notebooks/03_robust_evaluation.ipynb`.

## v0.5 — Deployment hardening

- [x] Add request logging (`StructuredRequestLogger` middleware emitting one JSON line per request).
- [x] Add health and readiness checks (`/health/live`, `/health/ready` with diagnostic payload when the predictor fails to load).
- [x] Add model warm-up (`warm_up_texts` argument on `create_app`).
- [x] Add Docker Compose serving example (`docker-compose.yml` + Dockerfile `HEALTHCHECK` wired to `/health/ready`).
- [x] Add optional Prometheus metrics (`enable_metrics=True` mounts `/metrics` via `install_metrics`, requires `prometheus-client`).
- All demoed in `notebooks/08_serving_hardening.ipynb`.

## v0.6 — PEFT expansion

- Add LoRA target-module presets by architecture.
- Add adapter merge/export utilities.
- Add quantized loading path where available.
- Add adapter comparison reports.

## v0.7 — Sequence and token classification

- [x] Add NER data schema (`NERExample`, `generate_synthetic_ner_data`, `write_synthetic_ner_jsonl`).
- [x] Add subword alignment helper (`align_word_labels_to_subwords`) so the same labels work with any HF fast tokenizer.
- [x] Add sequence tagging metrics (`extract_entities`, `sequence_tagging_report` — entity-level micro/macro P/R/F1).
- All demoed in `notebooks/05_token_classification.ipynb` (offline synthetic NER + per-token logistic-regression baseline).
- Transformer fine-tuning path still TODO — keep it out of CI smoke; integrate as an opt-in cell when needed.

## v0.8 — Retrieval and embeddings

- [x] Add `EmbeddingIndex` for cosine-similarity search over any L2-normalisable matrix.
- [x] Add retrieval metrics (`recall_at_k`, `mean_reciprocal_rank`, `ndcg_at_k`, `retrieval_report`).
- [x] Add semantic search example (`notebooks/06_semantic_search.ipynb`) with a synthetic FAQ corpus, TF-IDF embeddings for offline smoke, and an opt-in sentence-transformer path.

## v0.9 — Reproducibility and model governance

- [x] Add model-card templates by task (`write_task_model_card`, `task_limitations` for text-classification / token-classification / retrieval).
- [x] Add dataset cards (`DatasetCard`, `DatasetColumn`, `DatasetSplit`, `write_dataset_card`).
- [x] Add risk and limitation sections (curated bullets per task, with project-specific extras).
- [x] Add reproducibility checklist (`ReproducibilityRecord`, `capture_environment`, `write_reproducibility_checklist` — Markdown + JSON sidecar).
- All demoed in `notebooks/07_governance_template.ipynb`.

## v1.0 — Stable Hugging Face production template

- [x] Stable CLI surface (`hf-lab list-commands`, `hf-lab version`).
- [x] Stable model artifact format (`hf_finetuning_lab.artifacts` + `hf-lab verify-artifact --strict`).
- [x] Tested CPU workflow (full pytest + nine-notebook smoke run in CI). GPU paths are documented but exercised manually until a GPU CI job lands.
- [x] Documentation refresh (`docs/architecture.md` lists the v1.0 module map, artifact contract, and notebook stack).
- [x] Version bumped to `1.0.0` in `pyproject.toml` and `hf_finetuning_lab.__version__`.
- All demoed in `notebooks/09_v1_capstone.ipynb`.
