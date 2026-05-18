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

- Add request logging.
- Add health and readiness checks.
- Add model warm-up.
- Add Docker Compose serving example.
- Add optional Prometheus metrics.

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

- Add sentence-transformer embedding generation.
- Add semantic search example.
- Add evaluation for retrieval tasks.

## v0.9 — Reproducibility and model governance

- Add model-card templates by task.
- Add dataset cards.
- Add risk and limitation sections.
- Add reproducibility checklist.

## v1.0 — Stable Hugging Face production template

- Stable CLI.
- Stable model artifact format.
- Tested CPU and GPU workflows.
- Documentation for fine-tuning, inference, serving, and governance.
