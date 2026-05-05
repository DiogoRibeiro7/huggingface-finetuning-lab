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

- Add explicit run IDs.
- Save full training configuration as JSON.
- Save dataset hashes.
- Add experiment comparison tables.
- Add support for repeated train/eval runs.
- Add confusion-matrix and per-class error reports.

## v0.3 — Real Hugging Face datasets

- Add `--hf-dataset-name` command path.
- Add split mapping for Hub datasets.
- Add label-name normalization.
- Add support for common datasets such as AG News, IMDb, Yelp, Banking77, and TweetEval.

## v0.4 — Robust evaluation

- Add calibration metrics.
- Add threshold optimization.
- Add subgroup metrics.
- Add prediction drift comparison between two datasets.
- Add bootstrap confidence intervals.

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

- Add NER data schema.
- Add token classification fine-tuning.
- Add sequence tagging metrics.

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
