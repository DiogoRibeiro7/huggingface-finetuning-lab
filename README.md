# Hugging Face Fine-Tuning Lab

Production-focused Hugging Face NLP fine-tuning workbench with reproducible training, robust evaluation, PEFT/LoRA support, batch inference, FastAPI serving, and model-card/report generation.

## Suggested GitHub About and Topics

Use the following in repository settings for stronger discoverability.

- About (repo description): `Production-focused Hugging Face NLP fine-tuning lab with reproducible training, evaluation, PEFT/LoRA, serving, and model cards.`
- Website (optional): your docs landing page or portfolio link.
- Topics:
  - `hugging-face`
  - `transformers`
  - `nlp`
  - `text-classification`
  - `fine-tuning`
  - `peft`
  - `lora`
  - `model-evaluation`
  - `fastapi`
  - `mlops`
  - `python`
  - `pytorch`

The project focuses on a practical end-to-end workflow:

1. Generate or load text-classification data.
2. Convert CSV/JSONL files into Hugging Face `Dataset` objects.
3. Tokenize text with `AutoTokenizer`.
4. Fine-tune a transformer with the Hugging Face `Trainer` API.
5. Optionally run PEFT/LoRA fine-tuning.
6. Evaluate model quality with classification metrics.
7. Export predictions, model cards, and reports.
8. Run batch inference.
9. Serve the model with FastAPI.

The default example uses a synthetic customer-support triage dataset. It is intentionally small so the full workflow can be tested locally before switching to a real Hugging Face Hub dataset or internal CSV export.

## Repository structure

```text
huggingface-finetuning-lab/
├── configs/
├── data/
├── docs/
├── notebooks/
├── prompts/
├── reports/
├── src/hf_finetuning_lab/
└── tests/
```

## Install

```bash
poetry install --with dev
```

For CPU-only local work this is enough. GPU execution depends on your local PyTorch/CUDA environment.

## Quality checks

Run the same checks used in CI:

```bash
make check
```

Or run tools directly:

```bash
poetry run ruff check .
poetry run mypy src
poetry run pytest
```

## Contributing

See `CONTRIBUTING.md` for development workflow, test constraints, and responsible AI contribution rules.

## Quickstart and architecture

- 5-minute setup: `docs/quickstart_5min.md`
- System architecture: `docs/architecture.md`
- Release/versioning policy: `docs/release_process.md`
- Changelog: `CHANGELOG.md`

## Notebook development

The repo ships seven notebooks under `notebooks/`:

1. `01_hf_text_classification_workflow.ipynb` — end-to-end workflow with a scikit-learn baseline and opt-in Hugging Face CLI commands.
2. `02_experiment_management.ipynb` — repeated runs with explicit run IDs, persisted training configs, dataset hashes, side-by-side comparison, per-class report, and confusion-matrix heatmap.
3. `03_robust_evaluation.ipynb` — calibration (reliability + ECE), threshold tuning, bootstrap confidence intervals, subgroup metrics, and prediction drift (PSI).
4. `04_hub_datasets.ipynb` — Hugging Face Hub presets (AG News, IMDb, Banking77, TweetEval sentiment) with split mapping, label normalization, and an offline mock for CI smoke (`RUN_HUB_DOWNLOAD=True` to fetch real data).
5. `05_token_classification.ipynb` — synthetic CoNLL-style NER, subword alignment, a per-token logistic-regression baseline, and entity-level BIO-span precision / recall / F1.
6. `06_semantic_search.ipynb` — TF-IDF embedding index over a synthetic FAQ corpus with cosine retrieval, Recall@k / MRR / nDCG@k metrics, and an opt-in sentence-transformer path (`RUN_SENTENCE_TRANSFORMER=True`).
7. `07_governance_template.ipynb` — dataset card, task-specific model card (with curated v0.4 / v0.7 / v0.8 limitations), and reproducibility checklist for one training run.

Run notebook quality checks locally:

```bash
make notebook-lint
make notebook-smoke
```

## Branch protection baseline

Apply branch protection (review + required checks) with:

```bash
pwsh -File scripts/setup_branch_protection.ps1 -Repo <owner/repo> -Branch main
```

This requires GitHub CLI (`gh`) authentication with repository admin permissions.

## Python and Poetry stability

This repository is pinned to Python `3.12` for local consistency (`.python-version`).
If Poetry selects an unsupported interpreter, run:

```bash
poetry env use 3.12
```

## Generate sample data

```bash
poetry run hf-lab sample-data \
  --output data/raw/support_tickets.csv \
  --rows 2000
```

## Train a transformer model

```bash
poetry run hf-lab train \
  --input data/raw/support_tickets.csv \
  --output-dir artifacts/models/support-triage \
  --model-name distilbert-base-uncased \
  --text-col text \
  --label-col label \
  --epochs 2 \
  --batch-size 16
```

## Train with LoRA

```bash
poetry run hf-lab train \
  --input data/raw/support_tickets.csv \
  --output-dir artifacts/models/support-triage-lora \
  --model-name distilbert-base-uncased \
  --text-col text \
  --label-col label \
  --epochs 2 \
  --batch-size 16 \
  --use-lora
```

## Evaluate

```bash
poetry run hf-lab evaluate \
  --model-dir artifacts/models/support-triage \
  --input data/raw/support_tickets.csv \
  --output reports/sample_run/evaluation.json
```

## Predict

```bash
poetry run hf-lab predict \
  --model-dir artifacts/models/support-triage \
  --input data/raw/support_tickets.csv \
  --output reports/sample_run/predictions.csv
```

## Serve

```bash
poetry run hf-lab serve \
  --model-dir artifacts/models/support-triage \
  --host 0.0.0.0 \
  --port 8000
```

Then call:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I cannot access my account and my payment failed"]}'
```

## Supported tasks in v0.1

- Text classification.
- Local CSV and JSONL datasets.
- Optional Hugging Face Hub dataset loading.
- Full fine-tuning.
- Optional PEFT/LoRA path.
- Batch inference.
- FastAPI serving.
- Model cards and evaluation reports.

## Responsible use

This repository is a technical template. The sample labels are synthetic and should not be treated as a real operational taxonomy. For real datasets, review privacy, label quality, subgroup performance, calibration, and failure modes before using a model in production.
