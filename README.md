# Hugging Face Fine-Tuning Lab

A production-oriented workbench for fine-tuning, evaluating, exporting, and serving Hugging Face text-classification models.

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
