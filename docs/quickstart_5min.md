# Quickstart (5 Minutes)

This quickstart is for first-time local validation on CPU.

## 1. Environment

Use Python 3.12 (recommended) and Poetry.

```bash
python --version
poetry --version
```

If you use `pyenv`, this repo includes `.python-version` set to `3.12`.

## 2. Install dependencies

```bash
poetry env use 3.12
poetry install --with dev
```

## 3. Generate synthetic sample data

```bash
poetry run hf-lab sample-data --output data/raw/support_tickets.csv --rows 500
```

## 4. Train a small baseline

```bash
poetry run hf-lab train \
  --input data/raw/support_tickets.csv \
  --output-dir artifacts/models/quickstart \
  --model-name distilbert-base-uncased \
  --epochs 1 \
  --batch-size 8
```

## 5. Run evaluation and tests

```bash
poetry run hf-lab evaluate \
  --model-dir artifacts/models/quickstart \
  --input data/raw/support_tickets.csv \
  --output reports/sample_run/evaluation.json

make check
```

## Notes

- Sample labels are synthetic and must not be treated as real operational taxonomy.
- For real datasets, validate privacy, quality, subgroup performance, and limitations before production use.
