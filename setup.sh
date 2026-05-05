#!/usr/bin/env bash
set -euo pipefail

poetry install --with dev
mkdir -p data/raw data/processed artifacts/models reports/sample_run

echo "Environment ready. Run: poetry run hf-lab sample-data --output data/raw/support_tickets.csv --rows 2000"
