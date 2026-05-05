.PHONY: install test lint format sample run

install:
	poetry install --with dev

test:
	poetry run pytest -q

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

sample:
	poetry run hf-lab sample-data --output data/raw/support_tickets.csv --rows 500

run:
	poetry run hf-lab train --input data/raw/support_tickets.csv --output-dir artifacts/models/support-triage --model-name distilbert-base-uncased --epochs 1 --batch-size 8
