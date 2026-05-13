.PHONY: install test lint typecheck check format sample run precommit release-check branch-protect

install:
	poetry install --with dev

test:
	poetry run pytest

lint:
	poetry run ruff check .

typecheck:
	poetry run mypy src

check: lint typecheck test

format:
	poetry run ruff format .

precommit:
	poetry run pre-commit run --all-files

release-check: check
	poetry build

branch-protect:
	pwsh -File scripts/setup_branch_protection.ps1 -Repo $(REPO) -Branch $${BRANCH:-main}

sample:
	poetry run hf-lab sample-data --output data/raw/support_tickets.csv --rows 500

run:
	poetry run hf-lab train --input data/raw/support_tickets.csv --output-dir artifacts/models/support-triage --model-name distilbert-base-uncased --epochs 1 --batch-size 8
