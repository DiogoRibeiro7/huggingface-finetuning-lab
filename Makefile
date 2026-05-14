.PHONY: install test lint typecheck check format sample run precommit release-check branch-protect notebook-lint notebook-smoke

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

notebook-lint:
	poetry run nbqa ruff check notebooks

notebook-smoke:
	poetry run jupyter nbconvert --to notebook --execute notebooks/01_hf_text_classification_workflow.ipynb --output 01_hf_text_classification_workflow.smoke.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=600
	poetry run jupyter nbconvert --to notebook --execute notebooks/02_experiment_management.ipynb --output 02_experiment_management.smoke.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=600
	poetry run jupyter nbconvert --to notebook --execute notebooks/03_robust_evaluation.ipynb --output 03_robust_evaluation.smoke.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=600

release-check: check
	poetry build

branch-protect:
	pwsh -File scripts/setup_branch_protection.ps1 -Repo $(REPO) -Branch $${BRANCH:-main}

sample:
	poetry run hf-lab sample-data --output data/raw/support_tickets.csv --rows 500

run:
	poetry run hf-lab train --input data/raw/support_tickets.csv --output-dir artifacts/models/support-triage --model-name distilbert-base-uncased --epochs 1 --batch-size 8
