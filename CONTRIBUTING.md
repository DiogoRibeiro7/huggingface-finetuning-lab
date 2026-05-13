# Contributing

Thanks for contributing to `huggingface-finetuning-lab`.

## Development setup

1. Install Python 3.10, 3.11, or 3.12.
2. Install Poetry.
3. Run:

```bash
poetry install --with dev
```

## Local quality gates

Run all checks before opening a PR:

```bash
poetry run ruff check .
poetry run ruff format .
poetry run mypy src
poetry run pytest -q
```

Optional pre-commit setup:

```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

## Testing constraints

- Keep unit tests offline by default.
- Use synthetic datasets for repeatable tests.
- Do not require model downloads in CI.
- Add tests for data loading, label mapping, metrics, model cards, and CLI smoke paths.

## Responsible AI requirements

- Do not represent synthetic labels as real labels.
- Do not imply clinical, legal, or safety validity.
- Include limitations for every generated model card.

## Pull request expectations

- Small, focused changes with clear commit messages.
- Update docs and tests when behavior changes.
- Keep CLI command names and flags stable unless a breaking change is explicitly documented.
