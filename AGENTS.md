# Agent Instructions

This repository is a Hugging Face fine-tuning workbench.

## Coding style

- Use typed Python.
- Keep Hugging Face imports local to functions where possible, so lightweight tests can run without loading models.
- Avoid hidden network calls in tests.
- Provide clear error messages when optional dependencies are missing.
- Add docstrings to public functions.
- Keep CLI commands stable.

## Testing rules

- Unit tests should not download models from the Hub.
- Use synthetic data for tests.
- Test data loading, label mapping, metrics, model-card writing, and CLI smoke paths.
- Heavy training should be documented but not required in CI.

## Responsible AI rules

- Do not present synthetic labels as real labels.
- Do not claim clinical, legal, or safety validity without domain validation.
- Always include limitations in generated model cards.
