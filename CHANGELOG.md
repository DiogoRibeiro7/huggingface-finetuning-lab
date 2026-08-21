# Changelog

All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog, and this project follows Semantic Versioning.

## [Unreleased]

### Added

- Hub publication lifecycle: `hf-lab push-to-hub`, `promote-to-hub` and `pull-model`.
  Publication lands on a `staging` branch and promotion points a release revision at that
  already-evaluated commit, so released weights are the reviewed ones — promotion never
  re-uploads. `--dry-run` runs every check and renders the card without contacting the Hub.
- Publication enforces the artifact contracts, because it is the last point they matter: an
  incomplete artifact is refused, and so is a **public** repository carrying raw held-out
  rows. Private repositories are allowed; training scratch is never uploaded.
- Hub-native model card. A repository renders `README.md` as its card and reads the YAML
  block for the task filter, base-model and dataset links, and the metrics table, so the
  lab now writes one with a `model-index` results block. The internal `model_card.md`
  report is unchanged — it describes a run, the repository card describes the published
  model.
- `provenance.json`, written during training: the base model and its resolved commit, the
  dataset id, revision and fingerprint, and the source commit. A repository id is mutable,
  so a run recorded by name alone cannot be reconstructed later. Resolution is best effort
  — a local checkpoint or an offline machine records what is known rather than failing.
- Serving from a pinned Hub revision: `hf-lab serve --model-repo owner/name
  --model-revision <sha>`. The revision is fetched at startup, health reports
  `repo@revision`, and the server warns when handed a branch, which can change under a
  running deployment.
- `docs/hub_lifecycle.md` covering the path from training to a served, pinned revision.

### Fixed

- Hub dataset presets used bare repository ids (`ag_news`, `imdb`, `banking77`,
  `tweet_eval`). Those resolved through Hub redirects until `huggingface-hub` 1.14,
  which rejects them: *Repository id must be 'namespace/name'*. All four are now
  namespaced, with `banking77` pointing at the parquet mirror because the PolyAI copy
  is still script-based and `datasets` >=4 no longer runs dataset scripts.
- `hf-lab train --config-file <file>` rejected every training flag as
  "explicitly passed" when none were. The guard compared a `ParameterSource` from the
  command context against one imported from `click`; typer ships its own copy of click,
  so those can be different enum classes that never compare equal. The comparison is now
  by name, and the package no longer imports click at all.

### Changed

- The serving image installs the CPU build of PyTorch. The default Linux `torch` wheel
  bundles CUDA, so the container carried 17 nvidia/triton packages it could never use:
  **8.66 GB to 2.59 GB**. The lock stays GPU-capable, since `poetry install` is also how a
  training environment is set up.
- Dependency updates: `typer` 0.27.1, `uvicorn` 0.52.3, `huggingface-hub` 1.28.0,
  `ruff` 0.16.3, `httpx` 0.28.1.
- `click` is no longer a direct dependency. Nothing imports it, and typer 0.27 does not
  require it either, so the `<8.2` cap that tied the two together is gone.
- Dependabot no longer groups pre-1.0 packages into the minor/patch pull request. A
  0.12 to 0.27 bump is classified as "minor" but is where breaking changes live for a 0.x
  project — that is how a typer upgrade that broke the CLI arrived inside a green grouped
  PR. Python base images above the supported range are ignored, since nothing in CI builds
  the Dockerfile and such a bump also arrives green.

### Added

- Architecture diagrams in the README and `docs/architecture.md`, replacing an ASCII
  sketch that drew the flow as a straight line and so could not show that the validation
  and test splits play different roles, or that the artifact is read by every downstream
  stage rather than being one step among many.
- A scheduled `Hub presets` workflow resolves every preset against the real Hub. Preset
  breakage is invisible to the rest of the suite, which builds datasets in memory — the
  `huggingface-hub` 1.28 failure above passed CI on both Python versions. The checks are
  marked `network` and excluded from the default run, so merges stay fast.
- Regression coverage for the `--config-file` path, which previously had none, and a test
  asserting every preset id is namespaced.

## [1.0.1] - 2026-08-21

### Fixed

- Hub datasets with non-numeric labels encoded each split independently, so the same
  `label_id` could mean different classes in train and test. One mapping is now built
  across all splits.
- The `splits` filter re-evaluated `set(splits)` per item, silently dropping splits when
  passed a generator.
- The promotion gate treated "nothing failed" as approval, so an empty report — or one
  whose checks were all skipped — authorized promotion. Criteria now carry a severity, and
  promotion requires every required criterion to have been evaluated and passed.
- Inference called the pipeline without the training `max_length`, so it could clip at a
  different boundary than training and evaluation.
- `hf-lab serve` ignored the `HF_LAB_*` variables `docker-compose.yml` declares, leaving
  `HF_LAB_MODEL_VERSION` and `HF_LAB_ENABLE_METRICS` inert.
- Notebook 03 selected a threshold on the test split and reported the result on the same
  rows, which is optimistic by construction.
- `CompatibleTrainer.create_optimizer` did not match the signature it overrides.

### Security

- The readiness endpoint returned `repr(exc)` on startup failure, which can carry local
  paths and configuration values. The detail is now logged and the response is opaque.
- `/predict` bounds per-text size; the batch cap limited how many texts arrived, not how
  large they were.
- The held-out split is recorded as a manifest (size, label distribution, seed,
  fingerprint, one-way row hashes) instead of raw rows, so a shared or published model
  directory no longer carries its evaluation text. Raw rows are opt-in through
  `TrainingConfig.persist_heldout_rows`.
- The container runs as an unprivileged user and no longer ships a compiler in the runtime
  stage.

### Added

- `hf-lab verify-artifact --deep`: parses the JSON, checks weights are non-empty, compares
  the label space with the model config, and loads the tokenizer and model. The layout pass
  accepted placeholder files.
- A train, save, verify, reload and predict regression test covering both the standard and
  LoRA paths, against a locally built tiny transformer so it needs no network.
- `preprocessing.json` in the model artifact, carrying the training-time tokenizer contract.
- `label_noise` and `ambiguity` on the sample generator, and `group_col` on the split
  helper so near-duplicate phrasings cannot span splits. `duplicate_text_report` surfaces
  repeats before splitting.
- `stratify=True` on `bootstrap_metric`, plus input validation across the robust-evaluation
  helpers and a `low_support` flag on subgroup metrics.
- Resolved package versions, accelerator details, the full git commit, and model and
  dataset revisions in the reproducibility record.
- `make check-fast` / `make check` / `make check-full`, and notebook lint in the release
  workflow.

### Changed

- The artifact contract accepts sharded checkpoints and SentencePiece tokenizers.
- Model cards report quality metrics only, not `Trainer` throughput and bookkeeping.
- `EmbeddingIndex` requires unique `doc_id` values, which ranking metrics assume.
- `TrainingConfig.validate` covers `weight_decay`, the seed, the metric name and every LoRA
  setting, so an invalid value fails before the base model is fetched.
- `run_training_pipeline` takes a `TrainingConfig` instead of duplicating a subset of it.
- The Docker image ships no model and mounts one at `/models`; Compose previously claimed
  an example model was baked in, which it never was.
- `prometheus-client` is a declared `metrics` extra rather than a manual install.

## [1.0.0] - 2026-08-21

### Security

- Upgraded the Hugging Face and serving stack to patched releases, clearing 56 Dependabot alerts (1 critical, 22 high): `transformers` `^5.5.0`, `datasets` `^5.0.0`, `torch` `^2.13.0`, `fastapi` `^0.141.0` (pulling in a patched `starlette`), plus `huggingface-hub` `^1.5.0`, `accelerate` `^1.1.0`, `peft` `^0.20.0`, `safetensors` `^0.8.0`, `evaluate` `^0.4.6`, and `pytest` `^9.0.3`. Transitive fixes cover `pillow`, `mistune`, `aiohttp`, `jupyter-server`, `jupyterlab`, and `setuptools`.

### Added

- `poetry.lock`, so dependency resolution is reproducible and CI's Poetry cache has a file to key on. Its absence was failing every CI run at `setup-python`.
- `.github/dependabot.yml`: weekly `pip`, `github-actions`, and `docker` update checks, with minor and patch bumps grouped into a single pull request so majors stay reviewable on their own.
- Public repository governance files: `SECURITY.md`, `.gitattributes`, issue templates, and a pull request template.
- Zenodo release metadata in `.zenodo.json`, with matching citation metadata in `CITATION.cff`.
- Repository professionalization baseline: CI quality gates, contributor workflow, and release automation.
- `hf_finetuning_lab.governance.promotion` module: `PromotionCriterion`, `PromotionReport`, `threshold_criterion` / `boolean_criterion` / `skipped_criterion` helpers, `write_promotion_report` (Markdown verdict + criteria table + JSON sidecar), and `aggregate_reports` for comparison tables.
- `notebooks/10_promotion_gate.ipynb`: composes v0.4 robust-evaluation checks (bootstrap CIs on macro F1, ECE, subgroup F1 ratio, train/test PSI drift), v0.9 governance artifacts (dataset card, model card, reproducibility checklist), and v1.0 artifact verification into a single Markdown + JSON promotion report with an explicit `should_promote` verdict.
- `hf_finetuning_lab.experiments` module: run IDs, dataset hashing, run-record persistence, and run-comparison DataFrame.
- `per_class_report` helper in `hf_finetuning_lab.evaluation.metrics`.
- `notebooks/02_experiment_management.ipynb`: repeated TF-IDF + LogReg runs with persisted records, side-by-side comparison, per-class report, and confusion-matrix heatmap.
- `hf_finetuning_lab.evaluation.robust` module: reliability curves, expected calibration error, bootstrap confidence intervals, threshold optimization, subgroup metrics, and label-share PSI drift.
- `notebooks/03_robust_evaluation.ipynb`: reliability diagram, threshold sweep, bootstrap CIs, subgroup table, and drift visualization on the synthetic support-ticket task.
- `hf_finetuning_lab.data.hub` module: `HubDatasetConfig`, `HUB_PRESETS` (AG News, IMDb, Banking77, TweetEval sentiment), `load_hub_dataset`, `normalize_hub_dataset_dict`, and `write_hub_dataset_csv`.
- CLI commands `hf-lab list-hub-datasets` and `hf-lab fetch-hub-dataset` for downloading Hub presets to local CSV.
- `notebooks/04_hub_datasets.ipynb`: preset registry walkthrough, offline mock-DatasetDict normalization, opt-in real Hub download, and a TF-IDF baseline on the normalized schema.
- `hf_finetuning_lab.token_classification` module: NER schema (`NERExample`, synthetic data generation, JSONL writer, validator), subword alignment (`align_word_labels_to_subwords` with `first` and `all` strategies), and entity-level metrics (`extract_entities`, `sequence_tagging_report`).
- `notebooks/05_token_classification.ipynb`: synthetic CoNLL-style NER, label/entity distribution, subword alignment demo, per-token logistic-regression baseline, and entity-level micro/macro P/R/F1.
- `hf_finetuning_lab.retrieval` module: `EmbeddingIndex` (cosine search over L2-normalised embeddings), `IndexEntry`, `l2_normalize`, plus retrieval metrics `recall_at_k`, `mean_reciprocal_rank`, `ndcg_at_k`, `retrieval_report`.
- `notebooks/06_semantic_search.ipynb`: synthetic FAQ corpus, TF-IDF embedding index, cosine retrieval with Recall@k / MRR / nDCG@k, error inspection, and an opt-in sentence-transformer comparison.
- `hf_finetuning_lab.governance` module: `DatasetCard` / `DatasetColumn` / `DatasetSplit` + `write_dataset_card`, `task_limitations` and `write_task_model_card` for text-classification, token-classification, and retrieval, and `ReproducibilityRecord` + `capture_environment` + `write_reproducibility_checklist` (Markdown + JSON sidecar with environment, seed, dataset hash, and git commit metadata).
- `notebooks/07_governance_template.ipynb`: end-to-end governance walkthrough — trains a small baseline, writes a dataset card with split-level label distributions, a task-specific model card, and a reproducibility checklist tying together run ID, dataset hash, environment snapshot, and metrics.
- `hf_finetuning_lab.serving` deployment hardening: `create_app` now accepts a `predictor_factory` (lazy/injectable predictor), runs model warm-up on startup, and exposes `/health/live` + `/health/ready` (with 503 + diagnostic payload when the predictor cannot load). `StructuredRequestLogger` emits one JSON log line per request; `install_metrics(app)` mounts a Prometheus `/metrics` endpoint when `prometheus-client` is installed.
- `docker-compose.yml` at the repo root plus a `HEALTHCHECK` in the Dockerfile wired to `/health/ready` so orchestrators only route traffic to healthy instances.
- `notebooks/08_serving_hardening.ipynb`: drives the hardened API offline via `TestClient` + a fake predictor; demonstrates warm-up evidence, structured logs, a 503 readiness failure, and the optional Prometheus metrics endpoint.
- `hf_finetuning_lab.artifacts` module: `ArtifactCheck`, `ArtifactReport`, and `verify_artifact(model_dir)` enforcing the stable v1.0 model-artifact layout (`config.json`, weights, tokenizer, plus recommended `tokenizer_config.json` / `special_tokens_map.json` / `model_card.md` / `test_metrics.json`).
- CLI commands `hf-lab version`, `hf-lab list-commands`, and `hf-lab verify-artifact --model-dir <path> [--strict]`.
- `notebooks/09_v1_capstone.ipynb`: enumerates the CLI surface, demonstrates `verify_artifact` on a synthetic artifact, lists the v1.0 module map and the notebook stack, and includes the release checklist.
- `docs/architecture.md` refreshed with the v1.0 module map, the artifact contract, and the notebook stack.

### Changed

- Hardened GitHub Actions with explicit permissions, concurrency, timeouts, release artifact retention, and a separate notebook smoke job.
- `trainer.py`: load classification models with `ignore_mismatched_sizes=True`. Transformers 5 raises on a classifier head whose label count differs from the checkpoint, where 4.x reinitialised it with a warning; the head is always retrained here, so the reinitialisation is now explicit.
- `trainer.py`: `CompatibleTrainer.create_optimizer` now accepts the model positionally and returns the optimizer, matching the Transformers 5 `Trainer` contract it overrides. The previous signature took no model and returned `None`, which would raise `TypeError` against Transformers 5.
- Pinned CI, the release workflow, and the Docker image to Poetry `2.2.1`, and install Poetry before `setup-python` so its dependency cache resolves.
- Expanded package metadata with license, project URLs, Trove classifiers, and Python version support.
- Bumped minimum supported Python to `3.11` (matches actual `datetime.UTC` usage). `pyproject.toml`, ruff `target-version`, mypy `python_version`, and the CI matrix all updated together.
- `trainer.py`: switched `TrainingArguments(evaluation_strategy=...)` to `eval_strategy=`, switched `Trainer(tokenizer=...)` to `processing_class=`, and modernised `isinstance(value, (int, float))` to `isinstance(value, int | float)`.
- `CHANGELOG.md`: added blank lines under each `### Added` / `### Changed` heading per Keep-a-Changelog convention.
- Bumped package version to `1.0.0` (`pyproject.toml` + `hf_finetuning_lab.__version__`).

## [0.1.0] - 2026-05-05

### Added

- End-to-end Hugging Face text-classification workbench with CLI, evaluation, serving, and tests.
