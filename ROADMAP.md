# Roadmap

The v0.1 → v1.1 releases are complete. See `CHANGELOG.md` for the per-version history and
the notebooks (`notebooks/01_*` through `notebooks/10_*`) for the shipped capabilities.

This document tracks the next phase. Milestones are sized to mirror the shape of the
earlier deliverables — a coherent feature, a thin support module under
`src/hf_finetuning_lab/`, unit tests that run without a model download, and a notebook
that ships with executed outputs. Unchecked items are speculative; check them as scope
solidifies.

## Shipped

### v1.0.1 — contracts and correctness

Closed the gaps between components rather than adding surface: one label mapping across
Hub splits, a fail-closed promotion gate, the training preprocessing contract applied at
inference, serving configuration actually read from the environment, and the held-out
split described by a manifest instead of shipping its raw rows.

### v1.1 — Hub publication and provenance

`hf-lab push-to-hub`, `promote-to-hub` and `pull-model`; a Hub-native `README.md` card
with `model-index` metadata; `provenance.json` resolving base-model and dataset names to
the commits behind them; and serving from a pinned revision. Publication enforces the
artifact contracts at the point they matter — an incomplete artifact, and a public
repository carrying raw held-out rows, are both refused.

## v1.2 — GPU CI and real-model regression

`tests/test_training_lifecycle.py` already covers train → save → verify → reload → predict
against a locally built tiny transformer, on CPU and with no network. The remaining gap is
a *real* model on real hardware, which nothing currently exercises.

- [ ] Self-hosted-GPU GitHub Actions job running `hf-lab train` against a small real base
      model, asserting the artifact passes `hf-lab verify-artifact --deep --strict`.
- [ ] Document GPU-runner provisioning prerequisites in `docs/`.
- [ ] A `gpu` pytest marker alongside the existing `network` marker, and a per-notebook
      `RUN_GPU=True` opt-in so notebooks can exercise the real training path when a GPU is
      present.
- [ ] Pin a reproducible run producing a known-good artifact, and snapshot its
      `test_metrics.json` for regression detection.
- Rationale: the last remaining path where a dependency bump can break the stack without
  any check noticing. The `network`-marked Hub preset job is the model to copy — scheduled
  rather than per-PR, because it needs hardware CI cannot guarantee.

## v1.3 — Active learning loop

- [ ] New module `hf_finetuning_lab.active_learning` with uncertainty sampling (margin,
      entropy, least confidence) and diversity sampling (k-center, BADGE-lite).
- [ ] CLI command `hf-lab pick-samples --model-dir <path> --pool <jsonl> --k <n>
      --strategy <name>`.
- [ ] Notebook 11 walks one full loop: train → score the unlabeled pool → select N →
      simulate human labels → retrain. Compares a random against an active sampler on a
      macro-F1-vs-labels trajectory.
- Rationale: closes the data side of the lab — every other notebook assumes labels already
  exist.

## v1.4 — Drift monitoring service

- [ ] Productionise the v0.4 drift logic: capture a reference snapshot at training time,
      run scheduled comparison jobs against live predictions.
- [ ] New module `hf_finetuning_lab.monitoring` with `DriftSnapshot`, `compare_snapshots`,
      alert thresholds, and a JSON report.
- [ ] CLI commands `hf-lab snapshot` and `hf-lab compare-drift`, the latter exiting
      non-zero on a threshold breach so a scheduled job can page.
- [ ] Notebook 12 demonstrates the loop end to end and writes a Markdown drift report.
- Note: a snapshot should record the serving revision, so a drift report says which
  published weights produced the predictions.

## v1.5 — Quantization and efficient inference

- [ ] Int8 / fp16 quantization paths (`bitsandbytes` for training-aware,
      `optimum-onnxruntime` for inference) behind opt-in flags.
- [ ] Latency-budget notebook (13): fp32 / fp16 / int8 / ONNX on the same artifact across
      batch sizes, plus a per-tier memory table.
- [ ] Promotion gate gains an optional latency-SLO criterion consuming the benchmark
      output.
- Note: the serving image already installs CPU-only PyTorch. A GPU serving variant belongs
  here rather than being bolted onto the CPU image.

## v1.6 — Sequence-to-sequence and generation

- [ ] Broaden task coverage to summarisation and short-form generation.
- [ ] New module `hf_finetuning_lab.generation` with `Seq2SeqExample`, decoding
      configuration, and ROUGE / faithfulness metrics.
- [ ] Notebook 14 fine-tunes a small T5 / BART variant on a synthetic seq2seq dataset; the
      CLI grows a `--task seq2seq` shape.
- Note: `pipeline_tag` in the Hub card is currently hard-coded to text classification and
  will need to follow the task.

## v1.7 — Multilingual coverage

- [ ] Language-aware presets (XLM-R, mBERT) in `data.hub` with per-language column hints.
- [ ] Extend `evaluation.robust.subgroup_metrics` examples with a language stratification
      cookbook — `min_support` matters most where some languages are thinly represented.
- [ ] Notebook 15 demonstrates a multilingual fine-tune with per-language metric tables and
      an audit slot in the model card.

## v1.8 — Continuous fine-tuning loop

- [ ] Drift-triggered retraining composed of v1.4 (snapshots), v1.3 (active sampling) and
      the existing training stack.
- [ ] Safe rollback keyed on the promotion-gate report, promoting a previous revision
      rather than retraining — the staging/release revision workflow already supports it.
- [ ] Notebook 16 simulates the full loop on synthetic data over several "days" and renders
      the resulting deployment timeline.

## v2.0 — Distributed training and multi-task heads

- [ ] Multi-node / multi-GPU documentation and `accelerate` / `deepspeed` launch examples.
- [ ] Multi-task heads (text classification + NER on a shared encoder) with a single
      artifact spec covering both label maps.
- [ ] Refreshed stability commitments covering the broadened surface.

## Always-on backlog

- [ ] Improve coverage of existing modules' edge cases as bugs surface in real use.
- [ ] Extend the notebook quality gate with a `--check-outputs` step failing on
      `output_type == "error"` cells.
- [ ] Keep dependency drift visible. Three breakages this cycle — a typer upgrade, a
      uvicorn bump and `huggingface-hub` rejecting bare repository ids — all passed CI
      because they sat on paths no test exercised. Two now have permanent coverage; when a
      bump breaks something, the durable fix is a test on that path, not just the version
      pin.
