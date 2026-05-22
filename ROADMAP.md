# Roadmap

The v0.1 → v1.0 release roadmap is complete. See `CHANGELOG.md` for the per-version history and the existing notebooks (`notebooks/01_*` through `notebooks/10_*`) for the shipped capabilities.

This document tracks the next phase of work. Milestones are sized to mirror the shape of the v0.x deliverables — a coherent feature, a thin support module under `src/hf_finetuning_lab/`, unit tests that run without a model download, and a notebook that ships with executed outputs. Unchecked items are speculative; check them as scope solidifies.

## v1.1 — GPU CI and real-model smoke

- [ ] Add a self-hosted-GPU GitHub Actions job that runs `hf-lab train` against a tiny base model and asserts the artifact passes `hf-lab verify-artifact --strict`.
- [ ] Document GPU-runner provisioning prerequisites in `docs/`.
- [ ] Add a `gpu` pytest marker and a per-notebook `RUN_GPU=True` opt-in flag so existing notebooks can exercise the real transformer training path when a GPU is present.
- [ ] Pin a reproducible training run that produces a known-good artifact + metrics, and snapshot its `metrics.json` for regression detection.
- Closes the one open caveat from the v1.0 release checklist.

## v1.2 — Active learning loop

- [ ] New module `hf_finetuning_lab.active_learning` with uncertainty sampling (margin, entropy, least confidence) and diversity sampling (k-center, BADGE-lite).
- [ ] CLI command `hf-lab pick-samples --model-dir <path> --pool <jsonl> --k <n> --strategy <name>`.
- [ ] Notebook 11 walks one full loop: train → score the unlabeled pool → select N → simulate human labels → retrain. Compares a "random" vs "active" sampler on a macro-F1-vs-labels trajectory.
- Rationale: closes the data side of the lab — every other notebook assumes labels already exist.

## v1.3 — Drift monitoring service

- [ ] Productionise the v0.4 drift logic: capture a reference snapshot at training time, run scheduled comparison jobs against live predictions.
- [ ] New module `hf_finetuning_lab.monitoring` with `DriftSnapshot`, `compare_snapshots`, alert thresholds, and a JSON report.
- [ ] CLI commands `hf-lab snapshot` (writes a reference snapshot) and `hf-lab compare-drift` (compares two snapshots and exits non-zero on threshold breach so a cron job can page).
- [ ] Notebook 12 demonstrates the loop end-to-end and writes a Markdown drift report.

## v1.4 — Quantization and efficient inference

- [ ] Add Int8 / fp16 quantization paths (`bitsandbytes` for training-aware, `optimum-onnxruntime` for inference) gated behind opt-in flags.
- [ ] Latency-budget notebook (13): compare fp32 / fp16 / int8 / ONNX on the same artifact across batch sizes, plus a per-tier memory table.
- [ ] Promotion gate gains an optional latency-SLO criterion that consumes the benchmark output.

## v1.5 — Sequence-to-sequence and generation

- [ ] Broaden task coverage to summarisation and short-form generation.
- [ ] New module `hf_finetuning_lab.generation` with `Seq2SeqExample`, decoding configuration, and ROUGE / faithfulness metrics.
- [ ] Notebook 14 walks fine-tuning a small T5 / BART variant on a synthetic seq2seq dataset; the existing CLI grows a `--task seq2seq` shape.

## v1.6 — Multilingual coverage

- [ ] Add language-aware presets (XLM-R, mBERT) to `data.hub` with per-language column hints.
- [ ] Extend `evaluation.robust.subgroup_metrics` examples with a language stratification cookbook.
- [ ] Notebook 15 demonstrates a multilingual fine-tune with per-language metric tables and an audit slot in the model card.

## v1.7 — Continuous fine-tuning loop

- [ ] Drift-triggered retraining job composed of v1.3 (snapshots + comparison), v1.2 (active sampling), and the existing training stack.
- [ ] Safe rollback gate keyed on the promotion-gate report from notebook 10.
- [ ] Notebook 16 simulates the full loop on synthetic data over multiple "days" and renders the resulting deployment timeline.

## v2.0 — Distributed training and multi-task heads

- [ ] Multi-node / multi-GPU training documentation and `accelerate` / `deepspeed` launch examples.
- [ ] Multi-task model heads (text classification + NER on a shared encoder) with a single artifact spec covering both label maps.
- [ ] Refreshed stability commitments — same shape as the v1.0 release checklist — covering the broadened surface.

## Always-on backlog

- [ ] Track new transformers / sklearn / datasets API drift each minor release and bump pins together with regression tests.
- [ ] Expand notebook quality gate: `nbqa ruff` on every notebook + a `--check-outputs` step that fails on `output_type=="error"` cells.
- [ ] Improve coverage of the existing modules' edge cases as new bugs surface in real use.
