# Methodology

This project implements a transformer fine-tuning workflow for text classification, plus
the evaluation and governance contracts that decide whether a trained model may ship.

## The rule that shapes everything else

**The test split is never used to choose anything.** Not a threshold, not a calibration
parameter, not a model, not a promotion policy.

```text
train       -> fit the model
validation  -> choose thresholds, calibration, any policy
test        -> score the frozen result, once
```

Selecting on a split makes that split optimistic: you kept whatever happened to work best
on those particular rows. Notebook 03 demonstrates the size of the effect by reporting the
validation-minus-test gap for a tuned threshold.

## Data

Input data needs a text column and a label column. `validate_text_classification_frame`
rejects frames that are missing them, are empty, or carry null labels.

The sample generator produces synthetic support tickets across account, billing,
technical, delivery, cancellation and security. It also takes `label_noise` and
`ambiguity`: with one distinct template per class the task is separable, every metric
lands on 1.0, and calibration and drift tooling has nothing to measure.

### Splitting

`stratified_train_valid_test_split` stratifies by label. When rows are **not independent** —
repeated customers, threads, documents, or near-duplicate phrasings — pass `group_col` so
whole groups stay within one split. A near-copy of a training row sitting in test reports a
score the model has not earned. `duplicate_text_report` surfaces repeats before splitting.

## Preprocessing

Text is tokenized with the tokenizer belonging to the selected model, truncated to
`max_length`. Padding is dynamic, applied per batch by the collator, rather than padding
everything to `max_length`.

The settings are written to `preprocessing.json` in the model directory and read back by
the predictor, so inference clips at the same boundary training did.

## Fine-tuning

Hugging Face `Trainer` with `AutoModelForSequenceClassification`. The classification head
is reinitialised for this dataset's label space, so a run can start from an existing
classifier checkpoint with a different label count.

LoRA is optional, via PEFT adapters over a smaller parameter set. The adapter is **merged
into the base weights** before saving, so the artifact is a standalone classifier rather
than an adapter that needs `peft` and the original base model at inference.

## Evaluation

Headline metrics: accuracy, precision, recall, F1, binary ROC AUC where defined, a
confusion matrix and a per-class report. The persisted label mapping is the source of
truth, so evaluation indices match training regardless of which labels appear in a
particular evaluation file.

Beyond the headline numbers:

- **Calibration** — reliability curve and expected calibration error. The ECE here is
  *top-confidence* ECE: rows are binned by the confidence of the predicted class. It is not
  positive-class calibration.
- **Confidence intervals** — percentile bootstrap. The default is an ordinary IID
  bootstrap, which assumes independent rows and understates uncertainty on grouped data.
  `stratify=True` resamples within each class, which matters on small or imbalanced samples
  where a replicate can otherwise drop a class entirely.
- **Subgroup metrics** — per-group scores with a `low_support` flag, because a metric over
  a handful of rows is not comparable with one over thousands.
- **Drift** — prediction-share PSI between two prediction sets.

## Artifacts

A model directory is expected to be self-describing: weights, tokenizer, label mapping,
training config, preprocessing contract, metrics and a model card. `verify_artifact`
checks the layout; `--deep` additionally parses the JSON, compares the label space with the
model config, and loads the tokenizer and model.

The held-out split is recorded as a **manifest** — size, label distribution, seed,
fingerprint and one-way row hashes — not as raw rows. Model directories get copied and
published, and the evaluation text should not travel with them. Raw rows are opt-in via
`TrainingConfig.persist_heldout_rows`.

Model cards report quality metrics only. `Trainer.evaluate()` also returns throughput and
bookkeeping entries, and `eval_samples_per_second` next to macro F1 reads as a result.

## Promotion

A promotion gate fails **closed**. Criteria are `required` or `advisory`; promotion needs
at least one required criterion, every required criterion evaluated, and all of them
passing. An empty report, or one whose required checks were skipped, blocks — a check that
did not run produced no evidence.

## Serving

FastAPI with liveness and readiness probes, startup model loading, warm-up, structured
request logs, bounded request size, and optional Prometheus metrics. See
[deployment.md](deployment.md).
