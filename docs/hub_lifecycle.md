# Publishing to the Hub

A model artifact becomes something other people run when it is published, so
publication is the last point at which the artifact contracts can be enforced —
and the point where names have to become commits.

## The path

```text
train  ->  artifact  ->  push-to-hub (staging)  ->  evaluate  ->  promote-to-hub (release)
                                                                        |
                                                          serve --model-repo @ revision
```

## Publish

```bash
hf-lab push-to-hub \
  --model-dir artifacts/models/support-triage \
  --repo-id me/support-triage \
  --license mit \
  --dataset fancyzhx/ag_news \
  --dry-run
```

`--dry-run` runs every check and renders the card that would be published,
without contacting the Hub. It is the same code path as a real publication, so a
rehearsal means something.

Publication lands on the `staging` branch by default. Releasing is a separate,
deliberate step.

### What is refused

| Refusal | Why |
| --- | --- |
| Incomplete artifact | An incomplete directory on the Hub is worse than none. |
| Raw held-out rows, public repository | Training keeps evaluation text out of the artifact by default; a public repository is the case that motivated it. Publish privately, or retrain without `persist_heldout_rows`. |

Training scratch (`trainer/`, `checkpoint-*/`) is excluded from every upload.

## Promote

```bash
hf-lab promote-to-hub --repo-id me/support-triage --from staging --to v1
```

Promotion points a release revision at a commit that is already published and
already evaluated. It never re-uploads, so the released weights are provably the
reviewed ones.

## Serve a published model

```bash
hf-lab serve --model-repo me/support-triage --model-revision 9f8e7d6c5b4a
```

The revision is fetched at startup, so a failure surfaces then rather than on the
first request. Health reports `repo@revision` when no explicit model version is
set, so what is serving is visible without inspecting the container.

Pin a commit sha or tag. A branch resolves to whatever it points at today, which
means weights can change under a running deployment — the server warns when it is
given one.

## What the card carries

The published `README.md` is the Hub-native model card: the YAML block drives the
task filter, the base-model and dataset links, and the metrics table, and the body
records provenance.

Provenance is resolved at training time into `provenance.json` — the base model
and its commit, the dataset and its revision and fingerprint, and the source
commit. A repository id is mutable, so recording only the name means a run cannot
be reconstructed later.

Resolution is best effort: a local checkpoint, an offline machine or a private
repository records what is known and leaves the rest unset. A run does not fail
because a metadata lookup did not answer.

## Reconstructing a run

From a published model, its card and `provenance.json`:

| Question | Recorded in |
| --- | --- |
| Which code? | `source_commit` |
| Which dependencies? | reproducibility record (`capture_environment`) |
| Which base model? | `base_model` + `base_model_revision` |
| Which data? | `dataset_id` + `dataset_revision` + `dataset_fingerprint` |
| Which split? | `heldout_manifest.json` (seed, sizes, row hashes) |
| Which preprocessing? | `preprocessing.json` |
| Which settings? | `training_config.json` |
| What did it score? | `test_metrics.json`, and the card's metrics table |
