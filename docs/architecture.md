# Architecture

## High-level flow

```text
Data Source (CSV/JSONL or HF Hub)
  -> Data IO + Validation
  -> Label Mapping + Splits
  -> Tokenization
  -> Trainer (full fine-tune or LoRA)
  -> Saved Artifacts (model, tokenizer, config)
  -> Evaluation + Model Card
  -> Batch Inference and FastAPI Serving
```

## Module map

- `hf_finetuning_lab.cli`: Stable CLI entrypoints.
- `hf_finetuning_lab.data`: Data loading, schema checks, split logic.
- `hf_finetuning_lab.tokenization`: Tokenizer setup and preprocessing.
- `hf_finetuning_lab.training`: Transformer/LoRA training orchestration.
- `hf_finetuning_lab.evaluation`: Metrics and evaluation report generation.
- `hf_finetuning_lab.model_cards`: Responsible model card generation with limitations.
- `hf_finetuning_lab.inference`: Batch prediction from local artifacts.
- `hf_finetuning_lab.serving`: FastAPI app bootstrap and runtime prediction endpoints.

## Runtime boundaries

- Unit tests use synthetic/offline data and avoid hidden network downloads.
- Heavy model training is supported but not required in CI.
- Hugging Face dependencies are imported close to runtime paths where practical.
