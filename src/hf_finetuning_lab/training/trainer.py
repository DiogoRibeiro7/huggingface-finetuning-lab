from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hf_finetuning_lab.config import TrainingConfig
from hf_finetuning_lab.data.io import (
    build_label_mapping,
    encode_labels,
    load_table,
    to_hf_dataset,
    validate_text_classification_frame,
)
from hf_finetuning_lab.data.splits import stratified_train_valid_test_split
from hf_finetuning_lab.evaluation.metrics import trainer_compute_metrics
from hf_finetuning_lab.model_cards.model_card import write_model_card
from hf_finetuning_lab.tokenization.tokenizer import load_tokenizer, tokenize_dataset


def _build_model(
    model_name: str,
    num_labels: int,
    id2label: dict[int, str],
    label2id: dict[str, int],
):
    """Load a sequence-classification model."""
    try:
        from transformers import AutoModelForSequenceClassification
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install `transformers` to fine-tune models.") from exc

    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    )


def _maybe_apply_lora(model: Any, config: TrainingConfig) -> Any:
    """Apply PEFT/LoRA if requested."""
    if not config.use_lora:
        return model
    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install `peft` to use LoRA fine-tuning.") from exc

    target_modules = config.lora_target_modules or None
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=target_modules,
    )
    return get_peft_model(model, lora_config)


def _wrap_missing_optimizer_mode(
    optimizer_wrapper: Any,
    method_name: str,
) -> None:
    """Install a no-op mode method when the wrapped optimizer lacks one.

    Newer Transformers/Accelerate stacks may call ``optimizer.train()`` /
    ``optimizer.eval()`` on the accelerator wrapper even when the wrapped
    PyTorch optimizer is a plain ``torch.optim.AdamW`` without those methods.
    """
    wrapped = getattr(optimizer_wrapper, "optimizer", None)
    shim_flag = f"_hf_lab_{method_name}_shim"
    if wrapped is None or hasattr(wrapped, method_name) or hasattr(optimizer_wrapper, shim_flag):
        return

    def _noop() -> None:
        return None

    setattr(optimizer_wrapper, method_name, _noop)
    setattr(optimizer_wrapper, shim_flag, True)


def _ensure_optimizer_mode_compatibility(optimizer_wrapper: Any) -> None:
    """Patch accelerator optimizer wrappers to tolerate plain PyTorch optimizers."""
    _wrap_missing_optimizer_mode(optimizer_wrapper, "train")
    _wrap_missing_optimizer_mode(optimizer_wrapper, "eval")


class CompatibleTrainer:
    """Mixin that patches optimizer mode hooks for broader HF stack compatibility."""

    def create_optimizer(self) -> None:
        super().create_optimizer()  # type: ignore[misc]
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            _ensure_optimizer_mode_compatibility(optimizer)

    def training_step(self, model: Any, inputs: dict[str, Any], num_items_in_batch: int | None = None):
        optimizer = getattr(self, "optimizer", None)
        if optimizer is not None:
            _ensure_optimizer_mode_compatibility(optimizer)
        return super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)  # type: ignore[misc]


def _write_heldout_test_split(test_df: Any, output_path: Path) -> Path:
    """Persist the held-out test split used for final evaluation."""
    destination = output_path / "heldout_test.csv"
    test_df.to_csv(destination, index=False)
    return destination


def train_text_classifier(input_path: str | Path, output_dir: str | Path, config: TrainingConfig) -> Path:
    """Fine-tune a Hugging Face text-classification model from a local dataset."""
    try:
        from transformers import DataCollatorWithPadding, Trainer, TrainingArguments
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install `transformers` to run training.") from exc

    config.validate()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_table(input_path)
    validate_text_classification_frame(df, text_col=config.text_col, label_col=config.label_col)
    label2id, id2label = build_label_mapping(df[config.label_col])
    encoded = encode_labels(df, config.label_col, label2id)

    train_df, valid_df, test_df = stratified_train_valid_test_split(
        encoded,
        label_col="label_id",
        test_size=config.test_size,
        validation_size=config.validation_size,
        seed=config.seed,
    )

    tokenizer = load_tokenizer(config.model_name)
    train_ds = tokenize_dataset(to_hf_dataset(train_df), tokenizer, config.text_col, config.max_length)
    valid_ds = tokenize_dataset(to_hf_dataset(valid_df), tokenizer, config.text_col, config.max_length)
    test_ds = tokenize_dataset(to_hf_dataset(test_df), tokenizer, config.text_col, config.max_length)

    # Trainer expects the supervised target column to be named `labels`.
    columns_to_remove = [col for col in train_ds.column_names if col not in {"input_ids", "attention_mask", "label_id"}]
    train_ds = train_ds.rename_column("label_id", "labels").remove_columns(columns_to_remove)
    valid_ds = valid_ds.rename_column("label_id", "labels").remove_columns(columns_to_remove)
    test_ds = test_ds.rename_column("label_id", "labels").remove_columns(columns_to_remove)

    model = _build_model(config.model_name, len(label2id), id2label, label2id)
    model = _maybe_apply_lora(model, config)

    args = TrainingArguments(
        output_dir=str(output_path / "trainer"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        weight_decay=config.weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model=config.metric_for_best_model,
        report_to=[],
        seed=config.seed,
    )

    class _TrainerWithCompat(CompatibleTrainer, Trainer):
        """Trainer variant that tolerates plain optimizers behind accelerate wrappers."""

    trainer = _TrainerWithCompat(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=trainer_compute_metrics,
    )
    trainer.train()

    eval_metrics = trainer.evaluate(eval_dataset=test_ds)
    if config.use_lora:
        # Merge the LoRA adapter into the base weights and save a standalone
        # classifier. Otherwise trainer.save_model would persist an adapter-only
        # checkpoint that requires `peft` at inference time and re-resolves the
        # base model from its name, breaking the local artifact contract.
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(str(output_path))
    else:
        trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    (output_path / "label_mapping.json").write_text(
        json.dumps({"label2id": label2id, "id2label": id2label}, indent=2),
        encoding="utf-8",
    )
    (output_path / "training_config.json").write_text(
        json.dumps(config.to_dict(), indent=2),
        encoding="utf-8",
    )
    (output_path / "test_metrics.json").write_text(
        json.dumps(eval_metrics, indent=2),
        encoding="utf-8",
    )
    _write_heldout_test_split(test_df, output_path)
    write_model_card(
        output_path=output_path / "model_card.md",
        model_name=config.model_name,
        task="text-classification",
        label_names=list(label2id),
        metrics={key: float(value) for key, value in eval_metrics.items() if isinstance(value, int | float)},
        limitations=[
            "The sample workflow uses synthetic data unless replaced by a real dataset.",
            "Subgroup performance and calibration must be validated before production use.",
            "The model may learn annotation artifacts or spurious textual cues.",
        ],
    )
    return output_path
