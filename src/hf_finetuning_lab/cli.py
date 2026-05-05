from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from hf_finetuning_lab.config import TrainingConfig
from hf_finetuning_lab.evaluation.evaluator import evaluate_model
from hf_finetuning_lab.inference.predictor import batch_predict
from hf_finetuning_lab.sample_data import write_sample_data
from hf_finetuning_lab.training.trainer import train_text_classifier

app = typer.Typer(help="Hugging Face fine-tuning lab CLI.")


@app.command("sample-data")
def sample_data(
    output: Path = typer.Option(..., help="Output CSV path."),
    rows: int = typer.Option(2000, help="Number of rows to generate."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Generate synthetic support-ticket text-classification data."""
    path = write_sample_data(output=output, rows=rows, seed=seed)
    typer.echo(f"Wrote sample data to {path}")


@app.command("train")
def train(
    input: Path = typer.Option(..., help="Input CSV or JSONL file."),  # noqa: A002
    output_dir: Path = typer.Option(..., help="Model output directory."),
    model_name: str = typer.Option("distilbert-base-uncased", help="Hugging Face model name."),
    text_col: str = typer.Option("text", help="Text column name."),
    label_col: str = typer.Option("label", help="Label column name."),
    epochs: int = typer.Option(2, help="Number of training epochs."),
    batch_size: int = typer.Option(16, help="Batch size."),
    learning_rate: float = typer.Option(2e-5, help="Learning rate."),
    max_length: int = typer.Option(160, help="Maximum tokenized length."),
    config_file: Optional[Path] = typer.Option(None, help="Optional YAML config."),
    use_lora: bool = typer.Option(False, help="Enable PEFT/LoRA fine-tuning."),
) -> None:
    """Fine-tune a transformer for text classification."""
    if config_file is not None:
        config = TrainingConfig.from_yaml(config_file)
    else:
        config = TrainingConfig(
            model_name=model_name,
            text_col=text_col,
            label_col=label_col,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
            use_lora=use_lora,
        )
    output_path = train_text_classifier(input_path=input, output_dir=output_dir, config=config)
    typer.echo(f"Saved model to {output_path}")


@app.command("evaluate")
def evaluate(
    model_dir: Path = typer.Option(..., help="Local model directory."),
    input: Path = typer.Option(..., help="Labelled CSV or JSONL file."),  # noqa: A002
    output: Path = typer.Option(..., help="Output JSON file."),
    text_col: str = typer.Option("text", help="Text column name."),
    label_col: str = typer.Option("label", help="Label column name."),
) -> None:
    """Evaluate a local model on labelled data."""
    path = evaluate_model(model_dir=model_dir, input_path=input, output_path=output, text_col=text_col, label_col=label_col)
    typer.echo(f"Wrote evaluation to {path}")


@app.command("predict")
def predict(
    model_dir: Path = typer.Option(..., help="Local model directory."),
    input: Path = typer.Option(..., help="Input CSV or JSONL file."),  # noqa: A002
    output: Path = typer.Option(..., help="Output predictions CSV."),
    text_col: str = typer.Option("text", help="Text column name."),
    device: int = typer.Option(-1, help="Device ID. Use -1 for CPU."),
) -> None:
    """Run batch inference with a local model."""
    path = batch_predict(model_dir=model_dir, input_path=input, output_path=output, text_col=text_col, device=device)
    typer.echo(f"Wrote predictions to {path}")


@app.command("serve")
def serve(
    model_dir: Path = typer.Option(..., help="Local model directory."),
    host: str = typer.Option("127.0.0.1", help="Server host."),
    port: int = typer.Option(8000, help="Server port."),
) -> None:
    """Serve a local text-classification model with FastAPI."""
    import uvicorn

    from hf_finetuning_lab.serving.api import create_app

    app_instance = create_app(model_dir=model_dir)
    uvicorn.run(app_instance, host=host, port=port)
