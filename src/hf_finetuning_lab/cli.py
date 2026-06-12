from __future__ import annotations

from pathlib import Path
from typing import Annotated

import click
import typer

from hf_finetuning_lab import __version__
from hf_finetuning_lab.artifacts import verify_artifact
from hf_finetuning_lab.config import TrainingConfig
from hf_finetuning_lab.data.hub import list_hub_presets, write_hub_dataset_csv
from hf_finetuning_lab.evaluation.evaluator import evaluate_model
from hf_finetuning_lab.inference.predictor import batch_predict
from hf_finetuning_lab.sample_data import write_sample_data
from hf_finetuning_lab.training.trainer import train_text_classifier

app = typer.Typer(help="Hugging Face fine-tuning lab CLI.", rich_markup_mode=None)


@app.command("version")
def version() -> None:
    """Print the installed package version."""
    typer.echo(__version__)


@app.command("list-commands")
def list_commands() -> None:
    """Print the names of every registered CLI command, sorted."""
    names: list[str] = []
    for command in app.registered_commands:
        if command.name:
            names.append(command.name)
        elif command.callback is not None:
            names.append(command.callback.__name__)
    for name in sorted(names):
        typer.echo(name)


@app.command("verify-artifact")
def verify_artifact_cmd(
    model_dir: Path = typer.Option(..., help="Local model directory to inspect."),
    strict: Annotated[
        bool,
        typer.Option("--strict", help="Exit with code 1 if any recommended file is missing."),
    ] = False,
) -> None:
    """Verify that a model directory follows the stable v1.0 artifact layout."""
    report = verify_artifact(model_dir)
    for check in report.checks:
        marker = {"ok": "OK ", "warning": "WARN", "missing": "MISS"}[check.status]
        typer.echo(f"[{marker}] {check.name}: {check.detail}")
    if not report.ok:
        raise typer.Exit(code=1)
    if strict and report.warnings:
        raise typer.Exit(code=1)


@app.command("sample-data")
def sample_data(
    output: Path = typer.Option(..., help="Output CSV path."),
    rows: int = typer.Option(2000, help="Number of rows to generate."),
    seed: int = typer.Option(42, help="Random seed."),
) -> None:
    """Generate synthetic support-ticket text-classification data."""
    path = write_sample_data(output=output, rows=rows, seed=seed)
    typer.echo(f"Wrote sample data to {path}")


@app.command("list-hub-datasets")
def list_hub_datasets_cmd() -> None:
    """List built-in Hugging Face Hub preset names."""
    for name in list_hub_presets():
        typer.echo(name)


@app.command("fetch-hub-dataset")
def fetch_hub_dataset(
    name: str = typer.Option(..., help="Hub preset name (see list-hub-datasets)."),
    output_dir: Path = typer.Option(..., help="Directory to write per-split CSV files."),
    max_rows_per_split: int | None = typer.Option(
        None, help="Cap rows per split. Omit to use the full dataset."
    ),
) -> None:
    """Download a preset Hub dataset and write one CSV per split."""
    paths = write_hub_dataset_csv(name, output_dir=output_dir, max_rows_per_split=max_rows_per_split)
    for split, path in paths.items():
        typer.echo(f"{split}: {path}")


@app.command("train")
def train(
    ctx: typer.Context,
    input: Path = typer.Option(..., help="Input CSV or JSONL file."),  # noqa: A002
    output_dir: Path = typer.Option(..., help="Model output directory."),
    model_name: str = typer.Option("distilbert-base-uncased", help="Hugging Face model name."),
    text_col: str = typer.Option("text", help="Text column name."),
    label_col: str = typer.Option("label", help="Label column name."),
    epochs: int = typer.Option(2, help="Number of training epochs."),
    batch_size: int = typer.Option(16, help="Batch size."),
    learning_rate: float = typer.Option(2e-5, help="Learning rate."),
    max_length: int = typer.Option(160, help="Maximum tokenized length."),
    config_file: Path | None = typer.Option(None, help="Optional YAML config."),
    use_lora: Annotated[
        bool,
        typer.Option("--use-lora", help="Enable PEFT/LoRA fine-tuning."),
    ] = False,
) -> None:
    """Fine-tune a transformer for text classification."""
    if config_file is not None:
        # The YAML file is the complete source of truth; reject explicitly
        # passed training flags so they cannot be silently ignored.
        config_flags = (
            "model_name", "text_col", "label_col", "epochs",
            "batch_size", "learning_rate", "max_length", "use_lora",
        )
        overridden = [
            name
            for name in config_flags
            if ctx.get_parameter_source(name) != click.core.ParameterSource.DEFAULT
        ]
        if overridden:
            raise typer.BadParameter(
                f"--config-file cannot be combined with explicit flags: {overridden}. "
                "Put these settings in the YAML file instead."
            )
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
