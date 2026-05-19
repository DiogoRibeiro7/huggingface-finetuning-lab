"""Task-specific model-card limitations and a thin wrapper over ``write_model_card``."""

from __future__ import annotations

from pathlib import Path

from hf_finetuning_lab.model_cards.model_card import write_model_card

SUPPORTED_TASKS: tuple[str, ...] = (
    "text-classification",
    "token-classification",
    "retrieval",
)

_TASK_LIMITATIONS: dict[str, tuple[str, ...]] = {
    "text-classification": (
        "Aggregate metrics can hide poor subgroup performance; pair this card with the v0.4 robust-evaluation report.",
        "Predicted-label distribution may drift when input distributions shift; track PSI between deployment windows.",
        "Default decision thresholds rarely maximise F1 — tune per class on a held-out validation set.",
    ),
    "token-classification": (
        "Entity-level metrics measure exact boundary + type matches, so off-by-one tokenisation errors are penalised.",
        "Out-of-vocabulary entity types (not seen during training) are not recovered by this model.",
        "Subword tokenisation alignment can drop labels when the tokenizer differs between training and inference.",
    ),
    "retrieval": (
        "Recall@k and nDCG@k depend on the curated relevance set; coverage gaps inflate apparent quality.",
        "Embedding domain shift between the corpus and live queries degrades retrieval; monitor query embedding norms and top-k score distributions.",
        "Dense retrieval can miss lexical exact-match cases; consider hybrid retrieval (BM25 + dense) for production.",
    ),
}


def task_limitations(task: str) -> list[str]:
    """Return the curated limitation bullets for ``task``."""
    if task not in _TASK_LIMITATIONS:
        raise ValueError(
            f"Unsupported task '{task}'. Choose one of: {sorted(_TASK_LIMITATIONS)}."
        )
    return list(_TASK_LIMITATIONS[task])


def write_task_model_card(
    output_path: str | Path,
    model_name: str,
    task: str,
    label_names: list[str],
    metrics: dict[str, float],
    extra_limitations: list[str] | None = None,
) -> Path:
    """Write a model card pre-loaded with task-specific limitations.

    ``extra_limitations`` are appended after the curated task limitations and
    are intended for project-specific caveats (synthetic labels, restricted
    geography, regulated domain, etc.).
    """
    limitations = task_limitations(task)
    if extra_limitations:
        limitations.extend(extra_limitations)
    return write_model_card(
        output_path=output_path,
        model_name=model_name,
        task=task,
        label_names=label_names,
        metrics=metrics,
        limitations=limitations,
    )
