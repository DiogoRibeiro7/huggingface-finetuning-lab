"""Token-classification (NER) utilities: schema, alignment, and BIO-span metrics."""

from hf_finetuning_lab.token_classification.alignment import align_word_labels_to_subwords
from hf_finetuning_lab.token_classification.metrics import (
    extract_entities,
    sequence_tagging_report,
)
from hf_finetuning_lab.token_classification.schema import (
    NERExample,
    generate_synthetic_ner_data,
    validate_ner_dataset,
    write_synthetic_ner_jsonl,
)

__all__ = [
    "NERExample",
    "align_word_labels_to_subwords",
    "extract_entities",
    "generate_synthetic_ner_data",
    "sequence_tagging_report",
    "validate_ner_dataset",
    "write_synthetic_ner_jsonl",
]
