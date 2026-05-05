from hf_finetuning_lab.model_cards.model_card import write_model_card


def test_write_model_card(tmp_path) -> None:
    output = tmp_path / "model_card.md"
    write_model_card(
        output_path=output,
        model_name="distilbert-base-uncased",
        task="text-classification",
        label_names=["a", "b"],
        metrics={"f1": 0.8},
        limitations=["synthetic data"],
    )
    text = output.read_text(encoding="utf-8")
    assert "Model Card" in text
    assert "synthetic data" in text
    assert "f1" in text
