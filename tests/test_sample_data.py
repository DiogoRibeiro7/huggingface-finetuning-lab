from hf_finetuning_lab.sample_data import generate_support_ticket_data


def test_generate_support_ticket_data_shape_and_columns() -> None:
    df = generate_support_ticket_data(rows=100, seed=1)
    assert len(df) == 100
    assert {"id", "text", "label"}.issubset(df.columns)
    assert df["label"].nunique() >= 2
    assert df["text"].str.len().min() > 0
