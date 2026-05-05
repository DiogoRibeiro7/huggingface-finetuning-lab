import pandas as pd

from hf_finetuning_lab.data.splits import stratified_train_valid_test_split


def test_stratified_split_preserves_total_rows() -> None:
    df = pd.DataFrame({"x": range(100), "label_id": [0, 1] * 50})
    train, valid, test = stratified_train_valid_test_split(df, "label_id", test_size=0.2, validation_size=0.1)
    assert len(train) + len(valid) + len(test) == 100
    assert set(train["label_id"].unique()) == {0, 1}
    assert set(test["label_id"].unique()) == {0, 1}
