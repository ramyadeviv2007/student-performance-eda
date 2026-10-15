"""Fast checks for target construction, splitting, and train-only preprocessing."""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import add_target, prepare_data


def sample_frame(rows: int = 100) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "school": ["GP" if index % 2 else "MS" for index in range(rows)],
            "sex": ["F" if index % 3 else "M" for index in range(rows)],
            "age": [15 + index % 4 for index in range(rows)],
            "absences": [index % 7 for index in range(rows)],
            "freetime": [1 + (index // 4) % 5 for index in range(rows)],
            "G1": [index % 20 for index in range(rows)],
            "G2": [index % 20 for index in range(rows)],
            "G3": [10 if index % 2 else 9 for index in range(rows)],
        }
    )


def test_source_dataset_has_expected_columns():
    from src.config import DATA_PATH

    assert DATA_PATH.exists(), "Run `python -m src.download_data` before the test suite."
    assert pd.read_csv(DATA_PATH, sep=";", encoding="utf-8").shape == (395, 33)


def test_target_threshold_is_inclusive():
    result = add_target(pd.DataFrame({"G3": [10, 9]}))
    assert result["pass_fail"].tolist() == [1, 0]


def test_exact_duplicates_are_removed():
    frame = sample_frame(30)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    prepared = prepare_data(frame, save_artifacts=False)
    assert len(prepared.y_train) + len(prepared.y_val) + len(prepared.y_test) == 30


def test_split_sizes_and_no_overlap():
    frame = sample_frame(100)
    prepared = prepare_data(frame, save_artifacts=False)
    assert [len(prepared.y_train), len(prepared.y_val), len(prepared.y_test)] == [70, 15, 15]
    train_ids, val_ids, test_ids = (
        set(prepared.raw_X_train.index),
        set(prepared.raw_X_val.index),
        set(prepared.raw_X_test.index),
    )
    assert not (train_ids & val_ids or train_ids & test_ids or val_ids & test_ids)


def test_grades_are_excluded_and_encoded_dimensions_match():
    prepared = prepare_data(sample_frame(100), save_artifacts=False)
    assert not {"G1", "G2", "G3"}.intersection(prepared.feature_names)
    assert prepared.X_train.shape[1] == prepared.X_val.shape[1] == prepared.X_test.shape[1]


def test_transformer_is_fitted_on_training_partition_only():
    frame = sample_frame(100)
    frame.loc[0, "age"] = 1000
    prepared = prepare_data(frame, save_artifacts=False)
    scaler = prepared.transformer.named_transformers_["numeric"]
    age_column_index = prepared.numeric_columns.index("age")
    assert isinstance(scaler, StandardScaler)
    assert scaler.mean_[age_column_index] == prepared.raw_X_train["age"].mean()
    assert scaler.mean_[age_column_index] != frame["age"].mean()
    assert isinstance(prepared.transformer.named_transformers_["categorical"], OneHotEncoder)
