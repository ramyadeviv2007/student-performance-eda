"""Prepare leak-free train, validation, and test datasets."""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import DATA_PATH, MODEL_DIR, OUTPUTS_DIR, PASS_THRESHOLD, SEED, ensure_project_dirs
from src.data_types import PreparedData

TARGET = "pass_fail"
EXCLUDED_GRADES = ("G1", "G2", "G3")


def add_target(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the chosen passing definition: G3 >= 10."""
    if "G3" not in frame.columns:
        raise ValueError("The input dataframe must contain the G3 final-grade column.")
    result = frame.copy()
    result[TARGET] = (result["G3"] >= PASS_THRESHOLD).astype("int64")
    return result


def prepare_data(frame: pd.DataFrame, *, save_artifacts: bool = True) -> PreparedData:
    """Clean exact duplicates, split, and fit preprocessing on training data only."""
    ensure_project_dirs()
    working = add_target(frame).drop_duplicates().copy()
    if working.isna().any().any():
        missing_rows = int(working.isna().any(axis=1).sum())
        working = working.dropna().copy()
        print(f"Removed {missing_rows} rows containing missing values.")
    if working.empty:
        raise ValueError("No complete observations remain after duplicate/missing-value handling.")

    X = working.drop(columns=[*EXCLUDED_GRADES, TARGET], errors="raise")
    assert not set(EXCLUDED_GRADES).intersection(X.columns), "G1/G2/G3 must not be model inputs."
    y = working[TARGET]
    if y.nunique() != 2 or y.value_counts().min() < 3:
        raise ValueError("Both target classes need at least three observations for a stratified split.")

    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_holdout, y_holdout, test_size=0.5, random_state=SEED, stratify=y_holdout
    )
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    outlier_records = []
    all_numeric = working.drop(columns=[TARGET]).select_dtypes(include=np.number)
    for column in all_numeric:
        q1, q3 = all_numeric[column].quantile([0.25, 0.75])
        spread = q3 - q1
        lower, upper = q1 - 1.5 * spread, q3 + 1.5 * spread
        outlier_records.append(
            {
                "feature": column,
                "q1": float(q1),
                "q3": float(q3),
                "lower_bound": float(lower),
                "upper_bound": float(upper),
                "outlier_count": int(((all_numeric[column] < lower) | (all_numeric[column] > upper)).sum()),
            }
        )
    outlier_frame = pd.DataFrame(outlier_records)
    if save_artifacts:
        outlier_frame.to_csv(OUTPUTS_DIR / "iqr_outlier_counts.csv", index=False, encoding="utf-8")
    if save_artifacts and not all_numeric.empty:
        fig, axes = plt.subplots(
            nrows=len(all_numeric.columns),
            figsize=(10, max(3, 0.75 * len(all_numeric.columns))),
            squeeze=False,
        )
        for axis, column in zip(axes[:, 0], all_numeric.columns):
            axis.boxplot(all_numeric[column].dropna(), orientation="horizontal")
            axis.set_ylabel(column, rotation=0, ha="right", va="center")
            axis.set_xlabel("Observed value")
        fig.suptitle("Numeric Feature Boxplots for IQR Outlier Inspection")
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "figures" / "iqr_boxplots.png", dpi=200)
        plt.close(fig)
    transformers: list[tuple[str, object, list[str]]] = []
    if categorical:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore"), categorical))
    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))
    transformer = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
    X_train_array = transformer.fit_transform(X_train).astype("float32")
    X_val_array = transformer.transform(X_val).astype("float32")
    X_test_array = transformer.transform(X_test).astype("float32")

    prepared = PreparedData(
        X_train=X_train_array,
        X_val=X_val_array,
        X_test=X_test_array,
        y_train=y_train.astype("float32"),
        y_val=y_val.astype("float32"),
        y_test=y_test.astype("int64"),
        raw_X_train=X_train,
        raw_X_val=X_val,
        raw_X_test=X_test,
        transformer=transformer,
        feature_names=X.columns.tolist(),
        categorical_columns=categorical,
        numeric_columns=numeric,
    )
    if save_artifacts:
        joblib.dump(transformer, MODEL_DIR / "preprocessor.joblib")
        joblib.dump(prepared, MODEL_DIR / "prepared_data.joblib")
        split_summary = pd.DataFrame(
            {
                "split": ["train", "validation", "test"],
                "rows": [len(y_train), len(y_val), len(y_test)],
                "pass_count": [
                    int((y_train == 1).sum()),
                    int((y_val == 1).sum()),
                    int((y_test == 1).sum()),
                ],
                "fail_count": [
                    int((y_train == 0).sum()),
                    int((y_val == 0).sum()),
                    int((y_test == 0).sum()),
                ],
            }
        )
        split_summary.to_csv(OUTPUTS_DIR / "split_summary.csv", index=False, encoding="utf-8")
        notes = (
            "# Preprocessing notes\n\n"
            f"- Passing is defined here as `G3 >= {PASS_THRESHOLD}`; this is the project's chosen threshold.\n"
            f"- Removed {int(frame.duplicated().sum())} exact duplicate source rows. "
            f"Missing-value rows removed after deduplication: {len(add_target(frame).drop_duplicates()) - len(working)}.\n"
            "- Excluded G1, G2, and G3 from predictors because they are grades from the same course and "
            "would allow the model to copy information directly related to the target.\n"
            f"- Checked {len(outlier_frame)} numeric columns using boxplots and the 1.5 x IQR rule. "
            f"Potential outlier count across columns (one observation can be counted in several columns): "
            f"{int(outlier_frame['outlier_count'].sum())}; retained all observations because unusual values "
            "are not automatically errors. Counts are in `outputs/iqr_outlier_counts.csv`.\n"
            f"- Stratified split with seed {SEED}: {len(y_train)} train, {len(y_val)} validation, "
            f"{len(y_test)} test observations.\n"
            "- One ColumnTransformer uses one-hot encoding for categories and standardization for numeric "
            "features; it is fitted on the training partition only.\n"
        )
        (OUTPUTS_DIR / "preprocessing_notes.md").write_text(notes, encoding="utf-8")
    print(f"Prepared split sizes: train={len(y_train)}, validation={len(y_val)}, test={len(y_test)}")
    print(f"Encoded feature count: {X_train_array.shape[1]}")
    return prepared


def load_and_prepare(path=DATA_PATH) -> PreparedData:
    """Read the source CSV and prepare model arrays."""
    frame = pd.read_csv(path, sep=";", encoding="utf-8")
    return prepare_data(frame)


if __name__ == "__main__":
    load_and_prepare()
