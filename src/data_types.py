"""Serializable data containers shared by the project modules."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer


@dataclass
class PreparedData:
    """Data splits and the fitted transformer, ready for modeling."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    raw_X_train: pd.DataFrame
    raw_X_val: pd.DataFrame
    raw_X_test: pd.DataFrame
    transformer: ColumnTransformer
    feature_names: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
