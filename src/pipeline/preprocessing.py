"""
sklearn preprocessing pipeline with ColumnTransformer.
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Optional


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Clip numerical features to [Q1 - 3*IQR, Q3 + 3*IQR]. Fit on train only."""
    def __init__(self, factor: float = 3.0):
        self.factor = factor
        self.lower_ = {}
        self.upper_ = {}

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for i, col in enumerate(X.columns):
            q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
            iqr = q3 - q1
            self.lower_[i] = q1 - self.factor * iqr
            self.upper_[i] = q3 + self.factor * iqr
        return self

    def transform(self, X, y=None):
        X = pd.DataFrame(X).copy()
        for i in range(X.shape[1]):
            X.iloc[:, i] = X.iloc[:, i].clip(
                lower=self.lower_.get(i, -np.inf),
                upper=self.upper_.get(i, np.inf),
            )
        return X.values


def build_preprocessing_pipeline(
    config: dict,
    available_columns: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Build the full ColumnTransformer preprocessing pipeline.

    Parameters
    ----------
    config : dict
        Full project config.
    available_columns : list, optional
        If provided, only include feature columns that are present in this list.
        This prevents KeyError when `country` or other high-cardinality columns
        have already been dropped during prepare_data().
    """
    def _filter(cols):
        if available_columns is None:
            return [f for f in cols if f]
        return [f for f in cols if f and f in available_columns]

    num_features   = _filter(config["features"]["numerical"])
    low_card_cat   = _filter(config["features"]["categorical"])
    high_card_cat  = _filter(config["features"].get("high_cardinality_categorical", []))
    bool_features  = _filter(config["features"]["boolean"])

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clipper", OutlierClipper(factor=3.0)),
        ("scaler", StandardScaler()),
    ])

    categorical_low_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")),
    ])

    categorical_high_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    bool_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    transformers = []
    if num_features:
        transformers.append(("num", numerical_pipeline, num_features))
    if low_card_cat:
        transformers.append(("cat_low", categorical_low_pipeline, low_card_cat))
    if high_card_cat:
        transformers.append(("cat_high", categorical_high_pipeline, high_card_cat))
    if bool_features:
        transformers.append(("bool", bool_pipeline, bool_features))

    return ColumnTransformer(transformers=transformers, remainder="drop")
