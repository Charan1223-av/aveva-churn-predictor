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


def build_preprocessing_pipeline(config: dict) -> ColumnTransformer:
    """Build the full ColumnTransformer preprocessing pipeline."""
    num_features = [f for f in config["features"]["numerical"] if f]
    low_card_cat = [f for f in config["features"]["categorical"] if f]
    high_card_cat = config["features"].get("high_cardinality_categorical", [])
    bool_features = [f for f in config["features"]["boolean"] if f]

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
