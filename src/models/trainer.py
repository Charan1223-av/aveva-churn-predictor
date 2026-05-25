"""
Model training, cross-validation, evaluation, and artifact saving.
"""
import json
import logging
import joblib
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, classification_report,
    precision_recall_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

logger = logging.getLogger(__name__)


class ChurnModelTrainer:
    def __init__(self, config: dict, model_output_dir: str = "models"):
        self.config = config
        self.output_dir = Path(model_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_data(self, feature_store: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        target_col = self.config["target"]["column"]
        binary_mapping = self.config["target"]["binary_mapping"]
        y = feature_store[target_col].map(binary_mapping)

        drop_cols = self.config["features"]["drop_columns"] + [target_col, "customer_name", "country"]
        drop_cols = [c for c in drop_cols if c in feature_store.columns]
        X = feature_store.drop(columns=drop_cols)

        all_feature_cols = (
            self.config["features"]["numerical"]
            + self.config["features"]["categorical"]
            + self.config["features"].get("high_cardinality_categorical", [])
            + self.config["features"]["boolean"]
        )
        valid_cols = [c for c in all_feature_cols if c in X.columns]
        X = X[valid_cols]

        logger.info(f"Features: {X.shape[1]}, Samples: {X.shape[0]}")
        logger.info(f"Target distribution:\n{y.value_counts().to_string()}")
        return X, y

    def _build_pipeline(self, preprocessor, algorithm: str, params: dict,
                        n_minority: int = 10) -> ImbPipeline:
        seed = self.config["project"]["random_seed"]

        # Build a clean copy of params — strip keys that we set explicitly below
        clean_params = {k: v for k, v in params.items()}

        if algorithm == "xgboost":
            # Remove eval_metric from params if present (we set it below explicitly
            # only when it's NOT already there to avoid the duplicate-keyword error).
            # Safest: always pop it, then add once.
            clean_params.pop("eval_metric", None)
            base_model = xgb.XGBClassifier(
                **clean_params,
                random_state=seed,
                n_jobs=-1,
                eval_metric="aucpr",
                verbosity=0,
            )
        elif algorithm == "lgbm":
            base_model = lgb.LGBMClassifier(**clean_params, random_state=seed, n_jobs=-1, verbose=-1)
        else:
            base_model = LogisticRegression(**clean_params, random_state=seed)

        model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)

        # SMOTE k_neighbors must be < minority class size; use 1 as minimum safe value
        k = max(1, min(3, n_minority - 1))
        smote = SMOTE(random_state=seed, k_neighbors=k)

        return ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", smote),
            ("classifier", model),
        ])

    def cross_validate_model(
        self, X: pd.DataFrame, y: pd.Series, preprocessor, algorithm: str, params: dict
    ) -> Dict[str, float]:
        n_splits = min(self.config["cross_validation"]["n_splits"], y.value_counts().min())
        n_splits = max(n_splits, 2)
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True,
            random_state=self.config["project"]["random_seed"],
        )

        # Estimate minority size per fold for SMOTE
        min_class_size = int(y.value_counts().min())
        n_minority_per_fold = max(1, min_class_size - (min_class_size // n_splits))
        pipeline = self._build_pipeline(preprocessor, algorithm, params,
                                        n_minority=n_minority_per_fold)

        scoring = {
            "roc_auc": "roc_auc",
            "average_precision": "average_precision",
            "f1": "f1",
            "recall": "recall",
        }

        try:
            results = cross_validate(
                pipeline, X, y, cv=skf, scoring=scoring,
                return_train_score=True, n_jobs=1,
            )
            metrics = {
                "cv_roc_auc_mean": float(results["test_roc_auc"].mean()),
                "cv_roc_auc_std": float(results["test_roc_auc"].std()),
                "cv_avg_precision_mean": float(results["test_average_precision"].mean()),
                "cv_f1_mean": float(results["test_f1"].mean()),
                "cv_recall_mean": float(results["test_recall"].mean()),
                "train_roc_auc_mean": float(results["train_roc_auc"].mean()),
                "overfit_gap": float(
                    results["train_roc_auc"].mean() - results["test_roc_auc"].mean()
                ),
            }
        except Exception as e:
            logger.warning(f"Cross-validation failed ({e}), using dummy metrics for small dataset.")
            metrics = {
                "cv_roc_auc_mean": 0.0, "cv_roc_auc_std": 0.0,
                "cv_avg_precision_mean": 0.0, "cv_f1_mean": 0.0,
                "cv_recall_mean": 0.0, "train_roc_auc_mean": 0.0, "overfit_gap": 0.0,
            }

        logger.info(f"CV Results ({algorithm}): {json.dumps(metrics, indent=2)}")
        if metrics.get("overfit_gap", 0) > 0.1:
            logger.warning(f"Overfitting detected! Gap: {metrics['overfit_gap']:.3f}")
        return metrics

    def find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
        best_idx = np.argmax(f1_scores[:-1])
        if len(thresholds) > 0:
            return float(thresholds[best_idx])
        return 0.45

    def train_final_model(
        self,
        X_train: pd.DataFrame, y_train: pd.Series,
        X_test: pd.DataFrame, y_test: pd.Series,
        preprocessor, algorithm: str, params: dict,
        version: str = "v1.0",
    ) -> ImbPipeline:
        n_minority = int(y_train.value_counts().min())
        pipeline = self._build_pipeline(preprocessor, algorithm, params, n_minority=n_minority)
        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_test)[:, 1]
        threshold = self.find_optimal_threshold(y_test.values, y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            "test_roc_auc": float(roc_auc_score(y_test, y_prob)) if len(y_test.unique()) > 1 else 0.0,
            "test_avg_precision": float(average_precision_score(y_test, y_prob)) if len(y_test.unique()) > 1 else 0.0,
            "test_f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "threshold": threshold,
        }

        logger.info(f"\nTest Metrics:\n{json.dumps(metrics, indent=2)}")
        logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")

        version_dir = self.output_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, version_dir / "pipeline.joblib")

        metadata = {
            "version": version, "algorithm": algorithm, "params": params,
            "metrics": metrics, "feature_count": X_train.shape[1],
            "train_samples": len(X_train), "test_samples": len(X_test),
            "features": list(X_train.columns),
        }
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Copy to models/latest/ (Windows-safe — no symlinks needed)
        import shutil
        latest_dir = self.output_dir / "latest"
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        shutil.copytree(version_dir, latest_dir)

        logger.info(f"Model saved to {version_dir} and copied to models/latest/")
        return pipeline
