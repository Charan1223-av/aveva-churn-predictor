"""
Model training, cross-validation, evaluation, and artifact saving.
"""
import json
import logging
import time
from pathlib import Path
from typing import Dict, Tuple, List

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from src.features.feature_policy import (
    EXCLUDED_FEATURES,
    EXCLUDED_FEATURE_TOKENS,
    PRIORITIZED_FEATURES,
)

logger = logging.getLogger(__name__)

class ChurnModelTrainer:
    SHAP_SAMPLE_SIZE = 300

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
        X = X[
            [
                c for c in X.columns
                if c not in EXCLUDED_FEATURES and not any(token in c.lower() for token in EXCLUDED_FEATURE_TOKENS)
            ]
        ]

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
        missing_prioritized = [f for f in PRIORITIZED_FEATURES if f not in X.columns]
        if missing_prioritized:
            logger.warning(f"Missing prioritized features in training set: {missing_prioritized}")
        return X, y

    def _build_pipeline(self, preprocessor, algorithm: str, params: dict, n_minority: int = 10) -> ImbPipeline:
        seed = self.config["project"]["random_seed"]
        clean_params = {k: v for k, v in params.items()}

        if algorithm == "xgboost":
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
        elif algorithm == "random_forest":
            base_model = RandomForestClassifier(**clean_params, random_state=seed, n_jobs=-1)
        else:
            base_model = LogisticRegression(**clean_params, random_state=seed)

        model = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
        if n_minority < 2:
            sampler = RandomOverSampler(random_state=seed)
        else:
            k = max(1, min(3, n_minority - 1))
            sampler = SMOTE(random_state=seed, k_neighbors=k)

        return ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", sampler),
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

        min_class_size = int(y.value_counts().min())
        n_minority_per_fold = max(1, (min_class_size * (n_splits - 1)) // n_splits)
        pipeline = self._build_pipeline(preprocessor, algorithm, params, n_minority=n_minority_per_fold)

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

    def compare_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        preprocessor,
    ) -> Tuple[str, dict, pd.DataFrame]:
        algorithms = self.config["model"].get("comparison_algorithms", [self.config["model"]["algorithm"]])
        explainability_scores = {
            "logistic_regression": 1.0,
            "random_forest": 0.8,
            "xgboost": 0.65,
            "lgbm": 0.65,
        }
        results: List[Dict[str, float]] = []

        for algorithm in algorithms:
            params = self.config["model"].get(algorithm, {})
            cv_metrics = self.cross_validate_model(X_train, y_train, preprocessor, algorithm, params)
            n_minority = int(y_train.value_counts().min())
            pipeline = self._build_pipeline(preprocessor, algorithm, params, n_minority=n_minority)
            pipeline.fit(X_train, y_train)

            y_prob = pipeline.predict_proba(X_test)[:, 1]
            latency_batch = X_test.head(min(100, len(X_test)))
            if len(latency_batch) == 0:
                inference_time_ms_per_sample = 0.0
            else:
                start = time.perf_counter()
                _ = pipeline.predict_proba(latency_batch)
                inference_time_ms_per_sample = ((time.perf_counter() - start) * 1000) / max(len(latency_batch), 1)
            threshold = self.find_optimal_threshold(y_test.values, y_prob)
            y_pred = (y_prob >= threshold).astype(int)

            results.append({
                "algorithm": algorithm,
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "precision": float(precision_score(y_test, y_pred, zero_division=0)),
                "recall": float(recall_score(y_test, y_pred, zero_division=0)),
                "f1": float(f1_score(y_test, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, y_prob)) if len(y_test.unique()) > 1 else 0.0,
                "avg_precision": float(average_precision_score(y_test, y_prob)) if len(y_test.unique()) > 1 else 0.0,
                "explainability_score": explainability_scores.get(algorithm, 0.5),
                "inference_ms_per_sample": float(inference_time_ms_per_sample),
                "cv_roc_auc_mean": cv_metrics.get("cv_roc_auc_mean", 0.0),
                "overfit_gap": cv_metrics.get("overfit_gap", 0.0),
            })

        comparison_df = pd.DataFrame(results).sort_values(
            by=["roc_auc", "f1", "inference_ms_per_sample"],
            ascending=[False, False, True],
        )
        comparison_df.to_csv(self.output_dir / "model_comparison.csv", index=False)

        best_algorithm = str(comparison_df.iloc[0]["algorithm"])
        best_params = self.config["model"][best_algorithm]
        logger.info(f"Selected model: {best_algorithm}")
        return best_algorithm, best_params, comparison_df

    def find_optimal_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
        best_idx = np.argmax(f1_scores[:-1])
        if len(thresholds) > 0:
            return float(thresholds[best_idx])
        return 0.45

    def _extract_feature_names(self, pipeline: ImbPipeline, X_ref: pd.DataFrame) -> List[str]:
        preprocessor = pipeline.named_steps["preprocessor"]
        try:
            return list(preprocessor.get_feature_names_out())
        except Exception:
            return list(X_ref.columns)

    @staticmethod
    def _align_feature_names(feature_names: List[str], target_length: int) -> List[str]:
        if len(feature_names) == target_length:
            return feature_names
        logger.warning(
            "Feature name alignment mismatch: names=%s, target=%s. Applying fallback alignment.",
            len(feature_names),
            target_length,
        )
        if len(feature_names) > target_length:
            return feature_names[:target_length]
        padded = list(feature_names)
        padded.extend([f"feature_{i}" for i in range(len(feature_names), target_length)])
        return padded

    @staticmethod
    def _is_prioritized_feature_match(feature_name: str, prioritized_name: str) -> bool:
        normalized = str(feature_name)
        return (
            normalized == prioritized_name
            or normalized.endswith(f"__{prioritized_name}")
            or normalized.endswith(f"_{prioritized_name}")
        )

    def _save_feature_importance_artifacts(
        self,
        pipeline: ImbPipeline,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        output_dir: Path,
    ) -> None:
        transformed_feature_names = self._extract_feature_names(pipeline, X_train)
        artifacts = []

        perm = permutation_importance(
            pipeline,
            X_test,
            y_test,
            n_repeats=5,
            random_state=self.config["project"]["random_seed"],
            scoring="roc_auc",
        )
        perm_df = pd.DataFrame({
            "feature": X_test.columns,
            "importance": perm.importances_mean,
            "method": "permutation_importance",
        })
        perm_df.to_csv(output_dir / "permutation_importance.csv", index=False)
        artifacts.append(perm_df)

        corr_df = X_test.copy()
        corr_df["target"] = y_test.values
        corr_series = corr_df.select_dtypes(include=[np.number]).corr()["target"].drop("target").abs().sort_values(ascending=False)
        corr_rank = corr_series.reset_index().rename(columns={"index": "feature", "target": "importance"})
        corr_rank["method"] = "correlation_abs_target"
        corr_rank.to_csv(output_dir / "correlation_importance.csv", index=False)
        artifacts.append(corr_rank)

        preprocessor = pipeline.named_steps["preprocessor"]
        x_train_tf = preprocessor.transform(X_train)
        rf = RandomForestClassifier(
            n_estimators=200,
            random_state=self.config["project"]["random_seed"],
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        rf.fit(x_train_tf, y_train)
        aligned_tree_names = self._align_feature_names(
            transformed_feature_names,
            len(rf.feature_importances_),
        )
        tree_df = pd.DataFrame({
            "feature": aligned_tree_names,
            "importance": rf.feature_importances_,
            "method": "tree_model_importance",
        })
        tree_df.to_csv(output_dir / "tree_feature_importance.csv", index=False)
        artifacts.append(tree_df)

        try:
            import shap
            import matplotlib.pyplot as plt

            x_test_tf = preprocessor.transform(X_test)
            if hasattr(x_test_tf, "toarray"):
                x_test_tf = x_test_tf.toarray()
            shap_sample = x_test_tf[: min(self.SHAP_SAMPLE_SIZE, len(x_test_tf))]
            explainer = shap.Explainer(rf, shap_sample)
            shap_values = explainer(shap_sample, check_additivity=False)
            shap_raw_values = shap_values.values
            if getattr(shap_raw_values, "ndim", 0) == 3:
                shap_raw_values = shap_raw_values[:, :, -1]
            shap_importance = np.abs(shap_raw_values).mean(axis=0)
            aligned_shap_names = self._align_feature_names(
                transformed_feature_names,
                len(shap_importance),
            )
            shap_df = pd.DataFrame({
                "feature": aligned_shap_names,
                "importance": shap_importance,
                "method": "shap",
            })
            shap_df.to_csv(output_dir / "shap_feature_importance.csv", index=False)
            artifacts.append(shap_df)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_raw_values,
                shap_sample,
                feature_names=self._align_feature_names(
                    transformed_feature_names,
                    shap_sample.shape[1],
                ),
                show=False,
            )
            plt.tight_layout()
            plt.savefig(output_dir / "shap_summary.png", dpi=150)
            plt.close()
        except Exception as e:
            logger.warning(f"SHAP analysis skipped due to error: {e}")

        combined = pd.concat(artifacts, ignore_index=True)
        combined["is_prioritized_feature"] = combined["feature"].apply(
            lambda f: any(self._is_prioritized_feature_match(f, pf) for pf in PRIORITIZED_FEATURES)
        )
        combined.to_csv(output_dir / "feature_importance_rankings.csv", index=False)

        prioritized_summary = {}
        for feature in PRIORITIZED_FEATURES:
            hits = combined[
                combined["feature"].apply(lambda f: self._is_prioritized_feature_match(f, feature))
            ]
            prioritized_summary[feature] = {
                "present_in_model_features": bool(feature in X_train.columns),
                "methods_available": sorted(hits["method"].unique().tolist()),
                "mean_importance": float(hits["importance"].mean()) if not hits.empty else 0.0,
            }
        with open(output_dir / "prioritized_feature_summary.json", "w") as f:
            json.dump(prioritized_summary, f, indent=2)

    def train_final_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        preprocessor,
        algorithm: str,
        params: dict,
        version: str = "v1.0",
        model_comparison: pd.DataFrame = None,
    ) -> ImbPipeline:
        n_minority = int(y_train.value_counts().min())
        pipeline = self._build_pipeline(preprocessor, algorithm, params, n_minority=n_minority)
        pipeline.fit(X_train, y_train)

        y_prob = pipeline.predict_proba(X_test)[:, 1]
        threshold = self.find_optimal_threshold(y_test.values, y_prob)
        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            "test_accuracy": float(accuracy_score(y_test, y_pred)),
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
        if model_comparison is not None:
            model_comparison.to_csv(version_dir / "model_comparison.csv", index=False)

        self._save_feature_importance_artifacts(pipeline, X_train, y_train, X_test, y_test, version_dir)

        metadata = {
            "version": version,
            "algorithm": algorithm,
            "params": params,
            "metrics": metrics,
            "feature_count": X_train.shape[1],
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": list(X_train.columns),
            "prioritized_features": PRIORITIZED_FEATURES,
            "excluded_feature_tokens": list(EXCLUDED_FEATURE_TOKENS),
        }
        with open(version_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        import shutil
        latest_dir = self.output_dir / "latest"
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        shutil.copytree(version_dir, latest_dir)

        logger.info(f"Model saved to {version_dir} and copied to models/latest/")
        return pipeline
