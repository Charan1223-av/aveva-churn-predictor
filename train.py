"""
Main training script.
Run: python train.py
"""
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.features.feature_store import FeatureStore
from src.models.trainer import ChurnModelTrainer
from src.pipeline.preprocessing import build_preprocessing_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("train.log"),
    ],
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=" * 60)
    logger.info("AVEVA Churn Predictor - Training Pipeline")
    logger.info("=" * 60)

    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # Step 1: Load data
    logger.info("Step 1/5: Loading raw data...")
    loader = DataLoader(config["data"]["raw_path"])
    tables = loader.load_all()

    # Step 2: Build feature store
    logger.info("Step 2/5: Building feature store...")
    fs_builder = FeatureStore(tables)
    feature_store = fs_builder.build()

    Path(config["data"]["processed_path"]).mkdir(parents=True, exist_ok=True)
    feature_store.to_parquet(config["data"]["feature_store_path"], index=False)
    logger.info(f"Feature store saved: {feature_store.shape}")

    # Step 3: Prepare X, y
    logger.info("Step 3/5: Preparing features and target...")
    trainer = ChurnModelTrainer(config)
    X, y = trainer.prepare_data(feature_store)

    if y.isna().any():
        logger.warning(f"Dropping {y.isna().sum()} rows with unmapped target values.")
        mask = y.notna()
        X, y = X[mask], y[mask]

    logger.info(f"Class distribution: {y.value_counts().to_dict()}")

    # Step 4: Split
    logger.info("Step 4/5: Splitting data...")
    test_size = config["splitting"]["test_size"]
    seed = config["project"]["random_seed"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed, stratify=y
        )
    except ValueError:
        logger.warning("Stratified split failed (too few samples per class), using random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed
        )

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Step 5: Compare + train
    logger.info("Step 5/5: Comparing candidate models and training final model...")

    # Build the preprocessor AFTER prepare_data so we only reference columns
    # that are actually in X (e.g. 'country' was already dropped).
    preprocessor = build_preprocessing_pipeline(config, available_columns=list(X_train.columns))

    algorithm, params, comparison_df = trainer.compare_models(
        X_train, y_train, X_test, y_test, preprocessor
    )
    logger.info(f"Model comparison complete. Best model: {algorithm}")
    logger.info(f"\n{comparison_df[['algorithm', 'roc_auc', 'f1', 'precision', 'recall', 'inference_ms_per_sample']].to_string(index=False)}")

    trainer.train_final_model(
        X_train, y_train, X_test, y_test,
        preprocessor, algorithm, params, version="v1.0",
        model_comparison=comparison_df,
    )

    logger.info("=" * 60)
    logger.info("Training complete! Model saved to: models/latest/")
    logger.info("Next steps:")
    logger.info("  API:       python -m uvicorn api.main:app --port 8000 --reload")
    logger.info("  Dashboard: streamlit run dashboard/app.py")
    logger.info("  Docker:    docker-compose up --build")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
