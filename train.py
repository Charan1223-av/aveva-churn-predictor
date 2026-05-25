"""
Main training script.
Run: python train.py

This script will:
1. Load all CSVs from data/raw/
2. Build the feature store
3. Train and evaluate XGBoost model
4. Save model artifacts to models/latest/
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
    logger.info(f"Columns: {list(feature_store.columns)}")

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

    # Step 5: Train
    logger.info("Step 5/5: Training model...")
    preprocessor = build_preprocessing_pipeline(config)
    algorithm = config["model"]["algorithm"]
    params = config["model"][algorithm]

    cv_metrics = trainer.cross_validate_model(X_train, y_train, preprocessor, algorithm, params)
    logger.info(
        f"CV ROC-AUC: {cv_metrics.get('cv_roc_auc_mean', 0):.4f} "
        f"(+/- {cv_metrics.get('cv_roc_auc_std', 0):.4f})"
    )

    trainer.train_final_model(
        X_train, y_train, X_test, y_test,
        preprocessor, algorithm, params, version="v1.0",
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
