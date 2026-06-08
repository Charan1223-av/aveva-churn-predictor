"""
Production Inference Script.
Runs predictions on NEW customers WITHOUT renewal_status.

Run: python predict_production.py
Requires: Trained model in models/latest/ (run train.py first)
"""
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import DataLoader
from src.features.feature_store import FeatureStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def prob_to_risk(prob: float) -> str:
    if prob < 0.25:
        return "Low"
    elif prob < 0.50:
        return "Medium"
    elif prob < 0.75:
        return "High"
    else:
        return "Critical"


def main():
    logger.info("=" * 60)
    logger.info("AVEVA Churn Predictor - PRODUCTION INFERENCE")
    logger.info("=" * 60)

    # Load config
    with open("config/config.yaml") as f:
        config = yaml.safe_load(f)

    # Step 1: Load trained model
    model_path = Path("models/latest/pipeline.joblib")
    meta_path = Path("models/latest/metadata.json")

    if not model_path.exists():
        logger.error("No trained model found! Run 'python train.py' first.")
        sys.exit(1)

    pipeline = joblib.load(model_path)
    with open(meta_path) as f:
        metadata = json.load(f)

    logger.info(f"Model loaded: {metadata['version']} ({metadata['algorithm']})")
    logger.info(f"Model features: {metadata['feature_count']}")

    # Step 2: Load production data (NO renewal_status)
    production_path = "data/production"
    logger.info(f"Loading production data from: {production_path}")

    loader = DataLoader(production_path)
    tables = loader.load_all()

    # Step 3: Build feature store (same engineering as training)
    logger.info("Building feature store for production customers...")
    fs_builder = FeatureStore(tables)
    feature_store = fs_builder.build()
    logger.info(f"Feature store shape: {feature_store.shape}")

    # Save production feature store for dashboard deep-dive
    prod_fs_path = Path("data/production/feature_store.parquet")
    feature_store.to_parquet(prod_fs_path, index=False)
    logger.info(f"Production feature store saved: {prod_fs_path}")

    # Step 4: Prepare features (same as training but WITHOUT target)
    customer_ids = feature_store["customer_id"].values
    customer_names = feature_store.get("customer_name", pd.Series([""] * len(feature_store))).values

    # Drop non-feature columns (same as training)
    drop_cols = config["features"]["drop_columns"] + ["customer_name", "country"]
    drop_cols = [c for c in drop_cols if c in feature_store.columns]
    X = feature_store.drop(columns=[c for c in drop_cols if c in feature_store.columns], errors="ignore")
    X = X[[c for c in X.columns if "nps" not in c.lower() and "training" not in c.lower()]]

    # Keep only columns the model was trained on
    trained_features = metadata["features"]
    excluded_trained = [c for c in trained_features if "nps" in c.lower() or "training" in c.lower()]
    if excluded_trained:
        logger.warning(
            "Loaded model still expects excluded features (NPS/training). "
            "Retrain with updated pipeline to fully remove them."
        )
    available_features = [c for c in trained_features if c in X.columns]
    missing_features = [c for c in trained_features if c not in X.columns]

    if missing_features:
        logger.warning(f"Missing features (will be filled with 0): {missing_features}")
        for col in missing_features:
            X[col] = 0

    X = X[trained_features]  # Ensure exact column order
    logger.info(f"Inference features: {X.shape}")

    # Step 5: Generate predictions
    logger.info("Generating churn predictions...")
    probabilities = pipeline.predict_proba(X)[:, 1]

    threshold = metadata.get("metrics", {}).get("threshold", config["model"]["threshold"])

    # Step 6: Build results
    results = []
    for i, (cid, name, prob) in enumerate(zip(customer_ids, customer_names, probabilities)):
        results.append({
            "customer_id": cid,
            "customer_name": name,
            "churn_probability": round(float(prob), 4),
            "risk_category": prob_to_risk(prob),
            "is_at_risk": bool(prob >= threshold),
            "risk_score": int(round(prob * 100)),
        })

    results_df = pd.DataFrame(results).sort_values("churn_probability", ascending=False)

    # Step 7: Save results
    output_path = Path("data/production/predictions.csv")
    results_df.to_csv(output_path, index=False)

    # Step 8: Display summary
    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION RESULTS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total customers scored: {len(results_df)}")
    logger.info(f"Threshold used: {threshold:.4f}")
    logger.info(f"\nRisk Distribution:")
    risk_counts = results_df["risk_category"].value_counts()
    for cat in ["Critical", "High", "Medium", "Low"]:
        count = risk_counts.get(cat, 0)
        logger.info(f"  {cat:10s}: {count:3d} customers ({count/len(results_df)*100:.1f}%)")

    logger.info(f"\nAt-Risk customers (prob >= {threshold:.2f}): {results_df['is_at_risk'].sum()}")
    logger.info(f"Safe customers: {(~results_df['is_at_risk']).sum()}")

    logger.info(f"\n📊 Top 10 Highest Risk Customers:")
    logger.info("-" * 60)
    for _, row in results_df.head(10).iterrows():
        logger.info(
            f"  {row['customer_id']:20s} | {row['customer_name']:20s} | "
            f"Score: {row['risk_score']:3d}% | {row['risk_category']}"
        )

    logger.info(f"\n✅ Results saved to: {output_path}")
    logger.info(f"✅ Feature store saved to: {prod_fs_path}")
    logger.info(f"🚀 View in dashboard: streamlit run dashboard/app.py")
    logger.info(f"   → Navigate to '🚀 Production Predictions' page")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
