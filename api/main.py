"""
FastAPI application for churn prediction.
Supports both training data and production data (without renewal_status).
"""
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

model_state: Dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model...")
    try:
        model_path = Path("models/latest/pipeline.joblib")
        meta_path = Path("models/latest/metadata.json")
        config_path = Path("config/config.yaml")

        if not model_path.exists():
            logger.warning("No trained model found. Train first with: python train.py")
        else:
            model_state["pipeline"] = joblib.load(model_path)
            with open(meta_path) as f:
                model_state["metadata"] = json.load(f)
            logger.info(f"Model loaded: {model_state['metadata']['version']}")

        with open(config_path) as f:
            model_state["config"] = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Startup error: {e}")
    yield
    model_state.clear()


app = FastAPI(
    title="AVEVA Churn Prediction API",
    description="Predicts renewal risk for AVEVA Flex Credit customers.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChurnRequest(BaseModel):
    customer_id: str
    features: Dict[str, Any] = Field(..., description="Customer feature values")


class ChurnResponse(BaseModel):
    customer_id: str
    churn_probability: float
    risk_category: str
    is_at_risk: bool
    threshold_used: float
    model_version: str
    inference_time_ms: float


class BatchRequest(BaseModel):
    customers: List[ChurnRequest]


def prob_to_risk(prob: float) -> str:
    if prob < 0.25: return "Low"
    elif prob < 0.50: return "Medium"
    elif prob < 0.75: return "High"
    else: return "Critical"


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": "pipeline" in model_state,
        "model_version": model_state.get("metadata", {}).get("version", "N/A"),
    }


@app.get("/model/info")
async def model_info():
    if "metadata" not in model_state:
        raise HTTPException(status_code=404, detail="No model loaded. Run train.py first.")
    return model_state["metadata"]


@app.post("/predict", response_model=ChurnResponse)
async def predict(request: ChurnRequest):
    if "pipeline" not in model_state:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Run: python train.py")
    start = time.time()
    try:
        X = pd.DataFrame([request.features])
        prob = float(model_state["pipeline"].predict_proba(X)[0, 1])
        threshold = model_state["config"]["model"]["threshold"]
        return ChurnResponse(
            customer_id=request.customer_id,
            churn_probability=round(prob, 4),
            risk_category=prob_to_risk(prob),
            is_at_risk=prob >= threshold,
            threshold_used=threshold,
            model_version=model_state["metadata"]["version"],
            inference_time_ms=round((time.time() - start) * 1000, 2),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(request: BatchRequest):
    results = []
    for c in request.customers:
        r = await predict(c)
        results.append(r)
    return {"predictions": results, "count": len(results)}


@app.get("/customers")
async def list_customers():
    """List all customers with their risk scores from the feature store (training data)."""
    try:
        fs_path = Path("data/processed/feature_store.parquet")
        if not fs_path.exists():
            raise HTTPException(status_code=404,
                                detail="Feature store not found. Run train.py first.")
        fs = pd.read_parquet(fs_path)
        config = model_state.get("config", {})
        drop_cols = config.get("features", {}).get("drop_columns", []) + \
                    ["customer_name", "renewal_status"]

        results = []
        for _, row in fs.iterrows():
            cid = row.get("customer_id", "unknown")
            feat_dict = row.drop(
                labels=[c for c in drop_cols if c in row.index], errors="ignore"
            ).to_dict()

            prob = 0.0
            if "pipeline" in model_state:
                try:
                    X = pd.DataFrame([feat_dict])
                    prob = float(model_state["pipeline"].predict_proba(X)[0, 1])
                except Exception:
                    pass

            results.append({
                "customer_id": cid,
                "customer_name": row.get("customer_name", ""),
                "renewal_status": row.get("renewal_status", ""),
                "churn_probability": round(prob, 4),
                "risk_category": prob_to_risk(prob),
                "source": "training",
            })

        return {"customers": results, "count": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/customers/production")
async def list_production_customers():
    """List production customers with predictions (no renewal_status)."""
    try:
        # Option 1: Read pre-computed predictions
        pred_path = Path("data/production/predictions.csv")
        if pred_path.exists():
            df = pd.read_csv(pred_path)
            results = df.to_dict(orient="records")
            for r in results:
                r["source"] = "production"
                r["renewal_status"] = "Unknown (Predicted)"
            return {"customers": results, "count": len(results)}

        # Option 2: Score on-the-fly from production feature store
        fs_path = Path("data/production/feature_store.parquet")
        if not fs_path.exists():
            raise HTTPException(
                status_code=404,
                detail="No production data found. Run: python scripts/generate_production_dataset.py && python predict_production.py"
            )

        fs = pd.read_parquet(fs_path)
        config = model_state.get("config", {})
        drop_cols = config.get("features", {}).get("drop_columns", []) + ["customer_name"]

        results = []
        for _, row in fs.iterrows():
            cid = row.get("customer_id", "unknown")
            feat_dict = row.drop(
                labels=[c for c in drop_cols if c in row.index], errors="ignore"
            ).to_dict()

            prob = 0.0
            if "pipeline" in model_state:
                try:
                    X = pd.DataFrame([feat_dict])
                    prob = float(model_state["pipeline"].predict_proba(X)[0, 1])
                except Exception:
                    pass

            results.append({
                "customer_id": cid,
                "customer_name": row.get("customer_name", ""),
                "renewal_status": "Unknown (Predicted)",
                "churn_probability": round(prob, 4),
                "risk_category": prob_to_risk(prob),
                "source": "production",
            })

        return {"customers": results, "count": len(results)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/customers/all")
async def list_all_customers():
    """Combine training and production customers."""
    training = []
    production = []

    try:
        t = await list_customers()
        training = t.get("customers", [])
    except Exception:
        pass

    try:
        p = await list_production_customers()
        production = p.get("customers", [])
    except Exception:
        pass

    combined = training + production
    return {"customers": combined, "count": len(combined)}
