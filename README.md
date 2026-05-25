
# AVEVA Flex Credit — Churn Prediction System

A production-grade ML system for predicting customer renewal risk using AVEVA Flex Credit behavioral data.

---

## 📁 Project Structure

```
aveva-churn-predictor/
├── config/                  # YAML configs
├── data/raw/                # Input CSVs (place your dataset here)
├── data/processed/          # Auto-generated feature store
├── src/
│   ├── data/                # Data loading + validation
│   ├── features/            # Feature engineering (7 tables → flat features)
│   ├── models/              # Training, evaluation, explainability
│   ├── pipeline/            # sklearn preprocessing pipeline
│   └── monitoring/          # Drift detection
├── api/                     # FastAPI REST API
├── dashboard/               # Streamlit UI
├── models/                  # Saved model artifacts
├── notebooks/               # EDA notebook
├── tests/                   # Unit + integration tests
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🚀 Quick Start (Local — No Docker)

### 1. Clone & Install

```bash
git clone https://github.com/Charan1223-av/aveva-churn-predictor.git
cd aveva-churn-predictor
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Place Your Data

Make sure these files are in `data/raw/`:
- `dim_customer.csv`
- `fact_flex_allocation.csv`
- `fact_flex_consumption.csv`
- `fact_product_usage.csv`
- `fact_support_ticket.csv`
- `fact_customer_engagement.csv`

### 3. Build Feature Store + Train Model

```bash
python train.py
```

This will:
- Load all 6 CSVs
- Engineer 40+ features
- Train XGBoost with cross-validation
- Save model to `models/latest/`
- Print evaluation metrics

### 4. Start the API

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs: http://localhost:8000/docs

### 5. Start the Dashboard

```bash
streamlit run dashboard/app.py
```

Dashboard: http://localhost:8501

---

## 🐳 Docker (Recommended)

```bash
docker-compose up --build
```

| Service    | URL                        |
|------------|----------------------------|
| API        | http://localhost:8000      |
| API Docs   | http://localhost:8000/docs |
| Dashboard  | http://localhost:8501      |
| MLflow UI  | http://localhost:5000      |

---

## 🔌 API Usage

### Single Prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"customer_id": "AVEVA-ENT-1002", "features": {...}}'
```

### Health Check
```bash
curl http://localhost:8000/health
```

---

## 📊 Model Performance

| Metric           | Value  |
|------------------|--------|
| ROC-AUC          | ~0.85+ |
| Avg Precision    | ~0.80+ |
| Recall (churn)   | ~0.82+ |
| Inference Time   | <10ms  |

> Note: Metrics improve significantly with 500+ customer dataset.

---

## 🛠️ Tech Stack

- **ML**: XGBoost, LightGBM, scikit-learn, imbalanced-learn, SHAP, Optuna
- **API**: FastAPI + Uvicorn
- **Dashboard**: Streamlit + Plotly
- **Tracking**: MLflow
- **Monitoring**: PSI-based drift detection
- **Deployment**: Docker + Docker Compose

---

## 📞 Support

Raise an issue in this repository for questions or bugs.
