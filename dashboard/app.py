"""
Streamlit Dashboard for AVEVA Churn Intelligence.
Run: streamlit run dashboard/app.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(
    page_title="AVEVA Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://localhost:8000"

st.markdown("""
<style>
    .risk-critical { color: #ef4444; font-weight: bold; }
    .risk-high     { color: #f97316; font-weight: bold; }
    .risk-medium   { color: #eab308; font-weight: bold; }
    .risk-low      { color: #22c55e; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=60)
def load_feature_store() -> pd.DataFrame:
    path = Path("data/processed/feature_store.parquet")
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame()


def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def get_all_predictions() -> list:
    try:
        r = requests.get(f"{API_URL}/customers", timeout=10)
        if r.status_code == 200:
            return r.json().get("customers", [])
    except Exception:
        pass
    return []


def predict_single(customer_id: str, features: dict) -> dict:
    try:
        r = requests.post(
            f"{API_URL}/predict",
            json={"customer_id": customer_id, "features": features},
            timeout=10,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        st.error(f"API Error: {e}")
    return {}


# ── Sidebar ────────────────────────────────────────────
with st.sidebar:
    st.title("AVEVA Churn AI")
    st.markdown("---")
    api_ok = check_api_health()
    if api_ok:
        st.success("✅ API Online")
    else:
        st.error("❌ API Offline")
        st.code("python -m uvicorn api.main:app --port 8000")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔮 Live Predict", "📈 Customer Deep Dive", "⚙️ Model Info"],
    )


fs = load_feature_store()


# ══════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 AVEVA Flex Credit — Churn Intelligence")
    st.markdown("Real-time renewal risk monitoring across your customer portfolio.")
    st.markdown("---")

    if fs.empty:
        st.warning("⚠️ No data found. Run `python train.py` first to build the feature store.")
        st.stop()

    total = len(fs)
    churned = fs["renewal_status"].isin(["Churned", "At Risk"]).sum() if "renewal_status" in fs.columns else 0
    renewed = fs["renewal_status"].isin(["Renewed", "On Track"]).sum() if "renewal_status" in fs.columns else 0
    avg_nps = fs["nps_score"].mean() if "nps_score" in fs.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", total)
    c2.metric("At-Risk / Churned", int(churned), delta=f"{churned/total*100:.0f}%", delta_color="inverse")
    c3.metric("Healthy Customers", int(renewed))
    c4.metric("Avg NPS Score", f"{avg_nps:.1f}")
    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        if "renewal_status" in fs.columns:
            status_counts = fs["renewal_status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig = px.pie(
                status_counts, values="Count", names="Status",
                color="Status",
                color_discrete_map={
                    "Churned": "#ef4444", "At Risk": "#f97316",
                    "On Track": "#3b82f6", "Renewed": "#22c55e",
                },
                title="Customer Renewal Status Distribution",
                hole=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        if "nps_score" in fs.columns and "renewal_status" in fs.columns:
            fig2 = px.box(
                fs, x="renewal_status", y="nps_score", color="renewal_status",
                color_discrete_map={
                    "Churned": "#ef4444", "At Risk": "#f97316",
                    "On Track": "#3b82f6", "Renewed": "#22c55e",
                },
                title="NPS Score by Renewal Status",
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 All Customers — Risk Scores")
    if api_ok:
        with st.spinner("Fetching risk scores from API..."):
            preds = get_all_predictions()
        if preds:
            pred_df = pd.DataFrame(preds).sort_values("churn_probability", ascending=False)

            def color_risk(val):
                colors = {
                    "Critical": "background-color: #fee2e2",
                    "High": "background-color: #ffedd5",
                    "Medium": "background-color: #fef9c3",
                    "Low": "background-color: #dcfce7",
                }
                return colors.get(val, "")

            styled = pred_df.style.applymap(color_risk, subset=["risk_category"])
            st.dataframe(styled, use_container_width=True, height=400)
        else:
            st.info("No predictions available yet.")
    else:
        st.warning("API is offline. Start the API to see live risk scores.")
        if "renewal_status" in fs.columns:
            display_cols = [c for c in ["customer_id", "customer_name", "renewal_status", "nps_score"] if c in fs.columns]
            st.dataframe(fs[display_cols].fillna("N/A"), use_container_width=True)

    if "avg_utilization_rate" in fs.columns and "renewal_status" in fs.columns:
        st.markdown("---")
        st.subheader("📉 Utilization vs Champion Score")
        y_col = "champion_score_last" if "champion_score_last" in fs.columns else "avg_utilization_rate"
        hover_cols = [c for c in ["customer_id", "customer_name"] if c in fs.columns]
        fig3 = px.scatter(
            fs, x="avg_utilization_rate", y=y_col,
            color="renewal_status",
            hover_data=hover_cols,
            size="contract_value_usd" if "contract_value_usd" in fs.columns else None,
            color_discrete_map={
                "Churned": "#ef4444", "At Risk": "#f97316",
                "On Track": "#3b82f6", "Renewed": "#22c55e",
            },
            title="Utilization Rate vs Champion Score (bubble size = contract value)",
        )
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════
# PAGE 2 — LIVE PREDICT
# ══════════════════════════════════════════════════════
elif page == "🔮 Live Predict":
    st.title("🔮 Live Churn Prediction")
    st.markdown("Select a customer from the feature store and get an instant prediction.")
    st.markdown("---")

    if fs.empty:
        st.warning("No feature store found. Run `python train.py` first.")
        st.stop()
    if not api_ok:
        st.error("API is offline. Start it: `python -m uvicorn api.main:app --port 8000`")
        st.stop()

    customer_options = fs["customer_id"].tolist() if "customer_id" in fs.columns else []
    if not customer_options:
        st.warning("No customers in feature store.")
        st.stop()

    def fmt(cid):
        name = ""
        if "customer_name" in fs.columns:
            row = fs[fs["customer_id"] == cid]
            if not row.empty:
                name = row.iloc[0].get("customer_name", "")
        return f"{cid} — {name}" if name else cid

    selected = st.selectbox("Select Customer", customer_options, format_func=fmt)
    row = fs[fs["customer_id"] == selected].iloc[0]

    drop_cols = ["customer_id", "customer_name", "renewal_status",
                 "contract_start_date", "contract_end_date", "first_contract_date"]
    feat_dict = {}
    for k, v in row.drop(labels=[c for c in drop_cols if c in row.index], errors="ignore").items():
        if not pd.isna(v):
            if isinstance(v, bool):
                feat_dict[k] = int(v)
            elif pd.api.types.is_float_dtype(type(v)):
                feat_dict[k] = float(v)
            else:
                feat_dict[k] = str(v)

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Customer Profile")
        profile_cols = ["customer_name", "industry_vertical", "region",
                        "customer_tier", "renewal_status", "nps_score", "contract_value_usd"]
        for c in profile_cols:
            if c in row.index and not pd.isna(row[c]):
                st.write(f"**{c.replace('_', ' ').title()}:** {row[c]}")

    with col2:
        if st.button("🚀 Predict Churn Risk", type="primary"):
            with st.spinner("Running prediction..."):
                result = predict_single(selected, feat_dict)

            if result:
                prob = result["churn_probability"]
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    domain={"x": [0, 1], "y": [0, 1]},
                    title={"text": "Churn Probability (%)", "font": {"size": 18}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#1e3a5f"},
                        "steps": [
                            {"range": [0, 25], "color": "#dcfce7"},
                            {"range": [25, 50], "color": "#fef9c3"},
                            {"range": [50, 75], "color": "#ffedd5"},
                            {"range": [75, 100], "color": "#fee2e2"},
                        ],
                        "threshold": {
                            "line": {"color": "red", "width": 4},
                            "thickness": 0.75,
                            "value": result["threshold_used"] * 100,
                        },
                    },
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)

                r1, r2, r3 = st.columns(3)
                r1.metric("Probability", f"{prob:.1%}")
                r2.metric("Risk Category", result["risk_category"])
                r3.metric("Decision", "⚠️ AT RISK" if result["is_at_risk"] else "✅ SAFE")

                if result["is_at_risk"]:
                    st.error("🚨 High churn risk detected. Immediate CSM intervention recommended.")
                else:
                    st.success("✅ Customer shows healthy engagement signals.")


# ══════════════════════════════════════════════════════
# PAGE 3 — CUSTOMER DEEP DIVE
# ══════════════════════════════════════════════════════
elif page == "📈 Customer Deep Dive":
    st.title("📈 Customer Deep Dive")
    st.markdown("---")

    if fs.empty:
        st.warning("No feature store found. Run `python train.py` first.")
        st.stop()

    selected = st.selectbox("Select Customer", fs["customer_id"].tolist())
    row = fs[fs["customer_id"] == selected].iloc[0]
    name = row.get("customer_name", selected) if "customer_name" in row.index else selected
    st.subheader(f"Customer: {name}")

    tab1, tab2, tab3, tab4 = st.tabs(["💳 Credits", "👥 Engagement", "🎫 Support", "📦 Product"])

    with tab1:
        credit_cols = [c for c in ["avg_utilization_rate", "utilization_trend_slope",
                                    "credit_expiry_rate", "utilization_recency_ratio",
                                    "credits_remaining_last", "top_up_count"] if c in row.index]
        if credit_cols:
            data = {c.replace("_", " ").title(): round(float(row[c]), 4)
                    for c in credit_cols if not pd.isna(row[c])}
            st.plotly_chart(
                px.bar(x=list(data.keys()), y=list(data.values()),
                       title="Credit Health Metrics", color_discrete_sequence=["#3b82f6"]),
                use_container_width=True,
            )

    with tab2:
        eng_cols = [c for c in ["avg_champion_score", "champion_score_last",
                                  "qbr_attendance_rate", "days_since_last_interaction_last",
                                  "gone_dark_flag", "total_training_sessions"] if c in row.index]
        if eng_cols:
            data = {c.replace("_", " ").title(): round(float(row[c]), 4)
                    for c in eng_cols if not pd.isna(row[c])}
            st.plotly_chart(
                px.bar(x=list(data.keys()), y=list(data.values()),
                       title="Engagement Metrics", color_discrete_sequence=["#8b5cf6"]),
                use_container_width=True,
            )
            if row.get("gone_dark_flag", 0) == 1:
                st.warning("⚠️ Customer has gone dark (>60 days since last interaction)")

    with tab3:
        supp_cols = [c for c in ["total_tickets", "p1_count", "p2_count", "sla_breach_rate",
                                   "avg_csat_score", "avg_sentiment_score",
                                   "credit_related_tickets"] if c in row.index]
        if supp_cols:
            data = {c.replace("_", " ").title(): round(float(row[c]), 4)
                    for c in supp_cols if not pd.isna(row[c])}
            st.plotly_chart(
                px.bar(x=list(data.keys()), y=list(data.values()),
                       title="Support Health Metrics", color_discrete_sequence=["#ef4444"]),
                use_container_width=True,
            )

    with tab4:
        prod_cols = [c for c in ["avg_adoption_rate", "adoption_rate_last",
                                   "adoption_rate_trend", "module_adoption_last",
                                   "product_breadth_last"] if c in row.index]
        if prod_cols:
            data = {c.replace("_", " ").title(): round(float(row[c]), 4)
                    for c in prod_cols if not pd.isna(row[c])}
            st.plotly_chart(
                px.bar(x=list(data.keys()), y=list(data.values()),
                       title="Product Adoption Metrics", color_discrete_sequence=["#22c55e"]),
                use_container_width=True,
            )


# ══════════════════════════════════════════════════════
# PAGE 4 — MODEL INFO
# ══════════════════════════════════════════════════════
elif page == "⚙️ Model Info":
    st.title("⚙️ Model Information")
    st.markdown("---")

    meta_path = Path("models/latest/metadata.json")
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Model Metadata")
            st.json({
                "Version": meta.get("version"),
                "Algorithm": meta.get("algorithm"),
                "Train Samples": meta.get("train_samples"),
                "Test Samples": meta.get("test_samples"),
                "Feature Count": meta.get("feature_count"),
            })

        with col2:
            st.subheader("Test Metrics")
            metrics = meta.get("metrics", {})
            for k, v in metrics.items():
                if isinstance(v, float):
                    st.metric(k.replace("test_", "").replace("_", " ").upper(), f"{v:.4f}")

        metric_items = {k: v for k, v in metrics.items()
                        if isinstance(v, float) and k != "threshold"}
        if metric_items:
            fig = px.bar(
                x=list(metric_items.keys()), y=list(metric_items.values()),
                title="Model Performance Metrics",
                color_discrete_sequence=["#3b82f6"],
            )
            fig.update_yaxes(range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        features = meta.get("features", [])
        if features:
            st.subheader(f"Features Used ({len(features)})")
            st.dataframe(pd.DataFrame({"Feature": features}), use_container_width=True, height=300)
    else:
        st.warning("No trained model found. Run `python train.py` first.")
        st.info("After training, this page will show model metrics, feature list, and performance charts.")
