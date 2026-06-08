"""
Modified FeatureStore that supports BOTH training (with renewal_status)
and production inference (without renewal_status).

This replaces src/features/feature_store.py with backward-compatible changes.
"""
import logging
import numpy as np
import pandas as pd
from scipy.stats import linregress
from typing import Dict
from src.features.feature_policy import USAGE_TREND_ORDINAL_MAP

logger = logging.getLogger(__name__)
SNAPSHOT_DATE = pd.Timestamp("2025-12-31")


def _trend_slope(series: pd.Series) -> float:
    if len(series) < 3:
        return 0.0
    x = np.arange(len(series))
    try:
        slope, _, _, _, _ = linregress(x, series.values.astype(float))
        return float(slope)
    except Exception:
        return 0.0


class FeatureStore:
    def __init__(self, tables: Dict[str, pd.DataFrame]):
        self.tables = tables

    def build(self) -> pd.DataFrame:
        logger.info("Building feature store...")
        base = self._base_customer_features()
        alloc = self._allocation_features()
        cons = self._consumption_features()
        prod = self._product_features()
        supp = self._support_features()
        eng = self._engagement_features()

        fs = base
        for df in [alloc, cons, prod, supp, eng]:
            fs = fs.merge(df, on="customer_id", how="left")

        # Fill NAs for customers with no tickets / no engagement data
        fs = fs.fillna(0)
        fs["utilization_x_module_adoption"] = (
            fs.get("utilization_rate_period", 0) * fs.get("module_adoption_pct", 0)
        )
        fs["escalation_x_sla_met"] = fs.get("escalation_count", 0) * fs.get("sla_met", 0)
        logger.info(f"Feature store shape: {fs.shape}")

        # Log whether this is training or production mode
        if "renewal_status" in fs.columns:
            logger.info("Mode: TRAINING (renewal_status present)")
        else:
            logger.info("Mode: PRODUCTION/INFERENCE (renewal_status NOT present)")

        return fs

    def _base_customer_features(self) -> pd.DataFrame:
        df = self.tables["dim_customer"].copy()
        df["days_until_contract_end"] = (df["contract_end_date"] - SNAPSHOT_DATE).dt.days
        df["customer_tenure_days"] = (SNAPSHOT_DATE - df["first_contract_date"]).dt.days
        df["contract_value_per_user"] = df["contract_value_usd"] / df["named_users_licensed"].clip(lower=1)
        df["is_multi_site"] = df["is_multi_site"].map({"True": 1, "False": 0, True: 1, False: 0}).fillna(0).astype(int)

        keep = [
            "customer_id", "customer_name", "industry_vertical", "region", "country",
            "customer_tier", "customer_size", "contract_duration_months",
            "contract_value_usd", "annual_recurring_revenue", "num_previous_renewals",
            "is_multi_site", "named_users_licensed",
            "days_until_contract_end", "customer_tenure_days", "contract_value_per_user",
        ]

        # Include renewal_status ONLY if present (training mode)
        if "renewal_status" in df.columns:
            keep.append("renewal_status")

        return df[[c for c in keep if c in df.columns]]

    def _allocation_features(self) -> pd.DataFrame:
        df = self.tables["fact_flex_allocation"].copy()
        df = df.sort_values(["customer_id", "allocation_period"])

        agg = df.groupby("customer_id").agg(
            utilization_rate_period=("utilization_rate_period", "mean"),
            avg_utilization_rate=("utilization_rate_period", "mean"),
            min_utilization_rate=("utilization_rate_period", "min"),
            std_utilization_rate=("utilization_rate_period", "std"),
            credits_expired=("credits_expired", "sum"),
            credits_expired_total=("credits_expired", "sum"),
            credits_rolled_over_total=("credits_rolled_over", "sum"),
            top_up_count=("top_up_credits", lambda x: (x > 0).sum()),
            top_up_total=("top_up_credits", "sum"),
            utilization_last=("utilization_rate_period", "last"),
            credits_remaining_last=("credits_remaining", "last"),
            allocation_type_last=("allocation_type", "last"),
            total_credits_purchased=("total_credits_purchased", "first"),
        ).reset_index()

        trend = (df.groupby("customer_id")
                   .apply(lambda g: _trend_slope(g["utilization_rate_period"]))
                   .reset_index().rename(columns={0: "utilization_trend_slope"}))
        agg = agg.merge(trend, on="customer_id", how="left")

        def recency_ratio(g):
            v = g["utilization_rate_period"].values
            if len(v) < 6:
                return 1.0
            return v[-3:].mean() / (v[:3].mean() + 1e-9)

        rr = (df.groupby("customer_id").apply(recency_ratio)
                .reset_index().rename(columns={0: "utilization_recency_ratio"}))
        agg = agg.merge(rr, on="customer_id", how="left")

        roll3 = (
            df.groupby("customer_id")["utilization_rate_period"]
            .apply(lambda s: s.rolling(window=3, min_periods=1).mean().iloc[-1])
            .reset_index()
            .rename(columns={"utilization_rate_period": "utilization_rate_period_roll3m"})
        )
        roll6 = (
            df.groupby("customer_id")["utilization_rate_period"]
            .apply(lambda s: s.rolling(window=6, min_periods=1).mean().iloc[-1])
            .reset_index()
            .rename(columns={"utilization_rate_period": "utilization_rate_period_roll6m"})
        )
        roc = (
            df.groupby("customer_id")["credits_expired"]
            .apply(
                lambda s: 0.0
                if len(s) < 2
                else float((s.iloc[-1] - s.iloc[0]) / (abs(s.iloc[0]) + 1e-9))
            )
            .reset_index()
            .rename(columns={"credits_expired": "credits_expired_rate_change"})
        )
        agg = agg.merge(roll3, on="customer_id", how="left")
        agg = agg.merge(roll6, on="customer_id", how="left")
        agg = agg.merge(roc, on="customer_id", how="left")
        agg["credit_expiry_rate"] = agg["credits_expired_total"] / agg["total_credits_purchased"].clip(lower=1)
        return agg

    def _consumption_features(self) -> pd.DataFrame:
        df = self.tables["fact_flex_consumption"].copy()
        df = df.sort_values(["customer_id", "consumption_month"])
        if "avg_session_duration_min" not in df.columns:
            df["avg_session_duration_min"] = 0.0

        monthly = df.groupby(["customer_id", "consumption_month"]).agg(
            total_credits=("credits_consumed", "sum"),
            distinct_users=("distinct_users", "sum"),
            products_used=("product_name", "nunique"),
            error_count=("error_count", "sum"),
            avg_session_duration_min=("avg_session_duration_min", "mean"),
        ).reset_index()

        agg = monthly.groupby("customer_id").agg(
            avg_monthly_credits=("total_credits", "mean"),
            std_monthly_credits=("total_credits", "std"),
            avg_distinct_users=("distinct_users", "mean"),
            avg_products_used=("products_used", "mean"),
            product_breadth_last=("products_used", "last"),
            total_errors=("error_count", "sum"),
            avg_session_duration_min=("avg_session_duration_min", "mean"),
            zero_consumption_months=("total_credits", lambda x: (x == 0).sum()),
        ).reset_index()

        ct = (monthly.groupby("customer_id")
                     .apply(lambda g: _trend_slope(g["total_credits"]))
                     .reset_index().rename(columns={0: "consumption_trend_slope"}))
        agg = agg.merge(ct, on="customer_id", how="left")

        ut = (monthly.groupby("customer_id")
                     .apply(lambda g: _trend_slope(g["distinct_users"]))
                     .reset_index().rename(columns={0: "active_users_trend"}))
        agg = agg.merge(ut, on="customer_id", how="left")

        def crr(g):
            v = g["total_credits"].values
            if len(v) < 6:
                return 1.0
            return v[-3:].mean() / (v[:3].mean() + 1e-9)

        cr = (monthly.groupby("customer_id").apply(crr)
                     .reset_index().rename(columns={0: "consumption_recency_ratio"}))
        agg = agg.merge(cr, on="customer_id", how="left")
        return agg

    def _product_features(self) -> pd.DataFrame:
        df = self.tables["fact_product_usage"].copy()
        df = df.sort_values(["customer_id", "snapshot_month"])
        if "usage_trend_30d" not in df.columns:
            df["usage_trend_30d"] = "Stable"

        monthly = df.groupby(["customer_id", "snapshot_month"]).agg(
            avg_adoption_rate=("adoption_rate", "mean"),
            module_adoption_pct=("module_adoption_pct", "mean"),
            usage_trend_30d=("usage_trend_30d", "last"),
        ).reset_index()
        monthly["usage_trend_30d"] = (
            monthly["usage_trend_30d"]
            .map(USAGE_TREND_ORDINAL_MAP)
            .fillna(USAGE_TREND_ORDINAL_MAP["Stable"])
            .astype(int)
        )

        agg = monthly.groupby("customer_id").agg(
            avg_adoption_rate=("avg_adoption_rate", "mean"),
            module_adoption_pct=("module_adoption_pct", "mean"),
            adoption_rate_last=("avg_adoption_rate", "last"),
            module_adoption_last=("module_adoption_pct", "last"),
            usage_trend_30d=("usage_trend_30d", "last"),
        ).reset_index()

        at = (monthly.groupby("customer_id")
                     .apply(lambda g: _trend_slope(g["avg_adoption_rate"]))
                     .reset_index().rename(columns={0: "adoption_rate_trend"}))
        agg = agg.merge(at, on="customer_id", how="left")

        mt = (monthly.groupby("customer_id")
                     .apply(lambda g: _trend_slope(g["module_adoption_pct"]))
                     .reset_index().rename(columns={0: "module_adoption_trend"}))
        agg = agg.merge(mt, on="customer_id", how="left")
        agg["is_declining_last"] = (agg["usage_trend_30d"] == 0).astype(int)
        return agg

    def _support_features(self) -> pd.DataFrame:
        df = self.tables["fact_support_ticket"].copy()
        df = df.sort_values(["customer_id", "created_date"])
        df["sla_met"] = df["sla_met"].map({"True": True, "False": False, True: True, False: False})
        df["is_credit_related"] = df["is_credit_related"].map({"True": True, "False": False, True: True, False: False})
        if "resolution_time_hours" not in df.columns:
            resolved = pd.to_datetime(df.get("resolved_date"), errors="coerce")
            created = pd.to_datetime(df.get("created_date"), errors="coerce")
            df["resolution_time_hours"] = (resolved - created).dt.total_seconds().div(3600)

        cutoff_3m = SNAPSHOT_DATE - pd.Timedelta(days=90)
        df_recent = df[df["created_date"] >= cutoff_3m]

        overall = df.groupby("customer_id").agg(
            total_tickets=("ticket_id", "count"),
            p1_count=("severity", lambda x: (x == "P1").sum()),
            p2_count=("severity", lambda x: (x == "P2").sum()),
            sla_breach_rate=("sla_met", lambda x: (~x).mean()),
            escalation_count=("escalation_count", "sum"),
            total_escalations=("escalation_count", "sum"),
            sla_met=("sla_met", lambda x: x.astype(bool).mean()),
            resolution_time_hours=("resolution_time_hours", "mean"),
            avg_csat_score=("csat_score", "mean"),
            avg_sentiment_score=("sentiment_score", "mean"),
            credit_related_tickets=("is_credit_related", lambda x: x.astype(bool).sum()),
            negative_sentiment_count=("sentiment_score", lambda x: (x < 0).sum()),
        ).reset_index()

        recent = df_recent.groupby("customer_id").agg(
            p1p2_tickets_last3m=("severity", lambda x: x.isin(["P1", "P2"]).sum()),
            avg_csat_last3m=("csat_score", "mean"),
            avg_sentiment_last3m=("sentiment_score", "mean"),
        ).reset_index()

        result = overall.merge(recent, on="customer_id", how="left").fillna(0)
        return result

    def _engagement_features(self) -> pd.DataFrame:
        df = self.tables["fact_customer_engagement"].copy()
        df = df.sort_values(["customer_id", "engagement_month"])
        df["qbr_participation"] = df["qbr_participation"].map({"True": True, "False": False, True: True, False: False})
        df["executive_sponsor_engaged"] = df["executive_sponsor_engaged"].map({"True": True, "False": False, True: True, False: False})

        agg = df.groupby("customer_id").agg(
            qbr_attendance_rate=("qbr_participation", lambda x: x.astype(bool).mean()),
            exec_sponsor_rate=("executive_sponsor_engaged", lambda x: x.astype(bool).mean()),
            avg_doc_views=("documentation_page_views", "mean"),
            avg_champion_score=("champion_activity_score", "mean"),
            champion_score_last=("champion_activity_score", "last"),
            days_since_last_interaction_last=("days_since_last_interaction", "last"),
            max_days_dark=("days_since_last_interaction", "max"),
        ).reset_index()

        ct = (df.groupby("customer_id")
                .apply(lambda g: _trend_slope(g["champion_activity_score"]))
                .reset_index().rename(columns={0: "champion_score_trend"}))
        agg = agg.merge(ct, on="customer_id", how="left")
        agg["gone_dark_flag"] = (agg["days_since_last_interaction_last"] > 60).astype(int)
        return agg
