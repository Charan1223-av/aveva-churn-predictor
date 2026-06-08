"""
Shared feature policy constants for engineering, training and inference.
"""

PRIORITIZED_FEATURES = [
    "utilization_rate_period",
    "credits_expired",
    "avg_session_duration_min",
    "module_adoption_pct",
    "usage_trend_30d",
    "resolution_time_hours",
    "escalation_count",
    "sla_met",
]

EXCLUDED_FEATURE_TOKENS = ("nps", "training")
EXCLUDED_FEATURES = {
    "nps_score",
    "training_sessions_attended",
    "training_users_participating",
    "total_training_sessions",
}

USAGE_TREND_ORDINAL_MAP = {
    "Declining": 0,
    "Stable": 1,
    "Growing": 2,
    "New": 3,
}
