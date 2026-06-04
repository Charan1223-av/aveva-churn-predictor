"""
Generate a realistic production dataset WITHOUT renewal_status.
This simulates new/current customers whose churn outcome is UNKNOWN.

Run: python scripts/generate_production_dataset.py
Output: data/production/new_customers.csv (dim_customer format)
        + fact tables for feature engineering
"""
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

random.seed(42)
np.random.seed(42)

OUTPUT_DIR = Path("data/production")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_CUSTOMERS = 100

# --- Customer Profiles ---
industries = ["Oil & Gas", "Power Generation", "Marine", "Manufacturing",
              "Mining", "Water Treatment", "Food & Beverage", "Chemicals",
              "Pharmaceuticals", "Infrastructure"]
regions = ["North America", "EMEA", "APAC", "LATAM"]
countries = ["USA", "UK", "Germany", "Australia", "Brazil", "India",
             "Singapore", "Canada", "France", "Japan", "UAE", "Mexico"]
tiers = ["Enterprise", "Strategic", "Growth", "Standard"]
sizes = ["Large", "Mid-Market", "SMB"]
allocation_types = ["Annual", "Multi-Year", "Quarterly"]


def generate_customer_profile(i: int, risk_level: str) -> dict:
    """Generate a single customer with behavior matching risk_level."""
    cid = f"PROD-{risk_level[0].upper()}-{i:04d}"

    # Base contract info
    if risk_level == "high":
        nps = np.random.uniform(2, 5)
        contract_value = np.random.uniform(50000, 300000)
        num_renewals = np.random.randint(0, 2)
        tenure_days = np.random.randint(180, 600)
    elif risk_level == "medium":
        nps = np.random.uniform(5, 7)
        contract_value = np.random.uniform(100000, 500000)
        num_renewals = np.random.randint(1, 4)
        tenure_days = np.random.randint(365, 1200)
    else:  # low risk
        nps = np.random.uniform(7, 10)
        contract_value = np.random.uniform(200000, 800000)
        num_renewals = np.random.randint(3, 8)
        tenure_days = np.random.randint(730, 2500)

    named_users = np.random.randint(10, 200)
    contract_duration = random.choice([12, 24, 36])
    days_until_end = np.random.randint(30, 180)

    return {
        "customer_id": cid,
        "customer_name": f"Company_{cid}",
        "industry_vertical": random.choice(industries),
        "region": random.choice(regions),
        "country": random.choice(countries),
        "customer_tier": random.choice(tiers),
        "customer_size": random.choice(sizes),
        "contract_duration_months": contract_duration,
        "contract_value_usd": round(contract_value, 2),
        "annual_recurring_revenue": round(contract_value / (contract_duration / 12), 2),
        "num_previous_renewals": num_renewals,
        "nps_score": round(nps, 1),
        "is_multi_site": random.choice([True, False]),
        "named_users_licensed": named_users,
        "contract_start_date": pd.Timestamp("2024-06-01"),
        "contract_end_date": pd.Timestamp("2025-12-31") + pd.Timedelta(days=days_until_end),
        "first_contract_date": pd.Timestamp("2025-12-31") - pd.Timedelta(days=tenure_days),
        # NO renewal_status — this is production data!
    }


def generate_allocation_data(customer_id: str, risk_level: str, periods: int = 12) -> list:
    """Generate flex credit allocation history."""
    rows = []
    base_date = pd.Timestamp("2025-01-01")

    if risk_level == "high":
        util_start, util_end = 0.6, 0.25
    elif risk_level == "medium":
        util_start, util_end = 0.7, 0.55
    else:
        util_start, util_end = 0.75, 0.85

    total_credits = np.random.randint(5000, 50000)

    for m in range(periods):
        util = util_start + (util_end - util_start) * (m / periods) + np.random.uniform(-0.05, 0.05)
        util = max(0, min(1, util))
        credits_used = int(total_credits / 12 * util)
        expired = int(total_credits / 12 * (1 - util) * np.random.uniform(0, 0.5))

        rows.append({
            "customer_id": customer_id,
            "allocation_period": base_date + pd.DateOffset(months=m),
            "allocation_type": random.choice(allocation_types),
            "total_credits_purchased": total_credits,
            "credits_allocated": int(total_credits / 12),
            "credits_consumed": credits_used,
            "credits_expired": expired,
            "credits_rolled_over": max(0, int(total_credits / 12) - credits_used - expired),
            "credits_remaining": max(0, total_credits - credits_used * (m + 1)),
            "utilization_rate_period": round(util, 4),
            "top_up_credits": np.random.randint(0, 500) if random.random() < 0.2 else 0,
        })
    return rows


def generate_consumption_data(customer_id: str, risk_level: str, months: int = 12) -> list:
    """Generate monthly consumption data."""
    rows = []
    base_date = pd.Timestamp("2025-01-01")
    products = ["PI System", "AVEVA Insight", "Unified Operations", "Predictive Analytics",
                "Asset Performance", "Digital Twin", "Process Simulation"]

    if risk_level == "high":
        avg_credits, avg_users = 200, 5
    elif risk_level == "medium":
        avg_credits, avg_users = 500, 15
    else:
        avg_credits, avg_users = 1000, 30

    for m in range(months):
        n_products = random.randint(1, min(4, len(products))) if risk_level == "high" else random.randint(2, 6)
        used_products = random.sample(products, n_products)

        for prod in used_products:
            credits = max(0, int(avg_credits * np.random.uniform(0.5, 1.5)))
            rows.append({
                "customer_id": customer_id,
                "consumption_month": base_date + pd.DateOffset(months=m),
                "product_name": prod,
                "credits_consumed": credits,
                "distinct_users": max(1, int(avg_users * np.random.uniform(0.5, 1.5))),
                "error_count": np.random.randint(0, 5) if risk_level == "high" else np.random.randint(0, 2),
            })
    return rows


def generate_product_usage(customer_id: str, risk_level: str, months: int = 12) -> list:
    """Generate product usage/adoption data."""
    rows = []
    base_date = pd.Timestamp("2025-01-01")

    if risk_level == "high":
        adoption_start, adoption_end = 0.5, 0.25
    elif risk_level == "medium":
        adoption_start, adoption_end = 0.6, 0.55
    else:
        adoption_start, adoption_end = 0.7, 0.85

    for m in range(months):
        adoption = adoption_start + (adoption_end - adoption_start) * (m / months)
        adoption += np.random.uniform(-0.05, 0.05)
        adoption = max(0, min(1, adoption))
        module_adoption = adoption * np.random.uniform(0.7, 1.0)

        trend = "Declining" if risk_level == "high" else ("Stable" if risk_level == "medium" else "Growing")

        rows.append({
            "customer_id": customer_id,
            "snapshot_month": base_date + pd.DateOffset(months=m),
            "adoption_rate": round(adoption, 4),
            "module_adoption_pct": round(module_adoption, 4),
            "usage_trend_30d": trend,
        })
    return rows


def generate_support_tickets(customer_id: str, risk_level: str) -> list:
    """Generate support ticket history."""
    rows = []
    base_date = pd.Timestamp("2025-01-01")

    if risk_level == "high":
        n_tickets = np.random.randint(8, 20)
    elif risk_level == "medium":
        n_tickets = np.random.randint(3, 8)
    else:
        n_tickets = np.random.randint(0, 4)

    severities = ["P1", "P2", "P3", "P4"]
    sev_weights = {
        "high": [0.25, 0.30, 0.25, 0.20],
        "medium": [0.10, 0.20, 0.40, 0.30],
        "low": [0.05, 0.10, 0.35, 0.50],
    }

    for t in range(n_tickets):
        severity = np.random.choice(severities, p=sev_weights[risk_level])
        created = base_date + pd.Timedelta(days=np.random.randint(0, 350))
        resolved = created + pd.Timedelta(days=np.random.randint(1, 30))

        if risk_level == "high":
            csat = np.random.uniform(1, 3.5)
            sentiment = np.random.uniform(-1, 0)
            sla_met = random.random() < 0.5
        elif risk_level == "medium":
            csat = np.random.uniform(3, 4)
            sentiment = np.random.uniform(-0.3, 0.4)
            sla_met = random.random() < 0.75
        else:
            csat = np.random.uniform(4, 5)
            sentiment = np.random.uniform(0.2, 1)
            sla_met = random.random() < 0.95

        rows.append({
            "customer_id": customer_id,
            "ticket_id": f"TKT-{customer_id}-{t:03d}",
            "created_date": created,
            "resolved_date": resolved,
            "severity": severity,
            "sla_met": sla_met,
            "escalation_count": np.random.randint(0, 3) if risk_level == "high" else 0,
            "csat_score": round(csat, 2),
            "sentiment_score": round(sentiment, 3),
            "is_credit_related": random.random() < (0.4 if risk_level == "high" else 0.1),
        })
    return rows


def generate_engagement_data(customer_id: str, risk_level: str, months: int = 12) -> list:
    """Generate customer engagement data."""
    rows = []
    base_date = pd.Timestamp("2025-01-01")

    for m in range(months):
        if risk_level == "high":
            champion_score = max(0, 3 - m * 0.2 + np.random.uniform(-0.5, 0.5))
            days_since = min(90, 10 + m * 5 + np.random.randint(0, 10))
            training = 0 if random.random() < 0.7 else 1
            qbr = random.random() < 0.2
        elif risk_level == "medium":
            champion_score = max(0, 5 + np.random.uniform(-1, 1))
            days_since = np.random.randint(5, 30)
            training = 1 if random.random() < 0.4 else 0
            qbr = random.random() < 0.5
        else:
            champion_score = max(0, 7 + np.random.uniform(-1, 1.5))
            days_since = np.random.randint(1, 15)
            training = 1 if random.random() < 0.7 else 0
            qbr = random.random() < 0.85

        rows.append({
            "customer_id": customer_id,
            "engagement_month": base_date + pd.DateOffset(months=m),
            "training_sessions_attended": training,
            "qbr_participation": qbr,
            "executive_sponsor_engaged": random.random() < (0.2 if risk_level == "high" else 0.7),
            "documentation_page_views": np.random.randint(0, 10) if risk_level == "high" else np.random.randint(10, 100),
            "champion_activity_score": round(champion_score, 2),
            "days_since_last_interaction": days_since,
        })
    return rows


def main():
    print("=" * 60)
    print("Generating Production Dataset (NO renewal_status)")
    print("=" * 60)

    # Generate customers: 30 high risk, 30 medium, 40 low
    customers = []
    all_allocations = []
    all_consumption = []
    all_product_usage = []
    all_tickets = []
    all_engagement = []

    risk_distribution = [("high", 30), ("medium", 30), ("low", 40)]
    idx = 0

    for risk_level, count in risk_distribution:
        for i in range(count):
            idx += 1
            cust = generate_customer_profile(idx, risk_level)
            customers.append(cust)

            cid = cust["customer_id"]
            all_allocations.extend(generate_allocation_data(cid, risk_level))
            all_consumption.extend(generate_consumption_data(cid, risk_level))
            all_product_usage.extend(generate_product_usage(cid, risk_level))
            all_tickets.extend(generate_support_tickets(cid, risk_level))
            all_engagement.extend(generate_engagement_data(cid, risk_level))

    # Save CSVs
    dim_customers = pd.DataFrame(customers)
    dim_customers.to_csv(OUTPUT_DIR / "dim_customer.csv", index=False)
    print(f"✅ dim_customer.csv: {len(dim_customers)} customers (NO renewal_status)")

    pd.DataFrame(all_allocations).to_csv(OUTPUT_DIR / "fact_flex_allocation.csv", index=False)
    print(f"✅ fact_flex_allocation.csv: {len(all_allocations)} records")

    pd.DataFrame(all_consumption).to_csv(OUTPUT_DIR / "fact_flex_consumption.csv", index=False)
    print(f"✅ fact_flex_consumption.csv: {len(all_consumption)} records")

    pd.DataFrame(all_product_usage).to_csv(OUTPUT_DIR / "fact_product_usage.csv", index=False)
    print(f"✅ fact_product_usage.csv: {len(all_product_usage)} records")

    pd.DataFrame(all_tickets).to_csv(OUTPUT_DIR / "fact_support_ticket.csv", index=False)
    print(f"✅ fact_support_ticket.csv: {len(all_tickets)} records")

    pd.DataFrame(all_engagement).to_csv(OUTPUT_DIR / "fact_customer_engagement.csv", index=False)
    print(f"✅ fact_customer_engagement.csv: {len(all_engagement)} records")

    print(f"\n📁 All files saved to: {OUTPUT_DIR}/")
    print(f"\nExpected Risk Distribution:")
    print(f"  High Risk (>50% probability):   ~30 customers (PROD-H-*)")
    print(f"  Medium Risk (25-50%):           ~30 customers (PROD-M-*)")
    print(f"  Low Risk (<25%):                ~40 customers (PROD-L-*)")
    print(f"\n🚀 Next: python predict_production.py")


if __name__ == "__main__":
    main()
