"""
Data loading with schema validation.
"""
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, raw_path: str):
        self.raw_path = Path(raw_path)

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all CSV tables."""
        tables = {}
        file_map = {
            "dim_customer": "dim_customer.csv",
            "fact_flex_allocation": "fact_flex_allocation.csv",
            "fact_flex_consumption": "fact_flex_consumption.csv",
            "fact_product_usage": "fact_product_usage.csv",
            "fact_support_ticket": "fact_support_ticket.csv",
            "fact_customer_engagement": "fact_customer_engagement.csv",
        }

        for table_name, filename in file_map.items():
            path = self.raw_path / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing table: {path}")
            df = pd.read_csv(path, low_memory=False)
            df = self._parse_dates(df, table_name)
            tables[table_name] = df
            logger.info(f"Loaded {table_name}: {df.shape}")

        return tables

    def _parse_dates(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        date_cols_map = {
            "dim_customer": ["contract_start_date", "contract_end_date", "first_contract_date"],
            "fact_flex_allocation": ["allocation_period"],
            "fact_flex_consumption": ["consumption_month"],
            "fact_product_usage": ["snapshot_month"],
            "fact_support_ticket": ["created_date", "resolved_date"],
            "fact_customer_engagement": ["engagement_month"],
        }
        for col in date_cols_map.get(table_name, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df
