"""Companion package for the Learn pandas & NumPy masterclass."""

from pipeline.analytics import (
    monthly_revenue,
    revenue_by_region_category,
    rolling_daily_revenue,
    share_of_region,
    top_customers,
)
from pipeline.arrays import moving_average, normalize_rows, pairwise_distances, zscore
from pipeline.frames import build_fact_table, load_customers, load_orders, load_products

__all__ = [
    "build_fact_table",
    "load_customers",
    "load_orders",
    "load_products",
    "monthly_revenue",
    "moving_average",
    "normalize_rows",
    "pairwise_distances",
    "revenue_by_region_category",
    "rolling_daily_revenue",
    "share_of_region",
    "top_customers",
    "zscore",
]
