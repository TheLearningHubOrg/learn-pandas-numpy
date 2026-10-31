"""Aggregations: groupby, pivots, resampling and window functions."""

from __future__ import annotations

import pandas as pd


def revenue_by_region_category(fact: pd.DataFrame) -> pd.DataFrame:
    """Region x category revenue matrix, zero-filled for empty cells."""
    return (
        fact.groupby(["region", "category"], observed=True)["revenue"]
        .sum()
        .unstack("category", fill_value=0.0)
        .round(2)
    )


def top_customers(fact: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """Named aggregation: order count, units and revenue per customer."""
    summary = fact.groupby("customer_name").agg(
        orders=("order_id", "nunique"),
        units=("quantity", "sum"),
        revenue=("revenue", "sum"),
    )
    summary["units"] = summary["units"].astype("int64")
    summary["revenue"] = summary["revenue"].round(2)
    return summary.nlargest(n, "revenue")


def monthly_revenue(fact: pd.DataFrame) -> pd.Series:
    """Calendar-month totals, labelled by month start ("MS")."""
    return fact.set_index("order_date")["revenue"].resample("MS").sum().round(2)


def rolling_daily_revenue(fact: pd.DataFrame, window: str = "7D") -> pd.Series:
    """Time-based trailing window over a gap-free daily series."""
    daily = fact.set_index("order_date")["revenue"].resample("D").sum()
    return daily.rolling(window, min_periods=1).mean()


def share_of_region(fact: pd.DataFrame) -> pd.Series:
    """Each order's share of its region's revenue, via transform (index preserved)."""
    region_total = fact.groupby("region", observed=True)["revenue"].transform("sum")
    return fact["revenue"] / region_total
