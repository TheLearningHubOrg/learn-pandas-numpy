"""Loading and shaping: explicit dtypes, deduplication and validated joins."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ORDER_STATUSES = ["shipped", "returned", "cancelled"]


def load_orders(path: str | Path) -> pd.DataFrame:
    """Read orders with explicit dtypes, drop exact duplicates and unusable rows."""
    orders = pd.read_csv(
        path,
        dtype={
            "order_id": "int64",
            "customer_id": "string",
            "product_id": "string",
            "quantity": "Int64",  # nullable integer: a blank cell becomes <NA>
            "unit_price": "float64",
        },
        parse_dates=["order_date"],
    )
    orders["status"] = pd.Categorical(orders["status"], categories=ORDER_STATUSES)
    orders = (
        orders.drop_duplicates()
        .dropna(subset=["quantity"])
        .assign(revenue=lambda df: df["quantity"].astype("float64") * df["unit_price"])
        .reset_index(drop=True)
    )
    if not orders["order_id"].is_unique:
        raise ValueError("order_id must be unique after removing exact duplicates")
    return orders


def load_customers(path: str | Path) -> pd.DataFrame:
    customers = pd.read_csv(
        path,
        dtype={"customer_id": "string", "name": "string"},
        parse_dates=["signup_date"],
    )
    customers["region"] = customers["region"].astype("category")
    return customers


def load_products(path: str | Path) -> pd.DataFrame:
    products = pd.read_csv(path, dtype={"product_id": "string", "name": "string"})
    products["category"] = products["category"].astype("category")
    return products


def build_fact_table(
    orders: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    include_statuses: tuple[str, ...] = ("shipped",),
) -> pd.DataFrame:
    """Join orders to both dimensions. validate= fails fast on duplicated keys."""
    kept = orders.loc[orders["status"].isin(include_statuses)]
    fact = kept.merge(
        customers.rename(columns={"name": "customer_name"}),
        on="customer_id",
        how="left",
        validate="many_to_one",
    ).merge(
        products.rename(columns={"name": "product_name"}),
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    missing = fact["region"].isna() | fact["category"].isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} orders reference unknown customers or products")
    return fact
