"""Run the end-to-end report: python -m pipeline [data_dir]"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from pipeline.analytics import monthly_revenue, revenue_by_region_category, top_customers
from pipeline.frames import build_fact_table, load_customers, load_orders, load_products


def run_report(data_dir: Path) -> str:
    orders = load_orders(data_dir / "orders.csv")
    customers = load_customers(data_dir / "customers.csv")
    products = load_products(data_dir / "products.csv")
    fact = build_fact_table(orders, customers, products)

    with pd.option_context("display.width", 120, "display.max_columns", 20):
        sections = [
            f"Shipped orders: {len(fact)}  |  Revenue: {fact['revenue'].sum():,.2f}",
            "\n== Revenue by region and category ==",
            revenue_by_region_category(fact).to_string(),
            "\n== Monthly revenue ==",
            monthly_revenue(fact).to_string(),
            "\n== Top customers ==",
            top_customers(fact).to_string(),
        ]
    return "\n".join(sections)


def main() -> None:
    default_dir = Path(__file__).resolve().parents[2] / "data"
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else default_dir
    print(run_report(data_dir))


if __name__ == "__main__":
    main()
