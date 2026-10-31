"""Checks for the companion pipeline.

The pandas results are compared against a plain-Python reference built with the
csv module, so the tests do not trust pandas to verify pandas.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline import (
    build_fact_table,
    load_customers,
    load_orders,
    load_products,
    monthly_revenue,
    moving_average,
    normalize_rows,
    pairwise_distances,
    revenue_by_region_category,
    rolling_daily_revenue,
    share_of_region,
    top_customers,
    zscore,
)
from pipeline.__main__ import run_report

DATA = Path(__file__).resolve().parents[1] / "data"


# ---------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def tables():
    return (
        load_orders(DATA / "orders.csv"),
        load_customers(DATA / "customers.csv"),
        load_products(DATA / "products.csv"),
    )


@pytest.fixture(scope="module")
def fact(tables):
    return build_fact_table(*tables)


@pytest.fixture(scope="module")
def reference_rows():
    """Shipped, deduplicated, complete orders joined by hand."""
    with open(DATA / "customers.csv", newline="") as fh:
        region = {r["customer_id"]: r["region"] for r in csv.DictReader(fh)}
    with open(DATA / "products.csv", newline="") as fh:
        category = {r["product_id"]: r["category"] for r in csv.DictReader(fh)}
    seen, rows = set(), []
    with open(DATA / "orders.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            key = tuple(r.values())
            if key in seen or not r["quantity"] or r["status"] != "shipped":
                seen.add(key)
                continue
            seen.add(key)
            rows.append(
                {
                    "month": r["order_date"][:7],
                    "region": region[r["customer_id"]],
                    "category": category[r["product_id"]],
                    "revenue": int(r["quantity"]) * float(r["unit_price"]),
                }
            )
    return rows


# ---------------------------------------------------------------- NumPy


def test_zscore_has_zero_mean_unit_std_and_handles_constant_columns():
    x = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0], [4.0, 5.0]])
    z = zscore(x, axis=0)
    np.testing.assert_allclose(z[:, 0].mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(z[:, 0].std(), 1.0)
    np.testing.assert_array_equal(z[:, 1], np.zeros(4))


def test_moving_average_matches_cumsum_formula():
    x = np.arange(10, dtype=float) ** 2
    window = 3
    c = np.cumsum(np.insert(x, 0, 0.0))
    expected = (c[window:] - c[:-window]) / window
    np.testing.assert_allclose(moving_average(x, window), expected)
    with pytest.raises(ValueError):
        moving_average(x, 11)


def test_pairwise_distances_matches_explicit_loop():
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=(5, 3)), rng.normal(size=(4, 3))
    expected = np.array([[np.linalg.norm(p - q) for q in b] for p in a])
    got = pairwise_distances(a, b)
    assert got.shape == (5, 4)
    np.testing.assert_allclose(got, expected)
    with pytest.raises(ValueError):
        pairwise_distances(a, rng.normal(size=(4, 2)))


def test_normalize_rows_sums_to_one_and_keeps_zero_rows():
    m = np.array([[1.0, 3.0], [0.0, 0.0], [2.0, 2.0]])
    out = normalize_rows(m)
    np.testing.assert_allclose(out.sum(axis=1), [1.0, 0.0, 1.0])
    np.testing.assert_allclose(out[0], [0.25, 0.75])


# ---------------------------------------------------------------- loading


def test_load_orders_cleans_duplicates_and_missing_quantities(tables):
    orders, _, _ = tables
    assert orders["order_id"].is_unique
    assert 1015 not in set(orders["order_id"])  # blank quantity dropped
    assert len(orders) == 25
    assert str(orders["quantity"].dtype) == "Int64"
    assert isinstance(orders["status"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_datetime64_any_dtype(orders["order_date"])


def test_fact_table_keeps_only_shipped_and_preserves_row_count(tables, fact, reference_rows):
    orders, _, _ = tables
    assert len(fact) == (orders["status"] == "shipped").sum() == len(reference_rows) == 22
    assert set(fact["status"].astype(str)) == {"shipped"}


def test_fact_table_rejects_unknown_keys(tables):
    orders, customers, products = tables
    with pytest.raises(ValueError, match="unknown customers or products"):
        build_fact_table(orders, customers.iloc[1:], products)


def test_fact_table_rejects_duplicated_dimension_keys(tables):
    orders, customers, products = tables
    duplicated = pd.concat([customers, customers.iloc[[0]]], ignore_index=True)
    with pytest.raises(pd.errors.MergeError):
        build_fact_table(orders, duplicated, products)


# ---------------------------------------------------------------- analytics


def test_region_category_matrix_matches_reference(fact, reference_rows):
    expected = defaultdict(float)
    for r in reference_rows:
        expected[(r["region"], r["category"])] += r["revenue"]
    matrix = revenue_by_region_category(fact)
    for (region, category), value in expected.items():
        assert matrix.loc[region, category] == pytest.approx(value, abs=0.01)
    assert matrix.to_numpy().sum() == pytest.approx(sum(expected.values()), abs=0.05)


def test_monthly_revenue_matches_reference(fact, reference_rows):
    expected = defaultdict(float)
    for r in reference_rows:
        expected[r["month"]] += r["revenue"]
    monthly = monthly_revenue(fact)
    got = {ts.strftime("%Y-%m"): value for ts, value in monthly.items()}
    assert got.keys() == expected.keys()
    for month, value in expected.items():
        assert got[month] == pytest.approx(value, abs=0.01)
    assert monthly.sum() == pytest.approx(3556.93, abs=0.01)


def test_top_customers_ranking(fact):
    top = top_customers(fact, n=3)
    assert list(top.index) == ["Ben Okafor", "Chen Wei", "Farid Haddad"]
    assert top.loc["Ben Okafor", "orders"] == 5
    assert top.loc["Ben Okafor", "revenue"] == pytest.approx(955.46)
    assert top["revenue"].is_monotonic_decreasing


def test_rolling_daily_revenue_is_gap_free_and_bounded(fact):
    rolled = rolling_daily_revenue(fact, window="7D")
    days = (fact["order_date"].max() - fact["order_date"].min()).days + 1
    assert len(rolled) == days
    assert rolled.index.is_monotonic_increasing
    daily_max = fact.groupby("order_date")["revenue"].sum().max()
    assert (rolled >= 0).all() and rolled.max() <= daily_max


def test_share_of_region_sums_to_one_per_region(fact):
    share = share_of_region(fact)
    assert share.index.equals(fact.index)
    sums = share.groupby(fact["region"], observed=True).sum()
    np.testing.assert_allclose(sums.to_numpy(), 1.0)


def test_report_renders_every_section():
    report = run_report(DATA)
    assert "Shipped orders: 22" in report
    for heading in ("Revenue by region and category", "Monthly revenue", "Top customers"):
        assert heading in report
