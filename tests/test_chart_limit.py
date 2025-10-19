import pandas as pd
from core.chart_card import limit_products, MAX_DISPLAY_PRODUCTS


def test_limit_products_maximum():
    records = []
    # create 80 products with increasing year_sum
    for i in range(80):
        code = f"P{i:03d}"
        for m in ["2024-01", "2024-02"]:
            records.append(
                {
                    "product_code": code,
                    "product_name": f"Product {i}",
                    "display_name": f"Product {i}",
                    "month": m,
                    "year_sum": float(i),
                    "delta": 0.0,
                    "yoy": 0.0,
                }
            )
    df = pd.DataFrame(records)
    df["month"] = pd.to_datetime(df["month"])

    limited = limit_products(df, max_products=MAX_DISPLAY_PRODUCTS)
    assert limited["product_code"].nunique() == MAX_DISPLAY_PRODUCTS
    # ensure that the highest year_sum products are kept
    expected_codes = {f"P{i:03d}" for i in range(80 - MAX_DISPLAY_PRODUCTS, 80)}
    assert set(limited["product_code"].unique()) == expected_codes
