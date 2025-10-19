"""サンプル取引データを生成するスクリプト。

36か月、3チャネル、200SKU の擬似データを生成する。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(0)

CHANNELS = ["Online", "Store", "Wholesale"]
CATEGORIES = ["A", "B", "C"]


def generate_transactions(months: int = 36, n_sku: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=months, freq="MS")
    records = []
    for ym in dates:
        for sku in range(n_sku):
            channel = RNG.choice(CHANNELS)
            category = RNG.choice(CATEGORIES)
            price = RNG.integers(100, 500)
            qty = RNG.poisson(5)
            if RNG.random() < 0.02:
                qty = -qty  # 返品
            revenue = price * qty
            records.append(
                {
                    "date": ym + pd.Timedelta(days=int(RNG.integers(0, 27))),
                    "order_id": f"{ym:%Y%m}-{sku:04d}",
                    "channel": channel,
                    "product_code": f"SKU{sku:04d}",
                    "product_name": f"商品{sku:04d}",
                    "category": category,
                    "qty": qty,
                    "unit_price": price,
                    "revenue": revenue,
                    "customer_id": f"C{RNG.integers(1000):04d}",
                    "discount": 0.0,
                    "returns_flag": qty < 0,
                }
            )
    return pd.DataFrame(records)


if __name__ == "__main__":
    df = generate_transactions()
    df.to_csv("sample_transactions.csv", index=False, encoding="utf-8-sig")
    print("generated sample_transactions.csv")
