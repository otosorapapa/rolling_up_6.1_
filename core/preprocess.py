"""前処理モジュール。"""

from __future__ import annotations

import pandas as pd


def to_monthly(
    df: pd.DataFrame,
    date_col: str = "date",
    revenue_col: str = "revenue",
    qty_col: str = "qty",
) -> pd.DataFrame:
    """取引データを月次に集計する。"""

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year_month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
    agg = (
        df.groupby("year_month")[[revenue_col, qty_col]]
        .sum()
        .reset_index()
        .sort_values("year_month")
    )
    return agg


def complete_months(df: pd.DataFrame, date_col: str = "year_month") -> pd.DataFrame:
    """欠測月を0で補完する。"""

    df = df.set_index(date_col)
    all_months = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    df = df.reindex(all_months, fill_value=0)
    df.index.name = date_col
    return df.reset_index()
