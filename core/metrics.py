"""指標計算モジュール。

売上年計（MAT）やPVM分解など、ダッシュボードで使用する
主要指標を純関数として定義する。
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def mat(series: pd.Series, window: int = 12) -> pd.Series:
    """12カ月移動合計（MAT）を計算する。

    Args:
        series: 月次売上などの時系列。月順で並んでいること。
        window: 集計対象の月数。通常は12。

    Returns:
        ``series`` と同じ長さの ``pd.Series``。
        先頭 ``window-1`` 要素は ``NaN``。
    """

    if window <= 0:
        raise ValueError("window は正の整数である必要があります。")
    return series.rolling(window=window, min_periods=window).sum()


@dataclass
class PVMResult:
    """PVM分解の結果。"""

    price_effect: float
    volume_effect: float
    mix_effect: float
    actual_diff: float


def pvm(
    df0: pd.DataFrame,
    df1: pd.DataFrame,
    *,
    price_col: str = "unit_price",
    qty_col: str = "qty",
) -> PVMResult:
    """価格(P)、数量(V)、ミックス(M)効果に売上差分を分解する。

    Args:
        df0: 基準期間のデータ。 ``price_col`` と ``qty_col`` を含むこと。
        df1: 比較期間のデータ。 ``df0`` と同一商品のみを含む想定。
        price_col: 単価列名。
        qty_col: 数量列名。

    Returns:
        :class:`PVMResult`
    """

    merged = (
        df0[[price_col, qty_col]]
        .join(df1[[price_col, qty_col]], lsuffix="0", rsuffix="1", how="outer")
        .fillna(0)
    )
    p0 = merged[f"{price_col}0"]
    q0 = merged[f"{qty_col}0"]
    p1 = merged[f"{price_col}1"]
    q1 = merged[f"{qty_col}1"]

    price_effect = ((p1 - p0) * q0).sum()
    volume_effect = (p0 * (q1 - q0)).sum()
    actual_diff = (p1 * q1 - p0 * q0).sum()
    mix_effect = actual_diff - price_effect - volume_effect
    return PVMResult(
        price_effect=price_effect,
        volume_effect=volume_effect,
        mix_effect=mix_effect,
        actual_diff=actual_diff,
    )
