"""Utility helpers for generating the built-in demo dataset."""

from __future__ import annotations

import io
from dataclasses import dataclass
from importlib import resources
from typing import Iterable, Dict, List

import numpy as np
import pandas as pd

__all__ = [
    "SampleCSVMeta",
    "list_sample_csv_meta",
    "get_sample_csv_meta",
    "get_sample_csv_bytes",
    "load_sample_csv_dataframe",
    "load_sample_dataset",
]


@dataclass(frozen=True)
class _ProductTemplate:
    code: str
    name: str
    base: float
    trend: float
    seasonality: float
    volatility: float
    phase: float = 0.0
    promo_every: int | None = None
    promo_lift: float = 1.0
    promo_offset: int = 0
    dip_month: str | None = None
    dip_scale: float = 1.0


def _simulate_products(months: Iterable[pd.Period]) -> pd.DataFrame:
    rng = np.random.default_rng(20240115)
    templates = [
        _ProductTemplate(
            code="SMP001",
            name="サンプル栄養ドリンクA",
            base=3_800_000,
            trend=0.010,
            seasonality=0.12,
            volatility=0.05,
        ),
        _ProductTemplate(
            code="SMP002",
            name="サンプル炭酸飲料B",
            base=2_900_000,
            trend=0.007,
            seasonality=0.18,
            volatility=0.06,
            phase=1.2,
        ),
        _ProductTemplate(
            code="SMP003",
            name="サンプルスナックC",
            base=2_100_000,
            trend=0.004,
            seasonality=0.08,
            volatility=0.05,
            phase=2.2,
        ),
        _ProductTemplate(
            code="SMP004",
            name="サンプル健康食品D",
            base=4_600_000,
            trend=0.016,
            seasonality=0.05,
            volatility=0.04,
            promo_every=6,
            promo_lift=1.18,
            promo_offset=2,
        ),
        _ProductTemplate(
            code="SMP005",
            name="サンプル冷凍食品E",
            base=3_000_000,
            trend=-0.003,
            seasonality=0.10,
            volatility=0.05,
            phase=0.8,
            dip_month="2023-08",
            dip_scale=0.65,
        ),
        _ProductTemplate(
            code="SMP006",
            name="サンプルコスメF",
            base=2_700_000,
            trend=0.020,
            seasonality=0.04,
            volatility=0.07,
            phase=3.1,
            promo_every=12,
            promo_lift=1.30,
        ),
        _ProductTemplate(
            code="SMP007",
            name="サンプル家電G",
            base=5_500_000,
            trend=0.008,
            seasonality=0.25,
            volatility=0.09,
            phase=1.6,
            promo_every=12,
            promo_lift=1.40,
            promo_offset=5,
        ),
        _ProductTemplate(
            code="SMP008",
            name="サンプルデジタルH",
            base=1_900_000,
            trend=0.013,
            seasonality=0.06,
            volatility=0.05,
            phase=2.8,
        ),
    ]

    records: list[dict[str, object]] = []
    two_pi = 2.0 * np.pi
    for idx, period in enumerate(months):
        month_str = period.strftime("%Y-%m")
        rotation = two_pi * (idx % 12) / 12.0
        for template in templates:
            base_level = template.base * ((1 + template.trend) ** idx)
            seasonal = 1 + template.seasonality * np.sin(rotation + template.phase)
            noise = 1 + rng.normal(0.0, template.volatility)
            estimate = base_level * seasonal * max(noise, 0.55)

            if template.promo_every and (
                (idx - template.promo_offset) % template.promo_every == 0
            ):
                estimate *= template.promo_lift

            if template.dip_month and month_str == template.dip_month:
                estimate *= template.dip_scale

            amount = float(max(0.0, estimate))
            records.append(
                {
                    "product_code": template.code,
                    "product_name": template.name,
                    "month": month_str,
                    "sales_amount_jpy": amount,
                    "is_missing": False,
                }
            )

    df = pd.DataFrame.from_records(records)
    df["sales_amount_jpy"] = df["sales_amount_jpy"].round(0)
    df["product_code"] = df["product_code"].astype(str)
    df["product_name"] = df["product_name"].astype(str)
    df["month"] = df["month"].astype(str)
    df["is_missing"] = df["is_missing"].astype(bool)
    return df


def load_sample_dataset() -> pd.DataFrame:
    """Return a synthetic long-form dataset for instant demos."""

    months = pd.period_range("2022-01", periods=30, freq="M")
    sample_df = _simulate_products(months)
    sample_df = sample_df.sort_values(["product_code", "month"], ignore_index=True)
    return sample_df


@dataclass(frozen=True)
class SampleCSVMeta:
    """Metadata describing a downloadable CSV sample for onboarding."""

    key: str
    title: str
    filename: str
    description: str
    name_column: str
    code_column: str
    download_name: str


_SAMPLE_CSV_FILES: List[SampleCSVMeta] = [
    SampleCSVMeta(
        key="sales",
        title="売上サンプル",
        filename="sales_sample.csv",
        description="商品別の売上推移（24ヶ月）が格納されたワイド形式データです。",
        name_column="商品名",
        code_column="商品コード",
        download_name="sample_sales.csv",
    ),
    SampleCSVMeta(
        key="purchase",
        title="仕入サンプル",
        filename="purchase_sample.csv",
        description="主要仕入品目の月次推移をまとめたデータです。",
        name_column="仕入品目",
        code_column="仕入コード",
        download_name="sample_purchase.csv",
    ),
    SampleCSVMeta(
        key="expense",
        title="経費サンプル",
        filename="expense_sample.csv",
        description="勘定科目ごとの経費推移を確認できるテンプレートです。",
        name_column="費目",
        code_column="費目コード",
        download_name="sample_expense.csv",
    ),
]

_SAMPLE_CSV_REGISTRY: Dict[str, SampleCSVMeta] = {
    meta.key: meta for meta in _SAMPLE_CSV_FILES
}


def list_sample_csv_meta() -> List[SampleCSVMeta]:
    """Return metadata for all bundled sample CSV files."""

    return list(_SAMPLE_CSV_FILES)


def get_sample_csv_meta(key: str) -> SampleCSVMeta:
    """Return metadata for a specific sample CSV by key."""

    if key not in _SAMPLE_CSV_REGISTRY:
        raise KeyError(f"Unknown sample CSV key: {key}")
    return _SAMPLE_CSV_REGISTRY[key]


def get_sample_csv_bytes(key: str) -> bytes:
    """Load the raw CSV bytes for download buttons."""

    meta = get_sample_csv_meta(key)
    return resources.files(__name__).joinpath(meta.filename).read_bytes()


def load_sample_csv_dataframe(key: str) -> pd.DataFrame:
    """Load a bundled CSV sample into a pandas DataFrame."""

    csv_bytes = get_sample_csv_bytes(key)
    return pd.read_csv(io.BytesIO(csv_bytes))
