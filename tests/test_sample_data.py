import pandas as pd

from sample_data import load_sample_dataset
from services import fill_missing_months, compute_year_rolling, compute_slopes


def test_sample_dataset_structure():
    df = load_sample_dataset()
    required_cols = {"product_code", "product_name", "month", "sales_amount_jpy", "is_missing"}
    assert required_cols.issubset(df.columns)
    assert df["product_code"].nunique() >= 5
    assert df["month"].nunique() >= 12
    assert (df["sales_amount_jpy"] >= 0).all()
    assert df["is_missing"].dtype == bool


def test_sample_dataset_pipeline_roundtrip():
    df = load_sample_dataset()
    filled = fill_missing_months(df, policy="zero_fill")
    year_df = compute_year_rolling(filled, window=12, policy="zero_fill")
    year_df = compute_slopes(year_df, last_n=12)

    assert not year_df.empty
    assert year_df["year_sum"].notna().any()
    assert not year_df["product_code"].isna().any()
    # Ensure chronological order per product
    grouped = year_df.groupby("product_code")
    for _, group in grouped:
        months = pd.to_datetime(group["month"]).sort_values()
        assert months.is_monotonic_increasing
