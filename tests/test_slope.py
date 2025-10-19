import numpy as np
import pandas as pd
import pytest

from services import slope_last_n, slopes_snapshot


def test_slope_last_n_all_period():
    s = pd.Series([1, 2, 3, 4, 5])
    m, r = slope_last_n(s, n=0)
    assert m == pytest.approx(1.0)
    assert r == pytest.approx(1 / 3)


def test_slope_last_n_insufficient_and_two_points():
    s_one = pd.Series([1])
    m1, r1 = slope_last_n(s_one, n=6)
    assert np.isnan(m1) and np.isnan(r1)

    s_two = pd.Series([1, 3])
    m2, r2 = slope_last_n(s_two, n=6)
    assert m2 == pytest.approx(2.0)
    assert r2 == pytest.approx(1.0)


def test_slopes_snapshot_handles_small_series():
    df = pd.DataFrame(
        {
            "product_code": ["A", "A", "B"],
            "month": [1, 2, 1],
            "year_sum": [1, 2, 1],
        }
    )
    snap = slopes_snapshot(df, n=0)
    a_slope = snap.loc[snap.product_code == "A", "slope_yen"].iloc[0]
    b_slope = snap.loc[snap.product_code == "B", "slope_yen"].iloc[0]
    assert a_slope == pytest.approx(1.0)
    assert np.isnan(b_slope)

