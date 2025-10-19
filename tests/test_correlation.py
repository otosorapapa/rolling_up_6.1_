import math

import pandas as pd

from core.correlation import corr_table


def test_corr_table_pairwise_counts_and_significance():
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0, None],
            "B": [1.0, 2.0, 3.0, 4.0],
            "C": [4.0, 3.0, 2.0, 1.0],
        }
    )

    tbl = corr_table(df, ["A", "B", "C"], pairwise=True, min_periods=3)

    pairs = {row["pair"]: row for _, row in tbl.iterrows()}

    assert pairs["A×B"]["n"] == 3
    assert pairs["B×C"]["n"] == 4
    assert pairs["A×C"]["n"] == 3

    # n=4 pair should exceed the confidence threshold, others lack degrees of freedom
    assert pairs["B×C"]["sig"] == "有意(95%)"
    assert pairs["A×B"]["sig"] in {"n.s.", "有意(95%)"}
    assert pairs["A×C"]["sig"] in {"n.s.", "有意(95%)"}


def test_corr_table_pairwise_insufficient_data():
    df = pd.DataFrame({"A": [1.0, None, 3.0], "B": [None, 5.0, None]})

    tbl = corr_table(df, ["A", "B"], pairwise=True, min_periods=3)

    assert len(tbl) == 1
    row = tbl.iloc[0]
    assert row["pair"] == "A×B"
    assert row["n"] == 0 or row["n"] < 3
    assert row["sig"] == "データ不足"
    assert math.isnan(row["r"])
