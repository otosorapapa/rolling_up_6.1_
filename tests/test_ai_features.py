import pandas as pd

from ai_features import (
    summarize_dataframe,
    generate_comment,
    explain_analysis,
    generate_anomaly_brief,
)


def test_summarize_dataframe_returns_text():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    txt = summarize_dataframe(df)
    assert isinstance(txt, str)
    assert len(txt) > 0


def test_generate_comment_returns_text():
    txt = generate_comment("売上が増加")
    assert isinstance(txt, str)
    assert len(txt) > 0


def test_explain_analysis_returns_text():
    txt = explain_analysis({"売上": 1200, "前年比": 1.05})
    assert isinstance(txt, str)
    assert "売上" in txt or len(txt) > 0


def test_generate_anomaly_brief_handles_data():
    df = pd.DataFrame(
        {
            "product_name": ["A", "B"],
            "month": ["2023-12", "2023-12"],
            "score": [3.2, -2.8],
            "year_sum": [1200000, 800000],
            "yoy": [0.12, -0.05],
            "delta": [50000, -30000],
        }
    )
    txt = generate_anomaly_brief(df)
    assert isinstance(txt, str)
    assert len(txt) > 0


def test_generate_anomaly_brief_empty():
    txt = generate_anomaly_brief(pd.DataFrame())
    assert "異常値は検出されませんでした" in txt
