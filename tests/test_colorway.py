import plotly.express as px
import plotly.graph_objects as go


def test_colorway_fallback():
    fig = go.Figure()
    palette = fig.layout.colorway or px.colors.qualitative.Safe
    assert palette[0] == px.colors.qualitative.Safe[0]

    fig.update_layout(colorway=["#111111", "#222222"])
    palette2 = fig.layout.colorway or px.colors.qualitative.Safe
    assert palette2[1] == "#222222"

