from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

from core.design_tokens import get_color, get_font_stack, rgba


PRIMARY = get_color("primary")
PRIMARY_TEXT = get_color("text")
SECONDARY_TEXT = get_color("secondary")
ACCENT_SOFT = get_color("accent", "soft")
BORDER_STRONG = get_color("border", "strong")

LIGHT_TEXT = PRIMARY_TEXT
LIGHT_GRID = rgba(BORDER_STRONG, 0.55)
LIGHT_AXIS = SECONDARY_TEXT
DARK_TEXT = "#E6EDF3"
DARK_GRID = rgba("#244865", 0.55)
DARK_AXIS = "#C9D1D9"

LAYOUT_BASE = {
    "autosize": True,
    "margin": {"l": 56, "r": 16, "t": 28, "b": 42},
    "legend": {
        "orientation": "h",
        "y": 1.02,
        "yanchor": "bottom",
        "x": 0,
        "font": {"size": 10},
    },
    "hovermode": "x unified",
    "uniformtext": {"mode": "hide", "minsize": 10},
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(0,0,0,0)",
}

XAXIS_BASE = {
    "automargin": True,
    "tickmode": "auto",
    "nticks": 6,
    "tickangle": -30,
    "ticklabeloverflow": "hide past div",
}

YAXIS_BASE = {
    "automargin": True,
    "rangemode": "tozero",
    "tickformat": ",~s",
}

CONFIG_BASE = {
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "autoScale2d",
        "hoverCompareCartesian",
    ],
    "toImageButtonOptions": {"scale": 2, "format": "png"},
}


def apply_elegant_theme(fig: go.Figure, theme: str = "light") -> go.Figure:
    """Apply subdued, elegant styling to Plotly figures when enabled."""
    if not st.session_state.get("elegant_on", True):
        return fig
    if theme == "dark":
        dark_bg = "#0F1A2C"
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor=dark_bg,
            plot_bgcolor=dark_bg,
            font=dict(
                family=get_font_stack("body"),
                size=12,
                color=DARK_TEXT,
            ),
            legend=dict(
                bgcolor=rgba("#0F1A28", 0.9),
                bordercolor=rgba("#244865", 0.65),
                borderwidth=1,
                font=dict(color=DARK_TEXT),
            ),
            hoverlabel=dict(
                bgcolor=rgba("#0F1A28", 0.98),
                bordercolor=rgba("#244865", 0.6),
                font=dict(color=DARK_TEXT),
            ),
            colorway=[
                get_color("accent"),
                PRIMARY,
                get_color("success"),
                get_color("secondary"),
                get_color("warning"),
                get_color("error"),
            ],
        )
        grid = DARK_GRID
        axisline = DARK_AXIS
        marker_border = rgba(ACCENT_SOFT, 0.65)
    else:
        fig.update_layout(
            template="plotly_white",
            paper_bgcolor=get_color("surface"),
            plot_bgcolor=get_color("surface"),
            font=dict(
                family=get_font_stack("body"),
                size=12,
                color=LIGHT_TEXT,
            ),
            legend=dict(
                bgcolor=rgba(get_color("surface"), 0.95),
                bordercolor=rgba(BORDER_STRONG, 0.8),
                borderwidth=1,
                font=dict(color=SECONDARY_TEXT),
            ),
            hoverlabel=dict(
                bgcolor=rgba(get_color("surface"), 0.98),
                bordercolor=rgba(BORDER_STRONG, 0.8),
                font=dict(color=SECONDARY_TEXT),
            ),
        )
        grid = LIGHT_GRID
        axisline = LIGHT_AXIS
        marker_border = rgba(PRIMARY, 0.4)
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid,
        linecolor=axisline,
        ticks="outside",
        ticklen=4,
        tickcolor=axisline,
        showline=True,
        linewidth=1,
        title_standoff=14,
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid,
        linecolor=axisline,
        ticks="outside",
        ticklen=4,
        tickcolor=axisline,
        showline=True,
        linewidth=1,
        title_standoff=16,
    )
    fig.update_traces(
        selector=lambda t: "markers" in getattr(t, "mode", ""),
        marker=dict(size=6, line=dict(width=1.2, color=marker_border)),
    )
    return fig


def _plot_area_height(fig: go.Figure) -> int:
    h = fig.layout.height or 520
    m = fig.layout.margin or {}
    t = getattr(m, "t", 40) or 40
    b = getattr(m, "b", 60) or 60
    return max(120, int(h - t - b))


def _iter_numeric(values: Iterable[Any]) -> Iterable[float]:
    for item in values:
        if isinstance(item, (list, tuple, np.ndarray, pd.Series)):
            yield from _iter_numeric(item)
        else:
            if item is None:
                continue
            try:
                value = float(item)
            except (TypeError, ValueError):
                continue
            if np.isnan(value) or np.isinf(value):
                continue
            yield value


def padded_range(
    values: Iterable[Any],
    *,
    quantile: float = 0.99,
    padding: float = 0.15,
    to_zero: bool = True,
) -> tuple[float, float] | None:
    """Return a padded [min, max] range for numeric values."""

    cleaned = list(_iter_numeric(values))
    if not cleaned:
        return None
    arr = np.asarray(cleaned, dtype=float)
    if arr.size == 0:
        return None
    if np.all(~np.isfinite(arr)):
        return None
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    max_val = float(np.max(arr))
    min_val = float(np.min(arr))
    if quantile is not None and arr.size > 1:
        quantile_val = float(np.quantile(arr, quantile))
        max_val = max(max_val, quantile_val)
    upper = max_val * (1 + padding)
    if upper == 0:
        upper = 1.0
    if to_zero and min_val >= 0:
        lower = 0.0
    else:
        lower = min_val - abs(min_val) * padding
        if lower == upper:
            upper = lower + 1.0
    return (lower, upper)


def _has_datetime_x(fig: go.Figure) -> bool:
    for trace in fig.data:
        x_values = getattr(trace, "x", None)
        if x_values is None:
            continue
        if isinstance(x_values, pd.Series):
            if pd.api.types.is_datetime64_any_dtype(x_values):
                return True
            sample = x_values.iloc[0] if not x_values.empty else None
        elif isinstance(x_values, np.ndarray):
            if np.issubdtype(x_values.dtype, np.datetime64):
                return True
            sample = x_values[0] if x_values.size else None
        elif isinstance(x_values, (list, tuple)):
            sample = x_values[0] if x_values else None
        else:
            sample = x_values
        if isinstance(sample, (pd.Timestamp, np.datetime64)):
            return True
    return False


def _ensure_axis_defaults(axis, defaults: dict) -> dict:
    updates = {}
    for key, value in defaults.items():
        current = getattr(axis, key, None)
        if current is None:
            updates[key] = value
    return updates


def _apply_layout_defaults(fig: go.Figure) -> None:
    layout_updates: dict[str, Any] = {}
    for key, value in LAYOUT_BASE.items():
        if key == "margin":
            margin = fig.layout.margin or {}
            merged = {**value}
            for side, default in value.items():
                current = getattr(margin, side, None)
                if current is not None and current >= default:
                    merged[side] = current
            layout_updates["margin"] = merged
        elif key == "legend":
            legend = fig.layout.legend or {}
            if not legend:
                layout_updates["legend"] = value
        else:
            if getattr(fig.layout, key, None) in (None, {}):
                layout_updates[key] = value
    height = fig.layout.height
    if height is None or height < 220:
        layout_updates["height"] = 260 if height is None else max(220, height)
    if layout_updates:
        fig.update_layout(**layout_updates)

    xaxis_updates = _ensure_axis_defaults(fig.layout.xaxis, XAXIS_BASE)
    if _has_datetime_x(fig):
        if getattr(fig.layout.xaxis, "tickformat", None) is None:
            xaxis_updates["tickformat"] = "%Y-%m"
        if getattr(fig.layout.xaxis, "dtick", None) is None:
            xaxis_updates["dtick"] = "M3"
    if xaxis_updates:
        fig.update_xaxes(**xaxis_updates)

    yaxis_updates = _ensure_axis_defaults(fig.layout.yaxis, YAXIS_BASE)
    if yaxis_updates:
        fig.update_yaxes(**yaxis_updates)

    if getattr(fig.layout.yaxis, "range", None) is None:
        y_values = []
        for trace in fig.data:
            if getattr(trace, "yaxis", "y") not in (None, "y"):
                continue
            y = getattr(trace, "y", None)
            if y is None:
                continue
            y_values.extend(_iter_numeric(y))
        if y_values:
            range_candidate = padded_range(y_values)
            if range_candidate is not None:
                fig.update_yaxes(range=range_candidate)


def _y_to_px(y, y0, y1, plot_h):
    if y1 == y0:
        y1 = y0 + 1.0
    return float((1 - (y - y0) / (y1 - y0)) * plot_h)


def add_latest_labels_no_overlap(
    fig: go.Figure,
    df_long: pd.DataFrame,
    label_col: str = "display_name",
    x_col: str = "month",
    y_col: str = "year_sum",
    max_labels: int = 12,
    min_gap_px: int = 12,
    alternate_side: bool = True,
    xpad_px: int = 8,
    font_size: int = 11,
):
    last = df_long.sort_values(x_col).groupby(label_col, as_index=False).tail(1)
    if len(last) == 0:
        return
    cand = last.sort_values(y_col, ascending=False).head(max_labels).copy()
    yaxis = fig.layout.yaxis
    if getattr(yaxis, "range", None):
        y0, y1 = yaxis.range
    else:
        y0, y1 = float(df_long[y_col].min()), float(df_long[y_col].max())
    plot_h = _plot_area_height(fig)
    cand["y_px"] = cand[y_col].apply(lambda v: _y_to_px(v, y0, y1, plot_h))
    cand = cand.sort_values("y_px")
    placed = []
    for _, r in cand.iterrows():
        base = r["y_px"]
        if placed and base <= placed[-1] + min_gap_px:
            base = placed[-1] + min_gap_px
        base = float(np.clip(base, 0 + 6, plot_h - 6))
        placed.append(base)
        yshift = -(base - r["y_px"])
        xshift = xpad_px if (not alternate_side or (len(placed) % 2 == 1)) else -xpad_px
        fig.add_annotation(
            x=r[x_col],
            y=r[y_col],
            text=f"{r[label_col]}：{r[y_col]:,.0f}（{pd.to_datetime(r[x_col]).strftime('%Y-%m')}）",
            showarrow=False,
            xanchor="left" if xshift >= 0 else "right",
            yanchor="middle",
            xshift=xshift,
            yshift=yshift,
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            font=dict(size=font_size),
        )


def render_plotly_with_spinner(
    fig: go.Figure,
    *,
    spinner_text: str = "グラフを描画中…",
    use_container_width: bool = True,
    config: dict | None = None,
    **kwargs: Any,
) -> None:
    """Render a Plotly figure with a spinner to highlight processing."""

    with st.spinner(spinner_text):
        height = kwargs.pop("height", None)
        if height is not None:
            fig.update_layout(height=max(height, 220))
        _apply_layout_defaults(fig)
        merged_config = dict(CONFIG_BASE)
        if config:
            merged_config.update(config)
            base_buttons = CONFIG_BASE.get("modeBarButtonsToRemove", [])
            extra_buttons = config.get("modeBarButtonsToRemove")
            if extra_buttons:
                merged_config["modeBarButtonsToRemove"] = list(
                    dict.fromkeys(base_buttons + list(extra_buttons))
                )
        container = st.container()
        container.markdown("<div class='plot-wrap'>", unsafe_allow_html=True)
        container.plotly_chart(
            fig,
            use_container_width=use_container_width,
            config=merged_config,
            **kwargs,
        )
        container.markdown("</div>", unsafe_allow_html=True)
