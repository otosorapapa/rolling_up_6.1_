import numpy as np
import pandas as pd
import plotly.express as px
from plotly.colors import hex_to_rgb, sample_colorscale
import streamlit as st

from services import (
    slopes_snapshot,
    shape_flags,
    top_growth_codes,
    forecast_linear_band,
    forecast_holt_linear,
    band_from_moving_stats,
    detect_linear_anomalies,
)
from core.plot_utils import (
    add_latest_labels_no_overlap,
    apply_elegant_theme,
    render_plotly_with_spinner,
)

UNIT_SCALE = {"円": 1, "千円": 1_000, "百万円": 1_000_000}
MAX_DISPLAY_PRODUCTS = 60
TREND_POS_THRESHOLD = 0.05
TREND_NEG_THRESHOLD = -0.05


def _sample_scale_colors(scale: list[str], count: int, *, low: float = 0.1, high: float = 0.9) -> list[str]:
    """Sample ``count`` colors from a Plotly color scale with optional padding."""

    if count <= 0:
        return []
    low = max(0.0, min(low, 1.0))
    high = max(low, min(high, 1.0))
    if count == 1:
        midpoint = (low + high) / 2 if high > low else 0.5
        return sample_colorscale(scale, [midpoint])
    if high == low:
        step = 1.0 / max(count - 1, 1)
        positions = [i * step for i in range(count)]
    else:
        step = (high - low) / max(count - 1, 1)
        positions = [low + step * i for i in range(count)]
    return sample_colorscale(scale, positions)


def _build_trend_color_map(latest: pd.DataFrame) -> dict[str, str]:
    """Return a color map keyed by ``display_name`` based on latest YoY values."""

    if latest.empty or "yoy" not in latest.columns:
        base = px.colors.qualitative.Safe
        return {name: base[i % len(base)] for i, name in enumerate(latest.index)}

    yoy = latest["yoy"]
    pos_names = (
        yoy[yoy >= TREND_POS_THRESHOLD]
        .sort_values(ascending=False)
        .index.tolist()
    )
    neg_names = yoy[yoy <= TREND_NEG_THRESHOLD].sort_values().index.tolist()
    taken = set(pos_names) | set(neg_names)
    neutral_names = [name for name in yoy.index if name not in taken]

    color_map: dict[str, str] = {}
    color_map.update(
        zip(
            pos_names,
            _sample_scale_colors(px.colors.sequential.Teal, len(pos_names), low=0.1, high=0.85),
        )
    )
    color_map.update(
        zip(
            neg_names,
            _sample_scale_colors(px.colors.sequential.OrRd, len(neg_names), low=0.0, high=0.85),
        )
    )
    color_map.update(
        zip(
            neutral_names,
            _sample_scale_colors(px.colors.sequential.Purples, len(neutral_names), low=0.1, high=0.9),
        )
    )
    return color_map


def format_int(val: float | int) -> str:
    try:
        return f"{int(val):,}"
    except (TypeError, ValueError):
        return "0"


def int_input(label: str, value: int, key: str | None = None) -> int:
    """Integer input widget that keeps thousands separators."""

    text = st.text_input(label, format_int(value), key=key)
    try:
        return int(text.replace(",", ""))
    except ValueError:
        return value


def marker_step(dates, target_points=24):
    n = len(pd.unique(dates))
    return max(1, round(n / target_points))


def limit_products(
    dfp: pd.DataFrame, max_products: int = MAX_DISPLAY_PRODUCTS
) -> pd.DataFrame:
    """Limit dataframe to at most ``max_products`` unique product codes.

    When the number of products exceeds the limit, the top products are
    selected based on the latest ``year_sum`` values.
    """
    codes = dfp["product_code"].unique()
    if len(codes) <= max_products:
        return dfp
    snapshot = dfp.sort_values("month").groupby("product_code").tail(1)
    top_codes = snapshot.nlargest(max_products, "year_sum")["product_code"]
    return dfp[dfp["product_code"].isin(top_codes)].copy()


def _ensure_css():
    if st.session_state.get("_chart_css_injected"):
        return
    st.markdown(
        """
<style>
.chart-card { position: relative; margin:var(--space-1,0.5rem) 0 var(--space-2,0.75rem); border-radius:16px;
  border:1px solid var(--border, rgba(var(--primary-rgb,11,31,59),0.18)); background:var(--panel,#ffffff);
  box-shadow:0 16px 32px rgba(var(--primary-rgb,11,31,59),0.08); display:grid;
  grid-template-rows:auto 1fr auto; row-gap:var(--space-2,0.75rem); }
.chart-toolbar { position: sticky; top:-1px; z-index:5;
  display:flex; gap:var(--space-1,0.5rem); flex-wrap:wrap; align-items:center;
  padding:0.5rem 0.75rem; background: linear-gradient(180deg, rgba(var(--accent-rgb,30,136,229),0.08), rgba(var(--accent-rgb,30,136,229),0.02));
  border-bottom:1px solid var(--border, rgba(var(--primary-rgb,11,31,59),0.18)); }
.chart-toolbar .stRadio, .chart-toolbar .stSelectbox, .chart-toolbar .stSlider,
.chart-toolbar .stMultiSelect, .chart-toolbar .stCheckbox { margin-bottom:0 !important; }
.chart-toolbar .stRadio > label, .chart-toolbar .stCheckbox > label { color:var(--ink,var(--primary,#0B1F3B)); font-weight:600; }
.chart-toolbar .stSlider label { color:var(--ink,var(--primary,#0B1F3B)); }
.chart-body { padding:var(--space-2,0.75rem) var(--space-3,1rem); display:flex; flex-direction:column; gap:var(--space-2,0.75rem); }
</style>
""",
        unsafe_allow_html=True,
    )
    st.session_state["_chart_css_injected"] = True


def toolbar_sku_detail(
    multi_mode: bool,
    key_prefix: str = "sku",
    include_expand_toggle: bool = True,
):
    _ensure_css()
    ui = st.session_state.setdefault("ui", {})

    def widget_key(name: str) -> str:
        return f"{key_prefix}_{name}"

    a, b, c, d, e, f = st.columns([1.05, 1.5, 1.05, 0.95, 0.85, 0.9])
    with a:
        period_opts = ["12ヶ月", "24ヶ月", "36ヶ月"]
        default = ui.get("period", "24ヶ月")
        period = st.radio(
            "期間",
            period_opts,
            horizontal=True,
            index=period_opts.index(default) if default in period_opts else 1,
            key=widget_key("period"),
        )
        ui["period"] = period
    with b:
        node_opts = ["自動", "主要ノードのみ", "すべて", "非表示"]
        node_default = ui.get("node_mode", "自動")
        node_mode = st.radio(
            "ノード表示",
            node_opts,
            horizontal=True,
            index=node_opts.index(node_default)
            if node_default in node_opts
            else 0,
            key=widget_key("node_mode"),
        )
        ui["node_mode"] = node_mode
    with c:
        hover_opts = ["個別", "同月まとめ"]
        hover_default = ui.get("hover_mode", "個別")
        hover_mode = st.radio(
            "ホバー",
            hover_opts,
            horizontal=True,
            index=hover_opts.index(hover_default)
            if hover_default in hover_opts
            else 0,
            key=widget_key("hover_mode"),
        )
        ui["hover_mode"] = hover_mode
    with d:
        op_opts = ["パン", "ズーム", "選択"]
        op_default = ui.get("op_mode", "パン")
        op_mode = st.radio(
            "操作",
            op_opts,
            horizontal=True,
            index=op_opts.index(op_default) if op_default in op_opts else 0,
            key=widget_key("op_mode"),
        )
        ui["op_mode"] = op_mode
    with e:
        peak_on = st.checkbox(
            "ピーク表示",
            value=ui.get("peak_on", False),
            key=widget_key("peak_on"),
        )
        ui["peak_on"] = peak_on
    with f:
        if include_expand_toggle:
            expand_mode = st.toggle(
                "拡大モード",
                value=ui.get("expand_mode", False),
                key="sku_expand_mode",
                help="グラフを拡大しながら同じ操作パネルを操作できます。",
            )
            ui["expand_mode"] = expand_mode
        else:
            expand_mode = ui.get("expand_mode", False)

    f, g, h, i = st.columns([1.0, 1.5, 1.4, 1.4])
    with f:
        unit_opts = ["円", "千円", "百万円"]
        unit = st.radio(
            "単位",
            unit_opts,
            horizontal=True,
            index=unit_opts.index(ui.get("unit", "千円")),
            key=widget_key("unit"),
        )
        ui["unit"] = unit
    with g:
        enable_avoid = st.checkbox(
            "ラベル衝突回避",
            value=ui.get("enable_avoid", True),
            key=widget_key("enable_avoid"),
        )
        ui["enable_avoid"] = enable_avoid
        gap_px = st.slider(
            "ラベル最小間隔(px)",
            8,
            24,
            int(ui.get("gap_px", 12)),
            key=widget_key("gap_px"),
        )
        ui["gap_px"] = gap_px
    with h:
        max_labels = st.slider(
            "ラベル最大件数",
            5,
            20,
            int(ui.get("max_labels", 12)),
            key=widget_key("max_labels"),
        )
        ui["max_labels"] = max_labels
    with i:
        alt_side = st.checkbox(
            "ラベル左右交互配置",
            value=ui.get("alt_side", True),
            key=widget_key("alt_side"),
        )
        ui["alt_side"] = alt_side

    slope_conf = None
    if multi_mode:
        j, k, l, m = st.columns([1.2, 1.6, 1.2, 1.6])
        with j:
            quick_opts = ["なし", "Top5", "Top10", "最新YoY上位", "直近6M伸長上位"]
            quick = st.radio(
                "クイック絞り込み",
                quick_opts,
                horizontal=True,
                index=quick_opts.index(ui.get("quick", "なし")),
                key=widget_key("quick"),
            )
            ui["quick"] = quick
        with k:
            n_win = st.slider(
                "傾きウィンドウ（月）",
                0,
                12,
                int(ui.get("n_win", 6)),
                1,
                help="0=自動（系列の全期間で判定）",
                key=widget_key("n_win"),
            )
            ui["n_win"] = n_win
            cmp_opts = ["以上", "未満"]
            cmp_mode = st.radio(
                "傾き条件",
                cmp_opts,
                horizontal=True,
                index=cmp_opts.index(ui.get("cmp_mode", "以上")),
                key=widget_key("cmp_mode"),
            )
            ui["cmp_mode"] = cmp_mode
        with l:
            thr_opts = ["円/月", "%/月", "zスコア"]
            thr_type = st.radio(
                "しきい値の種類",
                thr_opts,
                horizontal=True,
                index=thr_opts.index(ui.get("thr_type", "円/月")),
                key=widget_key("thr_type"),
            )
            ui["thr_type"] = thr_type
        with m:
            if thr_type == "円/月":
                thr_val = int_input(
                    "しきい値",
                    int(ui.get("thr_val", 0)),
                    key=widget_key("thr_val"),
                )
            else:
                thr_val = st.number_input(
                    "しきい値",
                    value=float(ui.get("thr_val", 0.0)),
                    step=0.01,
                    format="%.2f",
                    key=widget_key("thr_val"),
                )
            ui["thr_val"] = float(thr_val)
        s1, s2 = st.columns([1.2, 1.2])
        with s1:
            shape_opts = ["（なし）", "急勾配", "山（への字）", "谷（逆への字）"]
            shape_pick = st.radio(
                "形状抽出",
                shape_opts,
                horizontal=True,
                index=shape_opts.index(ui.get("shape_pick", "（なし）")),
                key=widget_key("shape_pick"),
            )
            ui["shape_pick"] = shape_pick
        with s2:
            sens = st.slider(
                "形状抽出の感度",
                0.0,
                1.0,
                float(ui.get("sens", 0.5)),
                0.05,
                key=widget_key("sens"),
            )
            ui["sens"] = sens
        slope_conf = dict(
            n_win=n_win,
            cmp_mode=cmp_mode,
            thr_type=thr_type,
            thr_val=float(thr_val),
            shape_pick=shape_pick,
            sens=sens,
            quick=quick,
        )
    p1, p2, p3, p4, p5 = st.columns([1.4, 0.9, 0.9, 1.2, 0.9])
    with p1:
        method_opts = [
            "なし",
            "ローカル線形±kσ",
            "Holt線形",
            "移動平均±kσ",
            "移動平均±MAD",
        ]
        f_method = st.selectbox(
            "予測帯",
            method_opts,
            index=method_opts.index(ui.get("f_method", "なし")),
            key=widget_key("f_method"),
        )
        ui["f_method"] = f_method
    with p2:
        f_win = st.selectbox(
            "学習窓幅",
            [6, 9, 12],
            index=[6, 9, 12].index(ui.get("f_win", 12)),
            key=widget_key("f_win"),
        )
        ui["f_win"] = f_win
    with p3:
        f_h = st.selectbox(
            "先の予測ステップ",
            [3, 6, 12],
            index=[3, 6, 12].index(ui.get("f_h", 6)),
            key=widget_key("f_h"),
        )
        ui["f_h"] = f_h
    with p4:
        f_k = st.slider(
            "バンド幅k",
            1.5,
            3.0,
            float(ui.get("f_k", 2.0)),
            0.1,
            key=widget_key("f_k"),
        )
        ui["f_k"] = f_k
    with p5:
        f_robust = st.checkbox(
            "ロバスト(MAD)",
            value=ui.get("f_robust", False),
            key=widget_key("f_robust"),
        )
        ui["f_robust"] = f_robust
    anom_opts = ["OFF", "z≥2.5", "MAD≥3.5"]
    anomaly = st.selectbox(
        "異常検知",
        anom_opts,
        index=anom_opts.index(ui.get("anomaly", "OFF")),
        key=widget_key("anomaly"),
    )
    ui["anomaly"] = anomaly
    if not include_expand_toggle and expand_mode:
        chart_height = st.slider(
            "チャート高さ(px)",
            600,
            900,
            int(ui.get("chart_height", 760)),
            step=20,
            key=widget_key("chart_height"),
        )
        ui["chart_height"] = chart_height
    else:
        ui.setdefault("chart_height", 600)
    st.session_state["ui"] = ui
    return dict(
        period=period,
        node_mode=node_mode,
        hover_mode=hover_mode,
        op_mode=op_mode,
        peak_on=peak_on,
        unit=unit,
        enable_avoid=enable_avoid,
        gap_px=gap_px,
        max_labels=max_labels,
        alt_side=alt_side,
        slope_conf=slope_conf,
        forecast_method=ui.get("f_method", "なし"),
        forecast_window=ui.get("f_win", 12),
        forecast_horizon=ui.get("f_h", 6),
        forecast_k=ui.get("f_k", 2.0),
        forecast_robust=ui.get("f_robust", False),
        anomaly=ui.get("anomaly", "OFF"),
        expand_mode=expand_mode,
        chart_height=int(ui.get("chart_height", 600)),
    )


def build_chart_card(
    df_long,
    selected_codes,
    multi_mode,
    tb,
    band_range=None,
    *,
    height: int | None = None,
    config: dict | None = None,
):
    months = {"12ヶ月": 12, "24ヶ月": 24, "36ヶ月": 36}[tb["period"]]
    dfp = df_long.sort_values("month").groupby("product_code").tail(months)
    if selected_codes:
        dfp = dfp[dfp["product_code"].isin(selected_codes)].copy()
    if dfp["product_code"].nunique() > MAX_DISPLAY_PRODUCTS:
        st.warning(f"表示件数が多いため上位{MAX_DISPLAY_PRODUCTS}件のみを描画します")
        dfp = limit_products(dfp)

    scale = UNIT_SCALE[tb["unit"]]
    dfp = dfp.sort_values("month").copy()
    if "yoy" not in dfp.columns:
        dfp["yoy"] = np.nan
    if "delta" not in dfp.columns:
        dfp["delta"] = np.nan

    if multi_mode and tb.get("slope_conf"):
        sc = tb["slope_conf"]
        snap = slopes_snapshot(dfp, n=sc["n_win"])
        key = {"円/月": "slope_yen", "%/月": "slope_ratio", "zスコア": "slope_z"}[
            sc["thr_type"]
        ]
        mask = (
            (snap[key] >= sc["thr_val"])
            if sc["cmp_mode"] == "以上"
            else (snap[key] <= sc["thr_val"])
        )
        codes_by_slope = set(snap.loc[mask, "product_code"])
        if sc.get("quick") and sc["quick"] != "なし":
            snapshot = dfp.sort_values("month").groupby("product_code").tail(1)
            if sc["quick"] == "Top5":
                quick_codes = snapshot.nlargest(5, "year_sum")["product_code"]
            elif sc["quick"] == "Top10":
                quick_codes = snapshot.nlargest(10, "year_sum")["product_code"]
            elif sc["quick"] == "最新YoY上位":
                quick_codes = (
                    snapshot.dropna(subset=["yoy"])
                    .sort_values("yoy", ascending=False)
                    .head(10)["product_code"]
                )
            elif sc["quick"] == "直近6M伸長上位":
                quick_codes = top_growth_codes(
                    dfp, dfp["month"].max(), window=6, top=10
                )
            else:
                quick_codes = snapshot["product_code"]
            codes_by_slope = codes_by_slope & set(quick_codes)
        if sc["shape_pick"] != "（なし）":
            eff_n = sc["n_win"] if sc["n_win"] > 0 else 12
            sh = shape_flags(
                dfp,
                window=max(6, eff_n * 2),
                alpha_ratio=0.02 * (1.0 - sc["sens"]),
                amp_ratio=0.06 * (1.0 - sc["sens"]),
            )
            pick_map = {
                "急勾配": snap.loc[snap["slope_z"].abs() >= 1.5, "product_code"],
                "山（への字）": sh.loc[sh["is_mountain"], "product_code"],
                "谷（逆への字）": sh.loc[sh["is_valley"], "product_code"],
            }
            pick = pick_map.get(sc["shape_pick"])
            dfp = dfp[dfp["product_code"].isin(set(pick).intersection(codes_by_slope))]
        else:
            dfp = dfp[dfp["product_code"].isin(codes_by_slope)]

    dfp["year_sum_disp"] = dfp["year_sum"] / scale
    dfp["yoy_display"] = dfp["yoy"].apply(
        lambda v: "—" if pd.isna(v) else f"{v * 100:+.1f}%"
    )
    dfp["delta_display"] = dfp["delta"].apply(
        lambda v: "—" if pd.isna(v) else f"{v / scale:+,.0f} {tb['unit']}"
    )
    latest_snapshot = (
        dfp.groupby("display_name", as_index=False)
        .tail(1)
        .set_index("display_name")
    )
    latest_yoy = (
        dfp.groupby("display_name")["yoy"].transform("last")
        if "yoy" in dfp.columns
        else pd.Series(np.nan, index=dfp.index)
    )
    dfp["label_with_yoy"] = dfp["display_name"] + latest_yoy.apply(
        lambda v: "" if pd.isna(v) else f"（YoY {v * 100:+.1f}%）"
    )
    color_map = _build_trend_color_map(latest_snapshot)

    line_kwargs = dict(
        x="month",
        y="year_sum_disp",
        color="display_name",
        custom_data=["display_name", "yoy_display", "delta_display"],
    )
    if color_map:
        line_kwargs["color_discrete_map"] = color_map
    fig = px.line(dfp, **line_kwargs)
    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        "月：%{x|%Y-%m}<br>"
        f"年計：%{{y:,.0f}} {tb['unit']}<br>"
        "前年同月比：%{customdata[1]}<br>"
        "前月差：%{customdata[2]}<extra></extra>"
    )
    fig.update_yaxes(title_text=f"売上 年計（{tb['unit']}）", tickformat="~,d")
    fig.update_traces(mode="lines+markers", hovertemplate=hovertemplate)
    if band_range:
        low, high = band_range
        fig.add_hrect(
            y0=low / scale,
            y1=high / scale,
            fillcolor="rgba(30, 167, 140, 0.18)",
            opacity=1.0,
            line=dict(color="rgba(30, 167, 140, 0.35)", width=1),
        )

    month_count = dfp["month"].nunique()
    if month_count <= 18:
        dtick = "M1"
    elif month_count <= 36:
        dtick = "M3"
    else:
        dtick = "M6"
    fig.update_xaxes(tickformat="%Y-%m", dtick=dtick)
    fig.update_layout(
        dragmode={"パン": "pan", "ズーム": "zoom", "選択": "select"}[tb["op_mode"]],
        hovermode="closest" if tb["hover_mode"] == "個別" else "x unified",
        legend=dict(
            orientation="h",
            title_text=None,
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            itemclick="toggleothers",
            itemdoubleclick="toggle",
        ),
        margin=dict(l=72, r=40, t=60, b=80),
    )
    latest_yoy_map = (
        latest_snapshot["yoy"].to_dict() if "yoy" in latest_snapshot.columns else {}
    )
    for tr in fig.data:
        if "lines" not in getattr(tr, "mode", ""):
            continue
        yoy_val = latest_yoy_map.get(tr.name)
        if yoy_val is None or pd.isna(yoy_val):
            tr.line.width = 2.3
        elif yoy_val >= TREND_POS_THRESHOLD:
            tr.line.width = 3.0
        elif yoy_val <= TREND_NEG_THRESHOLD:
            tr.line.width = 2.6
            tr.line.dash = "dot"
        else:
            tr.line.width = 2.4
    line_colors = {
        tr.name: tr.line.color
        for tr in fig.data
        if "lines" in getattr(tr, "mode", "")
    }

    if tb.get("forecast_method") and tb["forecast_method"] != "なし":
        method = tb["forecast_method"]
        win = tb.get("forecast_window", 12)
        horizon = tb.get("forecast_horizon", 6)
        k = tb.get("forecast_k", 2.0)
        robust = tb.get("forecast_robust", False)
        for name, d in dfp.groupby("display_name"):
            s = d.sort_values("month").set_index("month")["year_sum"]
            if method == "ローカル線形±kσ":
                f, lo, hi = forecast_linear_band(
                    s, window=win, horizon=horizon, k=k, robust=robust
                )
            elif method == "Holt線形":
                f = forecast_holt_linear(s, horizon=horizon)
                f2, lo, hi = forecast_linear_band(
                    s, window=win, horizon=horizon, k=k, robust=robust
                )
                lo, hi = f - (f2 - lo), f + (hi - f2)
            elif method == "移動平均±kσ":
                f, lo, hi = band_from_moving_stats(
                    s, window=win, horizon=horizon, k=k, robust=False
                )
            else:
                f, lo, hi = band_from_moving_stats(
                    s, window=win, horizon=horizon, k=k, robust=True
                )
            if len(f) == 0:
                continue
            last_t = pd.to_datetime(d["month"].max())
            future_idx = pd.period_range(
                last_t.to_period("M"), periods=horizon, freq="M"
            ).to_timestamp() + pd.offsets.MonthBegin(1)
            base_color = line_colors.get(name)
            line_style = dict(dash="dash")
            if base_color:
                line_style["color"] = base_color
            fig.add_scatter(
                x=future_idx,
                y=f / scale,
                mode="lines",
                name=f"{name}予測",
                line=line_style,
                showlegend=False,
            )
            fill_color = "rgba(113,178,255,.18)"
            if base_color and isinstance(base_color, str) and base_color.startswith("#"):
                r, g, b = hex_to_rgb(base_color)
                fill_color = f"rgba({r},{g},{b},0.12)"
            fig.add_scatter(
                x=future_idx,
                y=hi / scale,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
            )
            fig.add_scatter(
                x=future_idx,
                y=lo / scale,
                mode="lines",
                fill="tonexty",
                line=dict(width=0, color=base_color),
                fillcolor=fill_color,
                showlegend=False,
            )

    if tb.get("anomaly") and tb["anomaly"] != "OFF":
        robust = tb["anomaly"].startswith("MAD")
        thr = 3.5 if robust else 2.5
        for name, d in dfp.groupby("display_name"):
            s = d.sort_values("month").set_index("month")["year_sum"]
            res = detect_linear_anomalies(
                s, window=tb.get("forecast_window", 12), threshold=thr, robust=robust
            )
            if res.empty:
                continue
            fig.add_scatter(
                x=pd.to_datetime(res["month"]),
                y=res["value"] / scale,
                mode="markers",
                name=f"{name}異常",
                marker=dict(symbol="triangle-up", color="red", size=10),
                showlegend=False,
                customdata=np.stack([res["score"]], axis=-1),
                hovertemplate=f"<b>{name}</b><br>月：%{{x|%Y-%m}}<br>値：%{{y:,.0f}} {tb['unit']}<br>スコア：%{{customdata[0]:.2f}}<extra></extra>",
            )

    theme_is_dark = st.get_option("theme.base") == "dark"
    halo = "#ffffff" if theme_is_dark else "#222222"
    if tb["node_mode"] == "自動":
        step = marker_step(dfp["month"])
        df_nodes = (
            dfp.sort_values("month")
            .assign(_idx=dfp.sort_values("month").groupby("display_name").cumcount())
            .query("(_idx % @step) == 0")
        )
    elif tb["node_mode"] == "主要ノードのみ":
        g = dfp.sort_values("month").groupby("display_name")
        latest = g.tail(1)
        idxmax = dfp.loc[g["year_sum"].idxmax().dropna()]
        idxmin = dfp.loc[g["year_sum"].idxmin().dropna()]
        ystart = g.head(1)
        df_nodes = pd.concat([latest, idxmax, idxmin, ystart]).drop_duplicates(
            ["display_name", "month"]
        )
    elif tb["node_mode"] == "すべて":
        df_nodes = dfp.copy()
    else:
        df_nodes = dfp.iloc[0:0].copy()

    for name, d in df_nodes.groupby("display_name"):
        fig.add_scatter(
            x=d["month"],
            y=d["year_sum_disp"],
            mode="markers",
            name=name,
            legendgroup=name,
            showlegend=False,
            marker=dict(
                size=6,
                symbol="circle",
                line=dict(color=halo, width=1.6),
                opacity=0.95,
                color=line_colors.get(name, color_map.get(name)),
            ),
            customdata=np.stack(
                [d["display_name"], d["yoy_display"], d["delta_display"]],
                axis=-1,
            ),
            hovertemplate=hovertemplate,
        )

    if tb["enable_avoid"]:
        add_latest_labels_no_overlap(
            fig,
            dfp,
            label_col="label_with_yoy",
            x_col="month",
            y_col="year_sum_disp",
            max_labels=tb["max_labels"],
            min_gap_px=tb["gap_px"],
            alternate_side=tb["alt_side"],
        )

    fig = apply_elegant_theme(fig, theme=st.session_state.get("ui_theme", "light"))
    plot_height = height or int(tb.get("chart_height", 600))
    base_config = {
        "displaylogo": False,
        "scrollZoom": True,
        "doubleClick": "reset",
    }
    if config:
        base_config.update(config)
    render_plotly_with_spinner(
        fig,
        use_container_width=True,
        height=plot_height,
        config=base_config,
    )
    if (
        color_map
        and "yoy" in latest_snapshot.columns
        and latest_snapshot["yoy"].notna().any()
    ):
        st.caption(
            "色分けルール：青緑=前年同月比+5%以上、紫=±5%以内、サーモン=前年同月比-5%以下。"
        )
    return fig
