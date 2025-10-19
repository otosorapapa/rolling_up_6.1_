
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta

# ---------- Data classes ----------
@dataclass
class FactRecord:
    product_code: str
    product_name: str
    month: str  # 'YYYY-MM'
    sales_amount_jpy: float
    is_missing: bool = False


# ---------- Utilities ----------
def normalize_month_key(value: str) -> str:
    """Normalize a variety of month formats to 'YYYY-MM'."""
    if pd.isna(value):
        raise ValueError("Month value is NaN")
    v = str(value).strip()
    # Fast path exact patterns
    for fmt in ("%Y-%m", "%Y/%m", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.strptime(v, fmt)
            return dt.strftime("%Y-%m")
        except Exception:
            pass
    # Try pandas to_datetime coercion
    try:
        dt = pd.to_datetime(v, errors="raise", dayfirst=False)
        return dt.strftime("%Y-%m")
    except Exception:
        pass
    # Regex fallback like 202401 or 2024-01-01 00:00
    m = re.search(r"(20\d{2})[^\d]?(0[1-9]|1[0-2])", v)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    raise ValueError(f"Unrecognized month key: {value}")


def month_range(min_m: str, max_m: str) -> List[str]:
    """Inclusive month range list ['YYYY-MM', ...]."""
    start = datetime.strptime(min_m, "%Y-%m")
    end = datetime.strptime(max_m, "%Y-%m")
    out = []
    cur = start
    while cur <= end:
        out.append(cur.strftime("%Y-%m"))
        cur += relativedelta(months=1)
    return out


# ---------- Parsing & Normalization ----------
def parse_uploaded_table(df: pd.DataFrame,
                         product_name_col: Optional[str] = None,
                         product_code_col: Optional[str] = None) -> pd.DataFrame:
    """
    Accepts a wide table: first columns are product metadata, remaining columns are months.
    Returns long df: ['product_code','product_name','month','sales_amount_jpy']
    """
    cols = list(df.columns)
    # Auto-detect product name column if not provided: first column
    if product_name_col is None:
        product_name_col = cols[0]
    if product_name_col not in cols:
        raise ValueError("Product name column not found")

    # Auto-detect code column if provided
    code_present = False
    if product_code_col and product_code_col in cols:
        code_present = True

    # Detect month columns (anything convertible to YYYY-MM)
    month_cols = []
    for c in cols:
        if c == product_name_col or (code_present and c == product_code_col):
            continue
        try:
            _ = normalize_month_key(c)
            month_cols.append(c)
        except Exception:
            # not a month col
            pass
    if not month_cols:
        raise ValueError("No month columns detected")

    # Normalize month col names
    month_map = {c: normalize_month_key(c) for c in month_cols}
    df = df.rename(columns=month_map)

    # Melt
    id_vars = [product_name_col]
    if code_present:
        id_vars.insert(0, product_code_col)
    long_df = df.melt(id_vars=id_vars, var_name="month", value_name="sales_amount_jpy")

    # Normalize values
    long_df["month"] = long_df["month"].apply(normalize_month_key)
    # Ensure numeric
    long_df["sales_amount_jpy"] = pd.to_numeric(long_df["sales_amount_jpy"], errors="coerce")

    # Product code
    if not code_present:
        # Generate deterministic codes based on name index
        codes = {name: f"P{idx+1:06d}" for idx, name in enumerate(long_df[product_name_col].drop_duplicates().tolist())}
        long_df["product_code"] = long_df[product_name_col].map(codes)
    else:
        long_df = long_df.rename(columns={product_code_col: "product_code"})

    long_df = long_df.rename(columns={product_name_col: "product_name"})
    long_df["is_missing"] = long_df["sales_amount_jpy"].isna()

    return long_df[["product_code","product_name","month","sales_amount_jpy","is_missing"]]


def fill_missing_months(long_df: pd.DataFrame, policy: str = "zero_fill") -> pd.DataFrame:
    """
    Ensure every product has every month between global min and max.
    policy: 'zero_fill' => fill with 0; 'mark_missing' => leave NaN and is_missing=True
    """
    # Global month range
    min_m = long_df["month"].min()
    max_m = long_df["month"].max()
    all_months = month_range(min_m, max_m)

    products = long_df[["product_code","product_name"]].drop_duplicates()
    idx = pd.MultiIndex.from_product([products["product_code"], all_months], names=["product_code","month"])
    base = pd.DataFrame(index=idx).reset_index()

    base = base.merge(products, on="product_code", how="left")
    merged = base.merge(long_df, on=["product_code","product_name","month"], how="left")

    if policy == "zero_fill":
        merged["is_missing"] = merged["sales_amount_jpy"].isna() | merged["is_missing"].fillna(False)
        merged["sales_amount_jpy"] = merged["sales_amount_jpy"].fillna(0.0)
    else:
        # keep NaN, set is_missing True where NaN
        merged["is_missing"] = merged["sales_amount_jpy"].isna() | merged["is_missing"].fillna(False)

    # Order
    merged = merged.sort_values(["product_code","month"], ignore_index=True)
    return merged


# ---------- Metrics ----------
def compute_year_rolling(long_df: pd.DataFrame, window: int = 12, policy: str = "zero_fill") -> pd.DataFrame:
    """
    For each product_code, compute rolling 12M sum (year_sum), delta (mom), yoy, with mark_missing behavior.
    Returns df: ['product_code','product_name','month','year_sum','delta','yoy']
    """
    out_list = []
    for code, g in long_df.groupby("product_code", sort=False):
        g = g.sort_values("month")
        vals = g["sales_amount_jpy"].tolist()
        months = g["month"].tolist()
        miss = g["is_missing"].tolist()

        year_sum = [np.nan]*len(vals)
        delta = [np.nan]*len(vals)
        yoy = [np.nan]*len(vals)

        for i in range(len(vals)):
            if i + 1 < window:
                continue
            slice_vals = vals[i+1-window:i+1]
            slice_miss = miss[i+1-window:i+1]
            if policy == "mark_missing" and any(slice_miss):
                # skip computing if any missing in window
                continue
            s = np.nansum([0.0 if np.isnan(v) else v for v in slice_vals])
            year_sum[i] = s

        # delta
        for i in range(1, len(year_sum)):
            if not np.isnan(year_sum[i]) and not np.isnan(year_sum[i-1]):
                delta[i] = year_sum[i] - year_sum[i-1]

        # yoy
        for i in range(len(year_sum)):
            j = i - window
            if i >= window and (not np.isnan(year_sum[i])) and j >= 0 and (not np.isnan(year_sum[j])) and year_sum[j] != 0:
                yoy[i] = (year_sum[i] - year_sum[j]) / year_sum[j]

        tmp = pd.DataFrame({
            "product_code": code,
            "product_name": g["product_name"].iloc[0],
            "month": months,
            "year_sum": year_sum,
            "delta": delta,
            "yoy": yoy
        })
        out_list.append(tmp)

    year_df = pd.concat(out_list, ignore_index=True)
    return year_df


def slope_beta(values: List[float]) -> float:
    """OLS slope for sequential x=0..n-1"""
    if values is None or len(values) < 2:
        return 0.0
    n = len(values)
    x = np.arange(n, dtype=float)
    y = np.array(values, dtype=float)
    x_bar = x.mean()
    y_bar = np.nanmean(y)
    sxx = np.sum((x - x_bar)**2)
    sxy = np.nansum((x - x_bar) * (y - y_bar))
    if sxx == 0:
        return 0.0
    return float(sxy / sxx)


def compute_slopes(year_df: pd.DataFrame, last_n: int = 12) -> pd.DataFrame:
    """Add slope_beta column per product/month, computed over last_n year_sum values."""
    year_df = year_df.sort_values(["product_code","month"])
    out = []
    for code, g in year_df.groupby("product_code"):
        ys = g["year_sum"].tolist()
        months = g["month"].tolist()
        slopes = [np.nan]*len(ys)
        for i in range(len(ys)):
            start = max(0, i - last_n + 1)
            window_vals = [v for v in ys[start:i+1] if not np.isnan(v)]
            slopes[i] = slope_beta(window_vals) if len(window_vals) >= 2 else np.nan
        gg = g.copy()
        gg["slope_beta"] = slopes
        out.append(gg)
    return pd.concat(out, ignore_index=True)


def abc_classification(year_df: pd.DataFrame, end_month: str, cut_a: float = 0.80, cut_b: float = 0.95) -> pd.DataFrame:
    """Assign A/B/C by cumulative contribution at end_month."""
    snap = year_df[year_df["month"] == end_month].dropna(subset=["year_sum"]).copy()
    snap = snap.sort_values("year_sum", ascending=False)
    total = snap["year_sum"].sum()
    if total <= 0:
        snap["abc_class"] = "C"
        return snap[["product_code","abc_class"]]
    snap["share"] = snap["year_sum"] / total
    snap["cum_share"] = snap["share"].cumsum()
    snap["abc_class"] = np.where(snap["cum_share"] <= cut_a, "A",
                         np.where(snap["cum_share"] <= cut_b, "B", "C"))
    return snap[["product_code","abc_class"]]


def compute_hhi(year_df: pd.DataFrame, end_month: str) -> float:
    snap = year_df[year_df["month"] == end_month].dropna(subset=["year_sum"]).copy()
    total = snap["year_sum"].sum()
    if total <= 0:
        return 0.0
    snap["share"] = snap["year_sum"] / total
    return float(np.sum(np.square(snap["share"])))


def build_alerts(year_df: pd.DataFrame,
                 end_month: str,
                 yoy_threshold: float = -0.10,
                 delta_threshold: float = -300000.0,
                 slope_threshold: float = -1.0) -> pd.DataFrame:
    """
    Create alert records on end_month snapshot based on thresholds.
    slope_threshold is absolute slope threshold (negative means decreasing).
    """
    snap = year_df[year_df["month"] == end_month].copy()
    alerts = []
    for _, row in snap.iterrows():
        prod = row["product_code"]
        name = row["product_name"]
        ys = row["year_sum"]
        yy = row["yoy"]
        dl = row["delta"]
        sl = row.get("slope_beta", np.nan)

        if not pd.isna(yy) and yy <= yoy_threshold:
            alerts.append({"product_code": prod, "product_name": name, "month": end_month,
                           "metric": "yoy", "threshold": yoy_threshold, "actual": yy, "level": "warn"})
        if not pd.isna(dl) and dl <= delta_threshold:
            alerts.append({"product_code": prod, "product_name": name, "month": end_month,
                           "metric": "delta", "threshold": delta_threshold, "actual": dl, "level": "warn"})
        if not pd.isna(sl) and sl <= slope_threshold:
            alerts.append({"product_code": prod, "product_name": name, "month": end_month,
                           "metric": "slope", "threshold": slope_threshold, "actual": sl, "level": "warn"})
    return pd.DataFrame(alerts)


def aggregate_overview(year_df: pd.DataFrame, end_month: str) -> Dict[str, float]:
    """Compute overall KPIs for end_month and its lineage (YoY, delta derived from totals)."""
    # Build total series over months
    totals = year_df.groupby("month", as_index=False)["year_sum"].sum().sort_values("month")
    totals["delta"] = totals["year_sum"].diff()
    # yoy vs t-12
    months = totals["month"].tolist()
    yoy_vals = [np.nan] * len(months)
    for i, m in enumerate(months):
        j = i - 12
        if j >= 0 and totals.loc[j, "year_sum"] != 0 and not pd.isna(totals.loc[j, "year_sum"]) and not pd.isna(totals.loc[i, "year_sum"]):
            yoy_vals[i] = (totals.loc[i, "year_sum"] - totals.loc[j, "year_sum"]) / totals.loc[j, "year_sum"]
    totals["yoy"] = yoy_vals

    # pick end_month
    row = totals[totals["month"] == end_month]
    if row.empty:
        return {"total_year_sum": 0.0, "yoy": np.nan, "delta": np.nan}
    r = row.iloc[0]
    return {"total_year_sum": float(r["year_sum"]),
            "yoy": (float(r["yoy"]) if not pd.isna(r["yoy"]) else None),
            "delta": (float(r["delta"]) if not pd.isna(r["delta"]) else None)}


def get_comparables(year_df: pd.DataFrame,
                     end_month: str,
                     target_code: str,
                     mode: str = 'abs',
                     low: Optional[float] = None,
                     high: Optional[float] = None,
                     rank_k: int = 10,
                     filters: Optional[Dict] = None,
                     tags_map: Optional[Dict[str, List[str]]] = None,
                     limit: Optional[int] = None) -> pd.DataFrame:
    """Select comparable SKU snapshot rows based on target and range.

    Parameters
    ----------
    year_df : pd.DataFrame
        Yearly summary long-form dataframe.
    end_month : str
        Target month for snapshot.
    target_code : str
        Base product code for comparison.
    mode : str
        'abs' for absolute difference, 'pct' for percentage, 'rank' for
        rank based window.
    low, high : float
        Range definition depending on mode. For 'abs' and 'pct' modes,
        they represent +/- offset from target. If None, open-ended.
    rank_k : int
        Rank window for 'rank' mode (±rank_k).
    filters : dict
        Optional filters such as {'abc': ['A','B'], 'tags': ['x'],
        'yoy_le': -0.1, 'delta_le': -3e5, 'slope_le': -1.0}.
    tags_map : dict
        Mapping product_code -> list of tag strings.
    limit : int
        Optional maximum number of rows to return.
    """
    snap = year_df[year_df['month'] == end_month].copy()
    if snap.empty:
        return pd.DataFrame()
    snap = snap.dropna(subset=['year_sum'])
    snap = snap.sort_values('year_sum', ascending=False)
    snap['rank'] = np.arange(1, len(snap) + 1)

    # ABC classification merge
    abc_df = abc_classification(year_df, end_month)
    snap = snap.merge(abc_df, on='product_code', how='left')

    # Tags column
    if tags_map is not None:
        snap['tags'] = snap['product_code'].map(lambda c: ','.join(tags_map.get(c, [])))
    else:
        snap['tags'] = ''

    target_row = snap[snap['product_code'] == target_code]
    if target_row.empty:
        return pd.DataFrame()
    target_val = target_row['year_sum'].iloc[0]
    target_rank = target_row['rank'].iloc[0]

    if mode == 'abs':
        lo = target_val + (low if low is not None else -np.inf)
        hi = target_val + (high if high is not None else np.inf)
        cond = (snap['year_sum'] >= lo) & (snap['year_sum'] <= hi)
    elif mode == 'pct':
        lo = target_val * (1 + (low if low is not None else -np.inf))
        hi = target_val * (1 + (high if high is not None else np.inf))
        cond = (snap['year_sum'] >= lo) & (snap['year_sum'] <= hi)
    else:  # rank
        lo = max(1, target_rank - rank_k)
        hi = target_rank + rank_k
        cond = (snap['rank'] >= lo) & (snap['rank'] <= hi)

    out = snap[cond].copy()

    if filters:
        if filters.get('abc'):
            out = out[out['abc_class'].isin(filters['abc'])]
        if filters.get('tags'):
            tagset = set(filters['tags'])
            out = out[out['product_code'].apply(lambda c: bool(tagset.intersection(tags_map.get(c, []))) if tags_map else False)]
        if filters.get('yoy_le') is not None:
            out = out[out['yoy'] <= filters['yoy_le']]
        if filters.get('delta_le') is not None:
            out = out[out['delta'] <= filters['delta_le']]
        if filters.get('slope_le') is not None and 'slope_beta' in out.columns:
            out = out[out['slope_beta'] <= filters['slope_le']]

    if limit is not None:
        out = out.head(int(limit))

    cols = ['product_code', 'product_name', 'year_sum', 'yoy', 'delta',
            'slope_beta', 'abc_class', 'tags', 'rank']
    return out[cols]


def build_indexed_series(year_df: pd.DataFrame,
                         codes: List[str],
                         base: str = 'first_non_nan') -> pd.DataFrame:
    """Return long-form dataframe of index=100 series per SKU."""
    df = year_df[year_df['product_code'].isin(codes)].copy()
    pivot = df.pivot(index='month', columns='product_code', values='year_sum').sort_index()
    if base == 'first_non_nan':
        base_vals = pivot.apply(lambda col: col.dropna().iloc[0] if not col.dropna().empty else np.nan)
    else:
        base_vals = pivot.iloc[0]
    indexed = pivot.copy()
    for c in indexed.columns:
        b = base_vals.get(c, np.nan)
        if pd.isna(b) or b == 0:
            indexed[c] = np.nan
        else:
            indexed[c] = (indexed[c] / b) * 100.0
    long_df = indexed.reset_index().melt(id_vars='month', var_name='product_code', value_name='index_value')
    return long_df


# ---------- Snapshot & Band Utilities ----------
def latest_yearsum_snapshot(df_year: pd.DataFrame, end_month: str) -> pd.DataFrame:
    """指定した月の年計スナップショットを返す。

    パラメータ
    ----------
    df_year : pd.DataFrame
        `compute_year_rolling` の結果である年計ロングデータ。
    end_month : str
        対象とする終端月 (YYYY-MM)。

    戻り値
    ----------
    pd.DataFrame
        product_code, product_name, year_sum, rank, yoy, delta の列を持つ
        スナップショット。slope_beta 列が存在する場合は併せて含める。
    """
    snap = df_year[df_year["month"] == end_month].copy()
    if snap.empty:
        return pd.DataFrame(
            columns=["product_code", "product_name", "year_sum", "rank", "yoy", "delta", "slope_beta"]
        )
    snap = snap.dropna(subset=["year_sum"])
    snap = snap.sort_values("year_sum", ascending=False)
    snap["rank"] = np.arange(1, len(snap) + 1)
    cols = ["product_code", "product_name", "year_sum", "rank", "yoy", "delta"]
    if "slope_beta" in snap.columns:
        cols.append("slope_beta")
    return snap[cols]


def resolve_band(snapshot: pd.DataFrame, mode: str, params: Dict) -> Tuple[float, float]:
    """UIで指定されたモードとパラメータからバンド下限・上限を計算する。

    mode は以下をサポートする:
        - 'amount': 金額指定
        - 'two_products': 2商品の年計を基準
        - 'percentile': 百分位
        - 'rank': 順位帯
        - 'target_near': 基準商品近傍
    """
    if snapshot.empty:
        return (np.nan, np.nan)

    if mode == "amount":
        low = params.get("low_amount", -np.inf)
        high = params.get("high_amount", np.inf)
    elif mode == "two_products":
        a = snapshot[snapshot["product_code"] == params.get("prod_a")]["year_sum"].iloc[0]
        b = snapshot[snapshot["product_code"] == params.get("prod_b")]["year_sum"].iloc[0]
        low, high = sorted([float(a), float(b)])
    elif mode == "percentile":
        p_low = params.get("p_low", 0) / 100.0
        p_high = params.get("p_high", 100) / 100.0
        low = float(snapshot["year_sum"].quantile(p_low))
        high = float(snapshot["year_sum"].quantile(p_high))
    elif mode == "rank":
        r_low = params.get("r_low", 1)
        r_high = params.get("r_high", len(snapshot))
        subset = snapshot[(snapshot["rank"] >= r_low) & (snapshot["rank"] <= r_high)]
        low = float(subset["year_sum"].min())
        high = float(subset["year_sum"].max())
    else:  # target_near
        row = snapshot[snapshot["product_code"] == params.get("target_code")]
        if row.empty:
            return (np.nan, np.nan)
        base = float(row["year_sum"].iloc[0])
        if params.get("by", "pct") == "amt":
            width = float(params.get("width", 0.0))
            low = base - width
            high = base + width
        else:
            width = float(params.get("width", 0.0))
            low = base * (1 - width)
            high = base * (1 + width)
    return (low, high)


def filter_products_by_band(snapshot: pd.DataFrame, low: float, high: float) -> List[str]:
    """年計値が指定バンドに含まれる商品コードを返す。"""
    if snapshot.empty:
        return []
    cond = (snapshot["year_sum"] >= low) & (snapshot["year_sum"] <= high)
    return snapshot[cond]["product_code"].tolist()


def get_yearly_series(df_year: pd.DataFrame,
                      codes: Optional[List[str]] = None,
                      start: Optional[str] = None,
                      end: Optional[str] = None,
                      metric: str = "year_sum") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """年計ロングデータから指定SKUの縦持ち・横持ちデータを取得する。

    Parameters
    ----------
    df_year : pd.DataFrame
        年計ロングデータ。
    codes : list[str], optional
        対象とする商品コードリスト。None の場合は全件。
    start, end : str, optional
        期間フィルタ (YYYY-MM)。
    metric : str
        取得する数値列名。既定は 'year_sum'。

    Returns
    -------
    (long_df, pivot_df)
        long_df: フィルタ後のロングデータ
        pivot_df: 行=month, 列=product_code のピボットテーブル
    """
    df = df_year.copy()
    if codes is not None:
        df = df[df["product_code"].isin(codes)]
    if start is not None:
        df = df[df["month"] >= start]
    if end is not None:
        df = df[df["month"] <= end]
    pivot = df.pivot_table(
        index="month",
        columns="product_code",
        values=metric,
        aggfunc="sum",
    ).sort_index()
    return df, pivot


def top_growth_codes(df_year: pd.DataFrame, end_month: str, window: int = 6, top: int = 10) -> List[str]:
    """直近windowカ月の伸長上位商品コードを返す。"""
    if df_year.empty:
        return []
    end_dt = pd.to_datetime(end_month)
    start_dt = end_dt - pd.DateOffset(months=window)
    sub = df_year[(pd.to_datetime(df_year["month"]) >= start_dt) & (pd.to_datetime(df_year["month"]) <= end_dt)]
    pivot = sub.pivot_table(index="month", columns="product_code", values="year_sum").sort_index()
    if pivot.empty or len(pivot) < 2:
        return []
    diff = pivot.iloc[-1] - pivot.iloc[0]
    diff = diff.dropna().sort_values(ascending=False)
    return diff.head(top).index.tolist()


def trend_last6(series: pd.Series) -> dict:
    """直近6か月のトレンドスコアを計算する。

    Parameters
    ----------
    series : pd.Series
        月次の年計値。昇順に並んでいる必要がある。

    Returns
    -------
    dict
        slope: 月あたり増加額
        ratio: 平均年計に対する傾きの比率
        group: プレースホルダー（分類は呼び出し側で実施）
    """
    s = series.dropna().tail(6)
    if len(s) < 3:
        return {"slope": 0.0, "ratio": 0.0, "group": "横ばい"}
    x = np.arange(len(s))
    slope = np.polyfit(x, s.values, 1)[0]
    ratio = slope / max(1.0, s.mean())
    return {"slope": float(slope), "ratio": float(ratio), "group": None}


def slope_last_n(y: pd.Series, n: int = 6):
    """末尾n点で単回帰の傾き（円/月）と%/月を返す。

    n<=0（またはNone）の場合は全期間を対象とし、
    データ点が2点未満ならNaNを返す。
    """
    s = y.dropna()
    s = s if (n is None or n <= 0) else s.tail(n)
    L = len(s)
    if L < 2:
        return np.nan, np.nan  # データが足りない
    x = np.arange(L, dtype=float)
    if L == 2:
        m = s.iloc[1] - s.iloc[0]
    else:
        m, _ = np.polyfit(x, s.values.astype(float), 1)
    ratio = m / max(1.0, s.mean())  # %/月相当
    return float(m), float(ratio)


def slopes_snapshot(df_long: pd.DataFrame, x_col="month", y_col="year_sum",
                    key_col="product_code", n=6):
    """商品ごと末尾n点の傾きを一括算出。"""
    g = df_long.sort_values(x_col).groupby(key_col, as_index=False)
    rows = []
    for k, d in g:
        m, r = slope_last_n(d[y_col], n=n)
        rows.append({key_col: k, "slope_yen": m, "slope_ratio": r})
    snap = pd.DataFrame(rows)
    # zスコア
    mu, sd = snap["slope_yen"].mean(), snap["slope_yen"].std(ddof=0) or 1.0
    snap["slope_z"] = (snap["slope_yen"] - mu) / sd
    return snap


def shape_flags(df_long: pd.DataFrame, key_col="product_code",
                x_col="month", y_col="year_sum", window=12, alpha_ratio=0.02, amp_ratio=0.06):
    """
    '急勾配'：|傾きz| >= しきい値（後述UI）
    '山'：window内で最大点を中心に、前半の平均差分>+α かつ 後半<-α、
          かつ (最大-両端)/平均 >= 振幅比
    '谷'：最小点を中心に、前半<-α かつ 後半>+α、
          かつ (両端-最小)/平均 >= 振幅比
    αは平均値×alpha_ratio（月あたり）、振幅は系列平均に対する比。
    """
    out = []
    for code, d in df_long.sort_values(x_col).groupby(key_col):
        s = d[y_col].dropna().tail(window)
        if len(s) < max(6, window//2):
            out.append({key_col: code, "is_mountain": False, "is_valley": False})
            continue
        s_smooth = s.rolling(3, center=True, min_periods=1).mean()
        m = s_smooth.mean()
        alpha = m * alpha_ratio
        # 山
        tmax = s_smooth.idxmax()
        pre = s_smooth.loc[:tmax].diff().dropna()
        post = s_smooth.loc[tmax:].diff().dropna()
        is_mtn = (pre.mean() > alpha) and (post.mean() < -alpha) and \
                 ((s_smooth.max() - (s_smooth.iloc[0]+s_smooth.iloc[-1])/2) / max(1.0, m) >= amp_ratio)
        # 谷
        tmin = s_smooth.idxmin()
        pre2 = s_smooth.loc[:tmin].diff().dropna()
        post2 = s_smooth.loc[tmin:].diff().dropna()
        is_val = (pre2.mean() < -alpha) and (post2.mean() > alpha) and \
                 (((s_smooth.iloc[0]+s_smooth.iloc[-1])/2 - s_smooth.min()) / max(1.0, m) >= amp_ratio)
        out.append({key_col: code, "is_mountain": bool(is_mtn), "is_valley": bool(is_val)})
    return pd.DataFrame(out)


# ---------- Forecast & Anomaly Utilities ----------

def forecast_linear_band(y: pd.Series, window:int=12, horizon:int=6, k:float=2.0, robust:bool=False):
    """
    末尾window点で単回帰→将来horizon点を線形予測し、残差のσで±k帯。
    戻り値: (fcast: np.ndarray, lo: np.ndarray, hi: np.ndarray)
    """
    s = pd.Series(y).dropna()
    if len(s) < max(3, window):
        return np.array([]), np.array([]), np.array([])
    y_win = s.tail(window).to_numpy(dtype=float)
    x = np.arange(len(y_win), dtype=float)
    m, b = np.polyfit(x, y_win, 1)
    y_hat = m*x + b
    resid = y_win - y_hat
    if robust:
        sigma = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    else:
        sigma = resid.std(ddof=1) if len(resid) > 1 else 0.0
    x_f = np.arange(len(y_win), len(y_win)+horizon, dtype=float)
    f = m*x_f + b
    return f, f - k*sigma, f + k*sigma

def forecast_holt_linear(y: pd.Series, alpha:float=0.4, beta:float=0.2, horizon:int=6):
    """
    Holtの線形成分のみ（季節無し）を自前実装。戻り値: 予測np.ndarray
    """
    s = pd.Series(y).dropna()
    if len(s) < 3:
        return np.array([])
    l, b = float(s.iloc[0]), float(s.iloc[1]-s.iloc[0])
    for val in s:
        l_new = alpha*val + (1-alpha)*(l + b)
        b_new = beta*(l_new - l) + (1-beta)*b
        l, b = l_new, b_new
    return np.array([l + (i+1)*b for i in range(horizon)])

def band_from_moving_stats(y: pd.Series, window:int=12, horizon:int=6, k:float=2.0, robust:bool=False):
    """
    移動平均±kσ/MAD に基づく水平バンド（予測値は直近平均）
    """
    s = pd.Series(y).dropna()
    if len(s) < 3:
        return np.array([]), np.array([]), np.array([])
    tail = s.tail(window)
    mu = tail.mean()
    if robust:
        med = tail.median()
        sigma = 1.4826 * np.median(np.abs(tail - med))
    else:
        sigma = tail.std(ddof=1)
    f = np.full(horizon, mu, dtype=float)
    return f, f - k*sigma, f + k*sigma

def detect_linear_anomalies(y: pd.Series, window:int=12, threshold:float=2.5, robust:bool=False) -> pd.DataFrame:
    """
    ローカル線形回帰の残差に基づく異常検知。
    残差のzスコアまたはMADスコアがthresholdを超える点を返す。
    戻り値: DataFrame(month,value,score)
    """
    s = pd.Series(y).dropna()
    if len(s) < window + 1:
        return pd.DataFrame(columns=["month","value","score"])
    months = s.index.tolist()
    out = []
    for i in range(window, len(s)):
        y_win = s.iloc[i-window:i].to_numpy(dtype=float)
        x = np.arange(len(y_win), dtype=float)
        m, b = np.polyfit(x, y_win, 1)
        y_hat = m*len(y_win) + b
        resid = float(s.iloc[i]) - y_hat
        fit_resid = y_win - (m*x + b)
        if robust:
            sigma = 1.4826 * np.median(np.abs(fit_resid - np.median(fit_resid)))
        else:
            sigma = fit_resid.std(ddof=1) if len(fit_resid) > 1 else 0.0
        score = resid / sigma if sigma > 0 else 0.0
        if np.abs(score) >= threshold:
            out.append({"month": months[i], "value": float(s.iloc[i]), "score": float(score)})
    return pd.DataFrame(out)

