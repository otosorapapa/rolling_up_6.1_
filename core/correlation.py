"""Correlation utilities for dashboard analytics."""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def fisher_ci(r: float, n: int, zcrit: float = 1.96) -> tuple[float, float]:
    """Return the Fisher z confidence interval for a correlation coefficient."""

    if pd.isna(r):
        return np.nan, np.nan
    r = float(np.clip(r, -0.999999, 0.999999))
    if n <= 3:
        return np.nan, np.nan
    z = np.arctanh(r)
    se = 1 / np.sqrt(n - 3)
    lo, hi = np.tanh(z - zcrit * se), np.tanh(z + zcrit * se)
    return float(lo), float(hi)


def corr_table(
    df: pd.DataFrame,
    cols: Iterable[str],
    method: str = "pearson",
    *,
    pairwise: bool = False,
    min_periods: int = 3,
) -> pd.DataFrame:
    """Build tidy correlation table for selected columns."""

    cols = list(cols)
    rows: List[Dict[str, float | int | str]] = []

    if not cols:
        return pd.DataFrame(rows)

    if not pairwise:
        sub = df[cols].dropna()
        n = len(sub)
        if n == 0:
            return pd.DataFrame(rows)
        c = sub.corr(method=method)
        for i, a in enumerate(cols):
            for b in cols[i + 1 :]:
                r = c.loc[a, b]
                lo, hi = fisher_ci(r, n)
                sig = "有意(95%)" if (lo > 0 or hi < 0) else "n.s."
                rows.append(
                    {
                        "pair": f"{a}×{b}",
                        "r": float(r),
                        "n": n,
                        "ci_low": lo,
                        "ci_high": hi,
                        "sig": sig,
                    }
                )
        return pd.DataFrame(rows).sort_values("r", ascending=False)

    # Pairwise mode keeps the available observations for each pair individually.
    min_periods = max(int(min_periods), 2)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            sub = df[[a, b]].dropna()
            n = len(sub)
            if n < min_periods:
                rows.append(
                    {
                        "pair": f"{a}×{b}",
                        "r": np.nan,
                        "n": n,
                        "ci_low": np.nan,
                        "ci_high": np.nan,
                        "sig": "データ不足",
                    }
                )
                continue
            r = sub[a].corr(sub[b], method=method)
            lo, hi = fisher_ci(r, n)
            sig = "有意(95%)" if (lo > 0 or hi < 0) else "n.s."
            rows.append(
                {
                    "pair": f"{a}×{b}",
                    "r": float(r) if not pd.isna(r) else np.nan,
                    "n": n,
                    "ci_low": lo,
                    "ci_high": hi,
                    "sig": sig,
                }
            )

    if not rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(rows).sort_values("r", ascending=False, na_position="last")


def winsorize_frame(df: pd.DataFrame, cols: Iterable[str], p: float = 0.01) -> pd.DataFrame:
    """Winsorize selected columns by clipping both tails at percentage *p*."""

    out = df.copy()
    for col in cols:
        x = out[col]
        lo, hi = x.quantile(p), x.quantile(1 - p)
        out[col] = x.clip(lo, hi)
    return out


def maybe_log1p(df: pd.DataFrame, cols: Iterable[str], enable: bool) -> pd.DataFrame:
    """Apply log1p transform when *enable* is True and values are non-negative."""

    if not enable:
        return df
    out = df.copy()
    for col in cols:
        if (out[col] >= 0).all():
            out[col] = np.log1p(out[col])
    return out


def narrate_top_insights(tbl: pd.DataFrame, name_map: Dict[str, str], k: int = 3) -> List[str]:
    """Return human-readable highlights from correlation table."""

    if tbl.empty or "r" not in tbl:
        return []

    pos = tbl[tbl["r"] > 0].nlargest(k, "r")
    neg = tbl[tbl["r"] < 0].nsmallest(k, "r")
    lines: List[str] = []

    def jp(pair: str) -> str:
        a, b = pair.split("×")
        return f"「{name_map.get(a, a)}」と「{name_map.get(b, b)}」"

    for _, r in pos.iterrows():
        lines.append(
            f"{jp(r['pair'])} は **正の相関** (r={r['r']:.2f}, 95%CI [{r['ci_low']:.2f},{r['ci_high']:.2f}], n={int(r['n'])})."
        )
    for _, r in neg.iterrows():
        lines.append(
            f"{jp(r['pair'])} は **負の相関** (r={r['r']:.2f}, 95%CI [{r['ci_low']:.2f},{r['ci_high']:.2f}], n={int(r['n'])})."
        )
    return lines


def fit_line(x: pd.Series, y: pd.Series) -> tuple[float, float, float]:
    """Return slope, intercept and R² from a simple linear regression."""

    x = x.values.astype(float)
    y = y.values.astype(float)
    m, b = np.polyfit(x, y, 1)
    yhat = m * x + b
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return float(m), float(b), float(r2)

