"""Generative AI utilities for summarization and explanation.

This module provides light‑weight wrappers around an optional
`transformers` text2text generation pipeline.  If transformers or the
underlying model cannot be loaded (for example in offline
environments), the functions gracefully fall back to simple template
based implementations so that callers always receive some text output.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Optional

import numpy as np
import pandas as pd


@lru_cache(maxsize=1)
def _load_pipeline():  # pragma: no cover - heavy import
    """Return a cached text2text generation pipeline if available.

    The function tries to load a small Japanese/English capable model.
    If the environment does not have the required libraries or cannot
    download the model, ``None`` is returned so that callers can fall
    back to deterministic logic.
    """

    try:  # noqa: WPS501 - broad except to gracefully handle missing deps
        from transformers import pipeline

        # A very small model that supports summarisation and general
        # instruction following.  The size (~80MB) keeps CI reasonably
        # light while still demonstrating generative behaviour.
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-small",
        )
    except Exception:  # pragma: no cover - import failure branch
        return None


def summarize_dataframe(df: pd.DataFrame) -> str:
    """Summarise a numeric dataframe using a generative model.

    The dataframe's descriptive statistics are converted to text and
    fed into a text2text model.  When the model is unavailable a simple
    textual summary based on the statistics is returned.
    """

    stats = df.describe(include="all").fillna(0).to_string()
    pipe = _load_pipeline()
    if pipe is not None:
        prompt = (
            "以下の統計量からビジネス観点のハイライトを3点以内で箇条書きしてください。"
            "各行は短いセンテンスでまとめ、重要な指標名と方向性を含めてください。\n"
            + stats
        )
        result = pipe(prompt, max_new_tokens=120)[0]["generated_text"].strip()
        return result
    # Fallback summarisation
    lines = [f"行数{len(df)}・列数{len(df.columns)}。"]
    numeric = df.select_dtypes(include=[np.number]).copy()
    if not numeric.empty:
        means = numeric.mean().sort_values(ascending=False)
        top_metric = means.index[0]
        lines.append(f"平均が最も高い指標は{top_metric}（{means.iloc[0]:.2f}）。")
        if len(means) > 1:
            second = means.index[1]
            lines.append(f"次点は{second}（{means.iloc[1]:.2f}）。")
        if "yoy" in numeric.columns:
            yoy_pos = numeric["yoy"].gt(0).mean()
            lines.append(f"YoYプラスの比率は{yoy_pos * 100:.1f}%です。")
        if "delta" in numeric.columns:
            delta_avg = numeric["delta"].mean()
            trend = "増加" if delta_avg >= 0 else "減少"
            lines.append(f"Δの平均は{delta_avg:,.0f}で{trend}基調です。")
    else:
        sample = [line.strip() for line in stats.splitlines() if line.strip()][:2]
        lines.extend(sample)
    return " ".join(lines[:4])


def generate_comment(topic: str) -> str:
    """Generate a short comment about a topic.

    Parameters
    ----------
    topic:
        Free form text describing the subject of the comment.
    """

    pipe = _load_pipeline()
    if pipe is not None:
        prompt = f"次の内容について1文のコメントを日本語で書いてください: {topic}"
        return pipe(prompt, max_new_tokens=60)[0]["generated_text"].strip()
    return f"{topic}についてのコメント。"  # simple fallback


def explain_analysis(metrics: Dict[str, float]) -> str:
    """Explain analysis metrics using generative text.

    Metrics are provided as a mapping of indicator name to value.  The
    model is prompted to explain their meaning in a concise manner.  If
    the model is unavailable a deterministic sentence is composed.
    """

    summary = ", ".join(f"{k}={v}" for k, v in metrics.items()) or "指標なし"
    pipe = _load_pipeline()
    if pipe is not None:
        prompt = (
            "次の分析結果を分かりやすく説明してください:\n" + summary
        )
        return pipe(prompt, max_new_tokens=120)[0]["generated_text"].strip()
    return f"分析結果: {summary}"  # fallback


def generate_actions(metrics: Dict[str, float], focus: str) -> str:
    """Suggest follow-up actions based on KPI trends."""

    summary = ", ".join(f"{k}={v}" for k, v in metrics.items()) or "指標なし"
    pipe = _load_pipeline()
    if pipe is not None:
        prompt = (
            "次のKPIの状況を踏まえ、経営会議向けに3つの実行アクションを日本語で提案してください。"
            f"対象月: {focus}\n指標: {summary}"
        )
        return pipe(prompt, max_new_tokens=160)[0]["generated_text"].strip()

    yoy = metrics.get("yoy", 0.0) or 0.0
    delta = metrics.get("delta", 0.0) or 0.0
    actions = []
    if yoy < 0:
        actions.append("YoYがマイナスなので重点SKUの販促計画と価格見直しを検討する")
    else:
        actions.append("好調SKUの在庫確保と追加クロスセル施策を準備する")
    if delta < 0:
        actions.append("直近の落ち込み要因を営業ヒアリングで特定し対策を共有する")
    else:
        actions.append("増加分の持続可否を需給シミュレーションで検証する")
    actions.append("集中度を踏まえ、上位SKUへの依存リスクと補完商品の開発余地を評価する")
    return " / ".join(actions) + f"（対象月: {focus}）"


def answer_question(question: str, context: str) -> str:
    """Answer a free-form question using the provided context."""

    pipe = _load_pipeline()
    if pipe is not None:
        prompt = (
            "以下の事業データの要約を参考に、ビジネスアナリストとして質問に答えてください。"
            f"\nデータ要約:\n{context}\n\n質問:\n{question}\n"
            "回答は日本語で2~3文にまとめてください。"
        )
        return pipe(prompt, max_new_tokens=200)[0]["generated_text"].strip()

    parts = [p.strip() for p in context.split("｜") if p.strip()]
    lead = parts[0] if parts else "データが不足しています"
    detail = " / ".join(parts[1:3]) if len(parts) > 1 else ""
    detail_text = f" {detail}" if detail else ""
    return f"{lead}{detail_text}。この情報を前提に『{question}』への対応策を検討してください。"


def generate_anomaly_brief(anomalies: pd.DataFrame, top_n: int = 5) -> str:
    """Summarise anomaly detections highlighting notable cases."""

    if anomalies is None or anomalies.empty:
        return "異常値は検出されませんでした。"

    subset = anomalies.copy()
    if "score" in subset.columns:
        subset["score_abs"] = subset["score"].abs()
        subset = subset.sort_values("score_abs", ascending=False)
    else:
        subset["score_abs"] = 0.0
    subset = subset.head(top_n)

    pipe = _load_pipeline()
    if pipe is not None:
        bullet = []
        for _, row in subset.iterrows():
            name = row.get("product_name") or row.get("product_code") or "不明SKU"
            month = row.get("month", "-")
            score = row.get("score", 0.0)
            yoy = row.get("yoy")
            delta = row.get("delta")
            yoy_txt = f"YoY {yoy * 100:.1f}%" if yoy is not None and not pd.isna(yoy) else "YoY情報なし"
            delta_txt = (
                f"Δ {delta:,.0f}" if delta is not None and not pd.isna(delta) else "Δ情報なし"
            )
            bullet.append(f"商品:{name} 月:{month} スコア:{score:.2f} {yoy_txt} {delta_txt}")
        prompt = (
            "以下の異常検知結果について、重要ポイントをビジネス視点で箇条書きしてください。"
            "全体傾向と注目SKUを含め、3行以内でまとめてください。\n" + "\n".join(bullet)
        )
        return pipe(prompt, max_new_tokens=150)[0]["generated_text"].strip()

    scores = pd.to_numeric(
        anomalies.get("score", pd.Series(dtype=float)), errors="coerce"
    )
    total = len(anomalies)
    pos = int((scores > 0).sum())
    neg = int((scores < 0).sum())
    parts = [f"異常値{total}件（上振れ{pos}件・下振れ{neg}件）。"]
    highlights = []
    for _, row in subset.iterrows():
        name = row.get("product_name") or row.get("product_code") or "不明SKU"
        month = row.get("month", "-")
        score = row.get("score")
        yoy = row.get("yoy")
        delta = row.get("delta")
        detail = [f"スコア{score:.2f}" if score is not None and not pd.isna(score) else "スコア情報なし"]
        if yoy is not None and not pd.isna(yoy):
            detail.append(f"YoY {yoy * 100:.1f}%")
        if delta is not None and not pd.isna(delta):
            detail.append(f"Δ {delta:,.0f}")
        highlights.append(f"{name}（{month}）: " + " / ".join(detail))
    if highlights:
        parts.append("注目SKU: " + "; ".join(highlights))
    return " ".join(parts)
