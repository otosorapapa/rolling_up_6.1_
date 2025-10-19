"""Streamlit module for purchasing correlation clustering.

README: このモジュールは既存の Streamlit アプリに対し、併買商品の相関→グラフ→クラスタリング
までをワンストップで実行する機能を提供する。`render_correlation_category_module()` を
呼び出せば UI 全体が描画され、取引明細・ピボット・相関行列のいずれかを入力として
クラスタリングと可視化、CSV エクスポートを行える。

推奨初期設定：Louvain + 閾値0.3 + 最小クラスタサイズ3。SKU が 1,500 を超えて描画が
重いときは「売上（登場回数）上位500SKU」＋「エッジ上位30,000件」程度で十分実務的。

主な関数
----------
* build_matrix: 取引明細からユーザー/トランザクション×商品行列を生成
* compute_correlation: Pearson / Spearman / Jaccard の相関・類似度を計算
* graph_from_corr: 相関行列から NetworkX グラフを構築
* detect_communities: Louvain または階層クラスタで商品クラスタを抽出
* compute_cluster_metrics: 中心性・サポートなどノード/クラスタ指標を算出
* recommend_threshold: 相関分布から推奨閾値を提案
* render_correlation_category_module: Streamlit ページ描画エントリポイント

例外時は UI 上で警告を出し、推奨パラメータを提示する。`make_demo_transactions()` を
利用すればセルフテスト用のダミーデータ（A-B-C の併買が強く出る等）を生成できる。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from networkx import Graph
from networkx.algorithms.community import greedy_modularity_communities, modularity
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances

try:  # python-louvain は任意依存（インストールできない環境向けのフォールバック付き）
    from community import community_louvain
except ImportError:  # pragma: no cover - optional dependency
    community_louvain = None

from core.plot_utils import apply_elegant_theme, render_plotly_with_spinner


@dataclass
class MatrixBuildResult:
    """取引明細→ピボット変換の結果。"""

    matrix: pd.DataFrame
    support_counts: pd.Series
    total_events: int


def _read_table(upload) -> pd.DataFrame:
    """Streamlit の UploadedFile から DataFrame を生成する。"""

    if upload is None:
        return pd.DataFrame()
    name = getattr(upload, "name", "uploaded")
    suffix = Path(name).suffix.lower()
    try:
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(upload)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(upload)
        upload.seek(0)
        try:
            return pd.read_csv(upload)
        except UnicodeDecodeError:
            upload.seek(0)
            return pd.read_csv(upload, encoding="cp932")
    finally:
        upload.seek(0)


def make_demo_transactions(n_transactions: int = 200, seed: int = 42) -> pd.DataFrame:
    """併買パターンが分かりやすいダミー取引データを生成する。"""

    rng = np.random.default_rng(seed)
    base_products = list("ABCDEFGHIJ")
    dates = pd.date_range("2024-01-01", periods=90, freq="D")
    records: List[Dict[str, Any]] = []
    for tid in range(n_transactions):
        transaction_id = f"T{tid:05d}"
        user_id = f"U{rng.integers(0, 80):04d}"
        date = rng.choice(dates)
        basket_type = rng.choice(["ABC", "DE", "RANDOM"], p=[0.35, 0.2, 0.45])
        if basket_type == "ABC":
            basket = ["A", "B", "C"]
        elif basket_type == "DE":
            basket = ["D", "E"]
        else:
            basket = list(rng.choice(base_products[5:], size=rng.integers(1, 4), replace=False))
        if rng.random() < 0.1:
            basket.append(rng.choice(base_products))
        for pid in basket:
            qty = int(rng.integers(1, 4))
            price = float(rng.integers(100, 600))
            records.append(
                {
                    "transaction_id": transaction_id,
                    "user_id": user_id,
                    "product_id": pid,
                    "product_name": f"商品{pid}",
                    "qty": qty,
                    "amount": qty * price,
                    "date": date,
                }
            )
    return pd.DataFrame(records)


def build_matrix(
    df: pd.DataFrame,
    id_col: str,
    prod_col: str,
    val_col: Optional[str] = None,
    *,
    binary: bool = True,
) -> MatrixBuildResult:
    """取引明細からユーザー/トランザクション×商品行列を作成する。"""

    if df.empty:
        return MatrixBuildResult(pd.DataFrame(), pd.Series(dtype=float), 0)
    if id_col not in df.columns or prod_col not in df.columns:
        raise KeyError("指定した列がデータ内に存在しません。")

    work = df[[id_col, prod_col] + ([val_col] if val_col else [])].copy()
    work = work.dropna(subset=[id_col, prod_col])
    work[id_col] = work[id_col].astype(str)
    work[prod_col] = work[prod_col].astype(str)

    value_col = val_col if val_col else "_count"
    if not val_col:
        work[value_col] = 1
    pivot = (
        work.pivot_table(
            index=id_col,
            columns=prod_col,
            values=value_col,
            aggfunc="sum",
            fill_value=0,
        )
        .sort_index()
        .astype(float)
    )
    if binary:
        pivot = (pivot > 0).astype(int)
    support = (pivot > 0).sum(axis=0)
    total_events = pivot.shape[0]
    return MatrixBuildResult(pivot, support, total_events)

def compute_correlation(
    mat: pd.DataFrame,
    method: str = "pearson",
    *,
    sparse: bool = True,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """相関・類似度を計算し、閾値未満を0扱いでスパース化した行列を返す。"""

    if mat.empty:
        return pd.DataFrame()
    mat = mat.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    method = method.lower()
    if method == "jaccard":
        binary = (mat > 0).astype(int)
        data = binary.values.astype(float)
        dist = pairwise_distances(data.T, metric="jaccard")
        corr = 1 - dist
    elif method == "spearman":
        corr = mat.corr(method="spearman", min_periods=2).values
    else:
        corr = mat.corr(method="pearson", min_periods=2).values
    corr_df = pd.DataFrame(corr, index=mat.columns, columns=mat.columns)
    corr_df = corr_df.fillna(0.0)
    np.fill_diagonal(corr_df.values, 1.0)
    if threshold > 0:
        mask = np.abs(corr_df.values) < threshold
        np.fill_diagonal(mask, False)
        corr_df.values[mask] = 0.0
    return corr_df


def graph_from_corr(
    corr: pd.DataFrame,
    threshold: float,
    *,
    name_map: Optional[Dict[str, str]] = None,
) -> Graph:
    """相関行列から NetworkX グラフを構築する。"""

    G = nx.Graph()
    if corr.empty:
        return G
    cols = corr.columns.tolist()
    for col in cols:
        label = name_map.get(col, col) if name_map else col
        G.add_node(col, label=label)
    values = corr.values
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(values[i, j])
            if not np.isfinite(w):
                continue
            if abs(w) < threshold:
                continue
            G.add_edge(
                cols[i],
                cols[j],
                weight=w,
                abs_weight=abs(w),
            )
    return G


def _hierarchical_partition(G: Graph, min_size: int) -> Dict[str, int]:
    nodes = list(G.nodes())
    if len(nodes) <= 1:
        return {node: 0 for node in nodes}
    adj = nx.to_numpy_array(G, nodelist=nodes, weight="abs_weight", nonedge=0.0)
    dist = 1 - adj
    dist = np.clip(dist, 0.0, 1.0)
    if np.allclose(dist, 0.0):
        return {node: 0 for node in nodes}
    condensed = squareform(dist, checks=False)
    Z = hierarchy.linkage(condensed, method="average")
    max_clusters = max(1, len(nodes) // max(min_size, 1))
    labels = hierarchy.fcluster(Z, t=max_clusters, criterion="maxclust")
    return {node: int(labels[i]) for i, node in enumerate(nodes)}


def detect_communities(
    G: Graph,
    method: str = "louvain",
    min_size: int = 3,
    *,
    corr: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """グラフから商品クラスタを抽出し DataFrame で返す。"""

    nodes = list(G.nodes())
    if not nodes:
        return pd.DataFrame(columns=["product_id", "cluster_id", "cluster_size"])
    method = method.lower()
    if method == "louvain" and G.number_of_edges() > 0:
        if community_louvain is not None:
            try:
                partition = community_louvain.best_partition(
                    G, weight="abs_weight", random_state=42
                )
            except ValueError:
                partition = {node: idx for idx, node in enumerate(nodes)}
        else:
            communities = list(greedy_modularity_communities(G, weight="abs_weight"))
            partition = {}
            for idx, comm in enumerate(communities):
                for node in comm:
                    partition[node] = idx
            if not partition:
                partition = {node: idx for idx, node in enumerate(nodes)}
    elif method == "hierarchical":
        partition = _hierarchical_partition(G, min_size=min_size)
    else:
        partition = {node: idx for idx, node in enumerate(nodes)}

    df = pd.DataFrame({"product_id": nodes, "raw_cluster": [partition[n] for n in nodes]})
    df["cluster_size"] = df.groupby("raw_cluster")["product_id"].transform("size")
    valid_clusters = (
        df[df["cluster_size"] >= max(min_size, 1)]
        .groupby("raw_cluster")["product_id"]
        .count()
        .sort_values(ascending=False)
        .index.tolist()
    )
    cluster_id_map = {raw: f"C{idx + 1:02d}" for idx, raw in enumerate(valid_clusters)}
    df["cluster_id"] = df["raw_cluster"].map(cluster_id_map)
    df.loc[df["cluster_size"] < max(min_size, 1), "cluster_id"] = None

    communities_dict: Dict[int, set[str]] = {}
    for node, label in partition.items():
        communities_dict.setdefault(label, set()).add(node)
    communities_list = list(communities_dict.values())

    try:
        if community_louvain is not None and G.number_of_edges() > 0:
            modularity_score = community_louvain.modularity(
                partition, G, weight="abs_weight"
            )
        else:
            modularity_score = modularity(G, communities_list, weight="abs_weight")
    except Exception:
        modularity_score = np.nan
    df.attrs["partition"] = partition
    df.attrs["modularity"] = modularity_score
    df.attrs["cluster_count"] = len(valid_clusters)
    if corr is not None:
        df.attrs["corr_columns"] = corr.columns.tolist()
    return df.sort_values(["cluster_id", "cluster_size"], ascending=[True, False])

def compute_cluster_metrics(
    G: Graph,
    clusters_df: pd.DataFrame,
    *,
    support: Optional[pd.Series] = None,
    total_events: Optional[int] = None,
    corr: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """ノード単位の指標を計算しクラスタ情報と結合する。"""

    if clusters_df.empty:
        return pd.DataFrame(
            columns=[
                "product_id",
                "cluster_id",
                "cluster_size",
                "degree_centrality",
                "betweenness_centrality",
                "centrality_score",
                "cluster_avg_corr",
                "support_count",
                "support_rate",
                "lift",
                "is_representative",
            ]
        )

    df = clusters_df.copy()
    deg = nx.degree_centrality(G)
    if len(G) > 200:
        betw = nx.betweenness_centrality(
            G,
            weight="abs_weight",
            normalized=True,
            k=min(200, len(G)),
            seed=42,
        )
    else:
        betw = nx.betweenness_centrality(G, weight="abs_weight", normalized=True)
    df["degree_centrality"] = df["product_id"].map(deg).fillna(0.0)
    df["betweenness_centrality"] = df["product_id"].map(betw).fillna(0.0)
    df["centrality_score"] = 0.6 * df["degree_centrality"] + 0.4 * df["betweenness_centrality"]

    if corr is not None and not corr.empty:
        cluster_avg: Dict[str, float] = {}
        for cid, rows in df[df["cluster_id"].notna()].groupby("cluster_id"):
            nodes = rows["product_id"].tolist()
            sub = corr.loc[nodes, nodes]
            if sub.shape[0] <= 1:
                avg = 0.0
            else:
                vals = sub.values[np.triu_indices(sub.shape[0], k=1)]
                vals = vals[np.isfinite(vals)]
                vals = np.abs(vals)
                avg = float(vals.mean()) if vals.size else 0.0
            for node in nodes:
                cluster_avg[node] = avg
        df["cluster_avg_corr"] = df["product_id"].map(cluster_avg).fillna(0.0)
    else:
        df["cluster_avg_corr"] = 0.0

    if support is not None:
        df["support_count"] = df["product_id"].map(support).fillna(0.0)
        total = float(total_events) if total_events else float(max(df["support_count"].max(), 1))
        df["support_rate"] = df["support_count"] / total
        global_rate = df["support_rate"].mean()
        df["lift"] = df["support_rate"] / global_rate if global_rate > 0 else np.nan
    else:
        df["support_count"] = np.nan
        df["support_rate"] = np.nan
        df["lift"] = np.nan

    df["cluster_rank"] = (
        df[df["cluster_id"].notna()]
        .groupby("cluster_id")["centrality_score"]
        .rank("dense", ascending=False)
    )
    df["is_representative"] = df["cluster_rank"] == 1
    df.loc[df["cluster_id"].isna(), "is_representative"] = False

    df = df.sort_values(
        ["cluster_id", "is_representative", "centrality_score"],
        ascending=[True, False, False],
    )
    return df


def recommend_threshold(corr_values: np.ndarray) -> float:
    """相関値の上位25%点を推奨閾値として返す。"""

    if corr_values.size == 0:
        return 0.3
    vals = np.abs(corr_values[np.isfinite(corr_values)])
    vals = vals[vals < 0.999999]
    if vals.size == 0:
        return 0.3
    return float(np.quantile(vals, 0.75))


def _section_header(title: str, subtitle: str, icon: str = "") -> None:
    icon_html = f"<span class='mck-section-icon'>{icon}</span>" if icon else ""
    st.markdown(
        f"""
        <div class="mck-section-header">
            {icon_html}
            <div>
                <h2>{title}</h2>
                <p class="mck-section-subtitle">{subtitle}</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _format_product(product_id: str, name_map: Dict[str, str]) -> str:
    name = name_map.get(product_id)
    if name and name != product_id:
        return f"{product_id}｜{name}"
    return product_id

def render_correlation_category_module(
    *,
    plot_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Streamlit 用の併買クラスタリングモジュールを描画する。"""

    _section_header(
        "購買カテゴリ探索",
        "相関が高い商品群を自動クラスタリングし、クロスセルのタネを抽出。",
        icon="🧮",
    )
    st.caption(
        "フェルミ感覚：小売/ECの併買相関は Jaccard 0.2〜0.35 あたりから意味が出始めます。"
    )
    st.info(
        "初期推奨：Louvain＋閾値0.30＋クラスタ最小サイズ3。SKUが多い場合は出現上位で絞りましょう。"
    )
    with st.expander("初心者向けヒント", expanded=False):
        st.write("・相関が出ない場合は閾値を下げるか、対象商品数を増やしてください。")
        st.write("・二値化（購入有無）に切り替えると Jaccard が安定します。")
        st.write("・SKUが多い場合は出現上位に絞り、ネットワークは最大3万エッジ程度に抑えると軽快です。")

    side = st.sidebar.container()
    side.subheader("併買クラスタ設定")

    input_mode = side.radio(
        "入力形式",
        ("取引明細", "ユーザー×商品ピボット", "商品×商品相関行列"),
    )
    data_source = side.radio(
        "データソース",
        ("サンプルデータ", "ファイルアップロード"),
    )

    uploaded = None
    if data_source == "ファイルアップロード":
        uploaded = side.file_uploader(
            "CSV / Parquet / Excel",
            type=["csv", "tsv", "txt", "parquet", "pq", "xlsx", "xls"],
        )
        if uploaded is None:
            st.info("左サイドバーからファイルを選択してください。")
            return
        df_input = _read_table(uploaded)
    else:
        if input_mode == "取引明細":
            df_input = make_demo_transactions()
        elif input_mode == "ユーザー×商品ピボット":
            demo = make_demo_transactions()
            demo_build = build_matrix(demo, "transaction_id", "product_id", binary=True)
            df_input = demo_build.matrix.reset_index().rename(columns={"index": "transaction_id"})
        else:
            demo = make_demo_transactions()
            demo_build = build_matrix(demo, "transaction_id", "product_id", binary=True)
            df_input = compute_correlation(demo_build.matrix, method="jaccard", threshold=0.0)

    if df_input is None or df_input.empty:
        st.warning("データが読み込めませんでした。形式を確認してください。")
        return

    method_label = side.radio(
        "相関指標",
        ("Pearson", "Spearman", "Jaccard"),
    )
    method = method_label.lower()

    threshold = side.slider("相関閾値 t (|r|以上)", 0.0, 0.9, 0.3, 0.01)
    min_cluster_size = int(side.slider("最小クラスタサイズ k", 2, 20, 3))
    cluster_method = side.selectbox("コミュニティ検出", ("Louvain", "階層クラスタ"))
    sparse_on = side.checkbox("スパース化（閾値下を0に）", value=True)
    max_products = int(side.slider("最大商品数 (N)", 50, 3000, 500, 50))

    binary_key = "product_corr_binary"
    default_binary = st.session_state.get(binary_key, method == "jaccard")
    binary = side.checkbox(
        "購入有無で二値化", value=default_binary, key=binary_key, disabled=method == "jaccard"
    )
    if method == "jaccard":
        binary = True

    name_map: Dict[str, str] = {}
    build_result: Optional[MatrixBuildResult] = None
    corr_df: Optional[pd.DataFrame] = None
    warnings: List[str] = []
    logs: List[str] = []

    if input_mode == "取引明細":
        columns = df_input.columns.tolist()
        id_candidates = [c for c in columns if "user" in c.lower() or "trans" in c.lower()]
        if not id_candidates:
            id_candidates = columns
        id_col = side.selectbox("集計軸 (user_id / transaction_id)", id_candidates)
        prod_candidates = [c for c in columns if "product" in c.lower() or "sku" in c.lower()]
        if not prod_candidates:
            prod_candidates = [columns[0]]
        prod_col = side.selectbox("商品ID列", prod_candidates)
        numeric_cols = [c for c in columns if pd.api.types.is_numeric_dtype(df_input[c])]
        val_options = ["<二値>"] + numeric_cols
        val_col = side.selectbox("数量/金額列 (任意)", val_options, index=0)
        if val_col == "<二値>":
            val_col = None
        if "product_name" in df_input.columns:
            name_map = (
                df_input[[prod_col, "product_name"]]
                .dropna()
                .drop_duplicates(subset=[prod_col])
                .set_index(prod_col)["product_name"]
                .to_dict()
            )
        try:
            build_result = build_matrix(
                df_input,
                id_col=id_col,
                prod_col=prod_col,
                val_col=val_col,
                binary=binary,
            )
        except KeyError as exc:  # pragma: no cover (UI制御で保護)
            st.error(str(exc))
            return
    elif input_mode == "ユーザー×商品ピボット":
        columns = df_input.columns.tolist()
        id_col = side.selectbox("行ラベル列", columns)
        matrix = df_input.set_index(id_col)
        matrix.columns = matrix.columns.astype(str)
        matrix = matrix.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        if binary:
            matrix = (matrix > 0).astype(int)
        build_result = MatrixBuildResult(matrix, (matrix > 0).sum(axis=0), matrix.shape[0])
    else:
        corr_df = df_input.copy()
        if corr_df.columns[0] != corr_df.index.name and corr_df.columns[0] not in corr_df.columns[1:]:
            first_col = corr_df.columns[0]
            corr_df = corr_df.set_index(first_col)
        if corr_df.shape[0] != corr_df.shape[1]:
            st.error("相関行列は正方行列である必要があります。")
            return
        if corr_df.columns.tolist() != corr_df.index.tolist():
            corr_df.columns = corr_df.index
        corr_df = corr_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        np.fill_diagonal(corr_df.values, 1.0)
        logs.append("相関行列を直接読み込みました。")

    if build_result:
        matrix = build_result.matrix
        support = build_result.support_counts
        total_events = build_result.total_events
        if matrix.shape[1] == 0:
            st.warning("商品列が検出できませんでした。列指定を確認してください。")
            return
        if matrix.shape[1] > max_products:
            top_products = support.sort_values(ascending=False).head(max_products).index
            matrix = matrix[top_products]
            support = support[top_products]
            logs.append(f"商品数が多いため上位 {max_products} SKU に絞りました (残り {matrix.shape[1]}件)。")
        if matrix.shape[1] > 3000:
            warnings.append("商品数が3,000を超えています。対象を絞るかサンプリングしてください。")
        progress = st.progress(0, text="相関を計算中…")
        corr_df = compute_correlation(
            matrix,
            method=method,
            sparse=sparse_on,
            threshold=threshold if sparse_on else 0.0,
        )
        progress.progress(100, text="相関計算が完了しました。")
        if not sparse_on and threshold > 0:
            mask = np.abs(corr_df.values) < threshold
            np.fill_diagonal(mask, False)
            corr_df.values[mask] = 0.0
    else:
        support = None
        total_events = None

    if corr_df is None or corr_df.empty:
        st.info("相関計算対象の商品が不足しています。条件を見直してください。")
        return

    name_map = {k: v for k, v in name_map.items() if k in corr_df.columns}

    G = graph_from_corr(corr_df, threshold=max(threshold, 0.0 if not sparse_on else threshold), name_map=name_map)
    if G.number_of_edges() == 0:
        st.warning("閾値条件を満たすエッジがありません。閾値を下げて再実行してください。")
        rec = recommend_threshold(corr_df.values[np.triu_indices_from(corr_df, k=1)])
        st.caption(f"推奨閾値（上位25%点）: 約 {rec:.2f}")
        return

    clusters_df = detect_communities(
        G,
        method="louvain" if cluster_method.lower().startswith("louvain") else "hierarchical",
        min_size=min_cluster_size,
        corr=corr_df,
    )
    metrics_df = compute_cluster_metrics(
        G,
        clusters_df,
        support=support,
        total_events=total_events,
        corr=corr_df,
    )

    if name_map:
        metrics_df["product_name"] = metrics_df["product_id"].map(name_map)
    else:
        metrics_df["product_name"] = metrics_df["product_id"]
    assigned = metrics_df[metrics_df["cluster_id"].notna()]
    total_products = max(len(metrics_df["product_id"].unique()), 1)
    coverage = len(assigned["product_id"].unique()) / total_products
    cluster_count = int(clusters_df.attrs.get("cluster_count", assigned["cluster_id"].nunique()))
    avg_size = float(
        assigned.groupby("cluster_id")["product_id"].nunique().mean()
    ) if cluster_count else 0.0
    avg_corr = float(assigned["cluster_avg_corr"].mean()) if not assigned.empty else 0.0
    modularity = float(clusters_df.attrs.get("modularity", np.nan))

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("クラスタ数", f"{cluster_count}")
    c2.metric("平均クラスタサイズ", f"{avg_size:.1f}")
    c3.metric("カバー率", f"{coverage * 100:.1f}%")
    c4.metric("クラスタ内平均相関", f"{avg_corr:.2f}")
    c5.metric("モジュラリティ", "{:.2f}".format(modularity) if not np.isnan(modularity) else "—")

    st.markdown("---")
    st.subheader("相関ヒートマップ")
    limit_heatmap = st.checkbox("ヒートマップをクラスタ代表商品に限定", value=True)
    if limit_heatmap and not assigned.empty:
        top_nodes = (
            assigned.sort_values(["cluster_id", "centrality_score"], ascending=[True, False])
            .groupby("cluster_id")
            .head(5)["product_id"].tolist()
        )
        if top_nodes:
            sub_corr = corr_df.loc[top_nodes, top_nodes]
        else:
            sub_corr = corr_df
    else:
        sub_corr = corr_df
    display_corr = sub_corr.copy()
    display_corr.index = [
        _format_product(pid, name_map) if name_map else pid for pid in display_corr.index
    ]
    display_corr.columns = [
        _format_product(pid, name_map) if name_map else pid for pid in display_corr.columns
    ]
    heat_fig = px.imshow(
        display_corr,
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    heat_fig.update_layout(height=500)
    heat_fig = apply_elegant_theme(heat_fig, theme=st.session_state.get("ui_theme", "light"))
    render_plotly_with_spinner(heat_fig, config=plot_config or {})

    st.subheader("ネットワークグラフ")
    max_edges = 30000
    G_draw = G.copy()
    if G_draw.number_of_edges() > max_edges:
        top_edges = sorted(
            G_draw.edges(data=True),
            key=lambda x: x[2].get("abs_weight", 0),
            reverse=True,
        )[:max_edges]
        G_draw = nx.Graph()
        for node, data in G.nodes(data=True):
            G_draw.add_node(node, **data)
        G_draw.add_edges_from([(u, v, d) for u, v, d in top_edges])
        warnings.append(f"エッジ数が多いため上位 {max_edges:,} 本に制限しました。")
    max_nodes = 250
    if G_draw.number_of_nodes() > max_nodes:
        keep_nodes = (
            metrics_df.sort_values("centrality_score", ascending=False)
            .head(max_nodes)["product_id"].tolist()
        )
        G_draw = G_draw.subgraph(keep_nodes).copy()
        warnings.append(f"ノード数が多いため中心性上位 {max_nodes} 件に限定して描画しました。")

    pos = nx.spring_layout(G_draw, weight="abs_weight", seed=42, k=None)
    edge_x: List[float] = []
    edge_y: List[float] = []
    for u, v, data in G_draw.edges(data=True):
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    node_x = [pos[node][0] for node in G_draw.nodes()]
    node_y = [pos[node][1] for node in G_draw.nodes()]
    metrics_idx = metrics_df.set_index("product_id")

    cluster_colors = px.colors.qualitative.G10 + px.colors.qualitative.Safe + px.colors.qualitative.Bold
    color_map = {}
    for idx, cid in enumerate(sorted(assigned["cluster_id"].dropna().unique())):
        color_map[cid] = cluster_colors[idx % len(cluster_colors)]

    node_colors = []
    node_sizes = []
    node_text: List[str] = []
    for node in G_draw.nodes():
        if node in metrics_idx.index:
            row = metrics_idx.loc[node]
            cluster_label = row.get("cluster_id")
            if pd.isna(cluster_label) or not cluster_label:
                cluster_label = "未割当"
            support_txt = (
                f"<br>支持度: {row['support_rate']:.2%}"
                if not pd.isna(row.get("support_rate"))
                else ""
            )
            node_colors.append(color_map.get(row.get("cluster_id"), "#9ca3af"))
            node_sizes.append(18 + 120 * row.get("centrality_score", 0.0))
            node_text.append(
                f"{_format_product(node, name_map)}"
                f"<br>クラスタ: {cluster_label}"
                f"<br>中心性: {row.get('centrality_score', 0.0):.2f}"
                f"<br>次数: {row.get('degree_centrality', 0.0):.2f}"
                f"<br>媒介: {row.get('betweenness_centrality', 0.0):.2f}"
                f"{support_txt}"
            )
        else:
            node_colors.append("#9ca3af")
            node_sizes.append(12)
            node_text.append(_format_product(node, name_map))

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="rgba(100,100,100,0.4)"),
        hoverinfo="none",
        mode="lines",
    )
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(size=node_sizes, color=node_colors, line=dict(width=1, color="#1f2937")),
        hoverinfo="text",
        text=node_text,
    )
    net_fig = go.Figure(data=[edge_trace, node_trace])
    net_fig.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        height=620,
    )
    net_fig = apply_elegant_theme(net_fig, theme=st.session_state.get("ui_theme", "light"))
    render_plotly_with_spinner(net_fig, config=plot_config or {})

    st.subheader("クラスタ一覧")
    table_cols = [
        "cluster_id",
        "product_id",
        "product_name",
        "centrality_score",
        "degree_centrality",
        "betweenness_centrality",
        "cluster_avg_corr",
        "support_rate",
        "lift",
        "is_representative",
    ]
    table = metrics_df[table_cols].copy()
    table = table.rename(
        columns={
            "cluster_id": "クラスタ",
            "product_id": "商品ID",
            "product_name": "商品名",
            "centrality_score": "中心性スコア",
            "degree_centrality": "次数中心性",
            "betweenness_centrality": "媒介中心性",
            "cluster_avg_corr": "クラスタ平均相関",
            "support_rate": "支持度",
            "lift": "リフト",
            "is_representative": "代表",
        }
    )
    table["代表"] = table["代表"].map({True: "★", False: ""})
    st.dataframe(table, use_container_width=True)

    st.subheader("クラスタ別バー表示")
    if not assigned.empty:
        cluster_summary = (
            assigned.groupby("cluster_id")
            .agg(
                商品数=("product_id", "nunique"),
                平均相関=("cluster_avg_corr", "mean"),
                平均支持度=("support_rate", "mean"),
            )
            .reset_index()
        )
        bar_fig = px.bar(
            cluster_summary,
            x="cluster_id",
            y="商品数",
            color="平均相関",
            text="商品数",
            hover_data={"平均相関": ":.2f", "平均支持度": ":.2%"},
            color_continuous_scale="Blues",
        )
        bar_fig.update_layout(height=420, xaxis_title="クラスタ", yaxis_title="商品数")
        bar_fig = apply_elegant_theme(bar_fig, theme=st.session_state.get("ui_theme", "light"))
        render_plotly_with_spinner(bar_fig, config=plot_config or {})
    else:
        st.info("最小サイズ条件を満たすクラスタがありません。閾値や最小サイズを見直してください。")

    st.subheader("CSVエクスポート")
    corr_csv = corr_df.to_csv(index=True, encoding="utf-8-sig")
    edge_list = [
        {"source": u, "target": v, "weight": data.get("weight", 0.0)}
        for u, v, data in G.edges(data=True)
        if abs(data.get("weight", 0.0)) >= threshold
    ]
    edges_df = pd.DataFrame(edge_list)
    clusters_export = metrics_df.copy()
    st.download_button(
        "相関行列をCSV保存",
        data=corr_csv,
        file_name="corr_matrix.csv",
        mime="text/csv",
    )
    st.download_button(
        "エッジリストをCSV保存",
        data=edges_df.to_csv(index=False, encoding="utf-8-sig"),
        file_name="edges.csv",
        mime="text/csv",
    )
    st.download_button(
        "クラスタ割当をCSV保存",
        data=clusters_export.to_csv(index=False, encoding="utf-8-sig"),
        file_name="clusters.csv",
        mime="text/csv",
    )

    st.subheader("ログ・推奨値")
    for msg in logs:
        st.caption(f"LOG: {msg}")
    for warn in warnings:
        st.warning(warn)

