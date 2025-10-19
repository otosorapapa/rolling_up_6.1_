from __future__ import annotations

import numpy as np
import pandas as pd

import pytest

from core.product_clusters import (
    build_matrix,
    compute_correlation,
    compute_cluster_metrics,
    detect_communities,
    graph_from_corr,
    make_demo_transactions,
    recommend_threshold,
)


def test_demo_transactions_clustered() -> None:
    df = make_demo_transactions(n_transactions=300, seed=10)
    result = build_matrix(df, id_col="transaction_id", prod_col="product_id", binary=True)
    assert not result.matrix.empty
    corr = compute_correlation(result.matrix, method="jaccard", threshold=0.0)
    assert corr.loc["A", "A"] == pytest.approx(1.0)
    G = graph_from_corr(corr, threshold=0.2)
    assert G.number_of_nodes() == corr.shape[0]
    clusters = detect_communities(G, method="louvain", min_size=2, corr=corr)
    metrics = compute_cluster_metrics(
        G,
        clusters,
        support=result.support_counts,
        total_events=result.total_events,
        corr=corr,
    )
    cluster_map = metrics.set_index("product_id")["cluster_id"].to_dict()
    cluster_ids = {cluster_map.get(pid) for pid in ["A", "B", "C"]}
    cluster_ids.discard(None)
    assert len(cluster_ids) == 1, "A/B/C should share a detected cluster"
    # Ensure support rates are bounded between 0 and 1
    support_rates = metrics["support_rate"].dropna()
    assert ((support_rates >= 0) & (support_rates <= 1)).all()


@pytest.fixture(autouse=True)
def _set_display_precision() -> None:
    pd.set_option("display.precision", 4)


def test_recommend_threshold() -> None:
    values = np.array([0.1, 0.2, 0.5, 0.8, 0.9])
    thr = recommend_threshold(values)
    assert 0.5 <= thr <= 0.9
