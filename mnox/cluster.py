from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score


@dataclass
class ClusterOutput:
    """Container for positive cluster outputs."""

    labels: np.ndarray
    method: str
    k: int
    score: float


def pick_best_clustering(
    embeddings: np.ndarray,
    methods: list[str],
    k_min: int,
    k_max: int,
    random_state: int,
) -> ClusterOutput:
    """Grid search over method and k with silhouette score."""
    best: ClusterOutput | None = None
    for method in methods:
        for k in range(k_min, k_max + 1):
            if method == "agglomerative":
                model = AgglomerativeClustering(n_clusters=k)
                labels = model.fit_predict(embeddings)
            elif method == "kmeans":
                model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
                labels = model.fit_predict(embeddings)
            else:
                raise ValueError(f"Unsupported method: {method}")

            if len(set(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels, metric="cosine")
            if best is None or score > best.score:
                best = ClusterOutput(labels=labels, method=method, k=k, score=score)

    if best is None:
        raise RuntimeError("Unable to select clustering model.")
    return best


def compute_prototypes_and_medoids(
    ids: list[str], embeddings: np.ndarray, labels: np.ndarray
) -> tuple[pd.DataFrame, dict[int, np.ndarray], pd.DataFrame]:
    """Compute cluster assignments, prototypes, and medoids."""
    df = pd.DataFrame({"id": ids, "cluster": labels})
    prototypes: dict[int, np.ndarray] = {}
    medoid_rows: list[dict] = []

    for cluster_id in sorted(df["cluster"].unique()):
        idx = np.where(labels == cluster_id)[0]
        cluster_emb = embeddings[idx]
        proto = cluster_emb.mean(axis=0)
        prototypes[int(cluster_id)] = proto

        dists = np.linalg.norm(cluster_emb - proto[None, :], axis=1)
        medoid_local = int(np.argmin(dists))
        medoid_global_idx = idx[medoid_local]
        medoid_rows.append(
            {
                "cluster": int(cluster_id),
                "medoid_id": ids[medoid_global_idx],
                "distance_to_prototype": float(dists[medoid_local]),
            }
        )

    return df, prototypes, pd.DataFrame(medoid_rows)
