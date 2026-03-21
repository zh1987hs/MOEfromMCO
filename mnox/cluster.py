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
    logger=None,
) -> ClusterOutput:
    """Grid search over method and k with silhouette score and small-sample guards."""
    n = embeddings.shape[0]
    if n < 20 and logger is not None:
        logger.warning("Positive count is small (n=%d). Auto clustering may be unstable.", n)

    # Need at least 2 clusters and at most n-1 clusters for silhouette.
    k_min_eff = max(2, k_min)
    k_max_eff = min(k_max, n - 1)
    if k_max_eff < k_min_eff:
        raise RuntimeError(
            f"Invalid auto clustering range for n={n}: requested [{k_min}, {k_max}] -> effective [{k_min_eff}, {k_max_eff}]"
        )
    if (k_max_eff != k_max or k_min_eff != k_min) and logger is not None:
        logger.warning(
            "Adjusted clustering k-range from [%d, %d] to [%d, %d] due to sample size n=%d.",
            k_min,
            k_max,
            k_min_eff,
            k_max_eff,
            n,
        )

    best: ClusterOutput | None = None
    for method in methods:
        for k in range(k_min_eff, k_max_eff + 1):
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

    if logger is not None:
        counts = pd.Series(best.labels).value_counts()
        singleton = int((counts == 1).sum())
        if singleton > 0:
            logger.warning("Auto clustering produced %d singleton clusters.", singleton)
        if (counts < 3).sum() > 0:
            logger.warning("Auto clustering produced %d very small clusters (<3 members).", int((counts < 3).sum()))

    return best


def load_fixed_clusters(fixed_cluster_csv: str, positive_ids: list[str], metadata_df: pd.DataFrame) -> np.ndarray:
    """Load fixed clusters and validate IDs against positives and metadata."""
    fixed_df = pd.read_csv(fixed_cluster_csv)
    required = {"positive_id", "cluster"}
    if not required.issubset(set(fixed_df.columns)):
        raise RuntimeError(f"Fixed cluster CSV missing columns: {required - set(fixed_df.columns)}")

    pos_set = set(positive_ids)
    meta_set = set(metadata_df["positive_id"].tolist())
    fixed_set = set(fixed_df["positive_id"].tolist())

    missing_in_fasta = fixed_set - pos_set
    if missing_in_fasta:
        raise RuntimeError(f"IDs in fixed clusters but not in positives FASTA: {list(sorted(missing_in_fasta))[:10]}")

    missing_in_meta = fixed_set - meta_set
    if missing_in_meta:
        raise RuntimeError(f"IDs in fixed clusters but not in metadata CSV: {list(sorted(missing_in_meta))[:10]}")

    missing_in_fixed = pos_set - fixed_set
    if missing_in_fixed:
        raise RuntimeError(f"IDs in positives FASTA but not in fixed clusters CSV: {list(sorted(missing_in_fixed))[:10]}")

    mapper = dict(zip(fixed_df["positive_id"], fixed_df["cluster"]))
    return np.array([int(mapper[x]) for x in positive_ids], dtype=int)


def compute_prototypes_and_medoids(
    ids: list[str],
    embeddings: np.ndarray,
    labels: np.ndarray,
    metadata_df: pd.DataFrame | None = None,
    use_tier_weights: bool = False,
    gold_weight: float = 1.0,
    silver_weight: float = 1.0,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], pd.DataFrame]:
    """Compute cluster assignments, prototypes, and medoids."""
    df = pd.DataFrame({"id": ids, "cluster": labels})
    prototypes: dict[int, np.ndarray] = {}
    medoid_rows: list[dict] = []

    id_to_tier = None
    if metadata_df is not None and "tier" in metadata_df.columns:
        id_to_tier = dict(zip(metadata_df["positive_id"], metadata_df["tier"]))

    for cluster_id in sorted(df["cluster"].unique()):
        idx = np.where(labels == cluster_id)[0]
        cluster_emb = embeddings[idx]

        if use_tier_weights and id_to_tier is not None:
            w = []
            for i in idx:
                tier = str(id_to_tier.get(ids[i], "silver")).lower()
                w.append(gold_weight if tier == "gold" else silver_weight)
            w_arr = np.array(w, dtype=float)
            proto = np.average(cluster_emb, axis=0, weights=w_arr)
        else:
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
