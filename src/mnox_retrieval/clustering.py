from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

from .baseline_blast_like import approx_seq_identity


def cluster_positives_embedding(emb: np.ndarray, n_clusters: int = 5, seed: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return km.fit_predict(emb)


def cluster_positives_sequence(df: pd.DataFrame, n_clusters: int = 5, seed: int = 42) -> np.ndarray:
    n = len(df)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i, n):
            ident, _ = approx_seq_identity(df.iloc[i]["sequence"], df.iloc[j]["sequence"])
            sim[i, j] = sim[j, i] = ident
    dist = 1 - sim
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    return km.fit_predict(dist)


def build_prototypes(df: pd.DataFrame, emb: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for c in sorted(set(labels.tolist())):
        idx = np.where(labels == c)[0]
        sub = emb[idx]
        center = sub.mean(axis=0)
        d = pairwise_distances(sub, center.reshape(1, -1)).flatten()
        medoid_local = idx[int(np.argmin(d))]
        rows.append(
            {
                "cluster": int(c),
                "members": df.iloc[idx]["id"].tolist(),
                "prototype": center,
                "medoid_id": df.iloc[medoid_local]["id"],
                "size": int(len(idx)),
                "intra_mean_dist": float(d.mean()),
            }
        )
    return pd.DataFrame(rows)
