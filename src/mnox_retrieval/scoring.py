from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def positive_affinity(candidate_emb: np.ndarray, protos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dist = pairwise_distances(candidate_emb, protos)
    min_dist = dist.min(axis=1)
    nearest = dist.argmin(axis=1)
    affinity = 1 / (1 + min_dist)
    return affinity, nearest


def novelty_from_identity(identity: np.ndarray) -> np.ndarray:
    return 1 - identity


def local_support_score(candidate_emb: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    k = min(n_neighbors, len(candidate_emb))
    nn = NearestNeighbors(n_neighbors=max(2, k)).fit(candidate_emb)
    d, _ = nn.kneighbors(candidate_emb)
    mean_d = d[:, 1:].mean(axis=1)
    return 1 / (1 + mean_d)


def combine_scores(
    df: pd.DataFrame,
    w_affinity: float,
    w_novelty: float,
    w_local: float,
    w_penalty: float,
) -> pd.DataFrame:
    out = df.copy()
    out["final_score"] = (
        w_affinity * out["positive_affinity"]
        + w_novelty * out["novelty_score"]
        + w_local * out["local_support_score"]
        - w_penalty * out["easy_hit_penalty"]
    )
    out = out.sort_values("final_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out
