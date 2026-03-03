from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import pairwise_distances

from .baseline_blast_like import blast_like_search
from .baseline_hmm_like import hmm_like_search
from .scoring import combine_scores, local_support_score, novelty_from_identity, positive_affinity


def mahalanobis_score(train_emb: np.ndarray, cand_emb: np.ndarray) -> np.ndarray:
    cov = EmpiricalCovariance().fit(train_emb)
    d = cov.mahalanobis(cand_emb)
    return 1 / (1 + d)


def run_retrieval(
    positives_train: pd.DataFrame,
    positives_all: pd.DataFrame,
    unlabeled: pd.DataFrame,
    emb_pos_train: np.ndarray,
    emb_unl: np.ndarray,
    prototypes: pd.DataFrame,
    weights: dict,
    local_neighbors: int = 15,
) -> pd.DataFrame:
    blast = blast_like_search(positives_train, unlabeled)
    hmm = hmm_like_search(positives_train, unlabeled)
    proto_mat = np.vstack(prototypes["prototype"].values)
    aff, nearest_idx = positive_affinity(emb_unl, proto_mat)
    dist = pairwise_distances(emb_unl, proto_mat).min(axis=1)
    local = local_support_score(emb_unl, n_neighbors=local_neighbors)
    maha = mahalanobis_score(emb_pos_train, emb_unl)

    merged = unlabeled[["id", "sequence"]].rename(columns={"id": "candidate_id"}).merge(blast, on="candidate_id").merge(hmm, on="candidate_id")
    merged["positive_affinity"] = aff
    merged["embedding_distance"] = dist
    merged["local_support_score"] = local
    merged["mahalanobis_score"] = maha
    merged["novelty_score"] = novelty_from_identity(merged["approx_identity"].to_numpy())
    merged["easy_hit_penalty"] = ((merged["blast_like_score"] > weights.get("easy_hit_threshold", 0.72)) | (merged["hmm_like_score"] > weights.get("easy_hit_threshold", 0.72))).astype(float)
    merged["nearest_positive_cluster"] = [int(prototypes.iloc[i]["cluster"]) for i in nearest_idx]
    merged["nearest_positive_id"] = merged["best_hit_positive_id"]
    merged["sequence_length"] = merged["sequence"].str.len()

    ranked = combine_scores(
        merged,
        w_affinity=weights["weights"]["positive_affinity"],
        w_novelty=weights["weights"]["novelty"],
        w_local=weights["weights"]["local_support"],
        w_penalty=weights["weights"]["easy_hit_penalty"],
    )
    cols = [
        "candidate_id",
        "sequence_length",
        "nearest_positive_cluster",
        "nearest_positive_id",
        "best_hit_positive_id",
        "approx_identity",
        "approx_coverage",
        "blast_like_score",
        "best_hmm_cluster",
        "hmm_like_score",
        "embedding_distance",
        "positive_affinity",
        "novelty_score",
        "local_support_score",
        "easy_hit_penalty",
        "mahalanobis_score",
        "final_score",
        "rank",
    ]
    return ranked[cols]
