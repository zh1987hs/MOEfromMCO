from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


@dataclass
class FeatureBuildResult:
    """Container for candidate feature table and helper diagnostics."""

    features: pd.DataFrame


def _safe_loge(v: float, floor: float = 1e-300) -> float:
    return float(-np.log10(max(v, floor)))


def build_easy_rank_table(
    candidate_ids: list[str],
    mm_best: pd.DataFrame,
    hmm_best: pd.DataFrame,
    mm_flags: pd.DataFrame,
    hmm_flags: pd.DataFrame,
) -> pd.DataFrame:
    """Rank easy hits using combined MMseqs/HMM strength."""
    base = pd.DataFrame({"candidate_id": candidate_ids})
    base = base.merge(mm_flags, on="candidate_id", how="left").merge(hmm_flags, on="candidate_id", how="left")
    base["mmseqs_easy_hit_flag"] = base["mmseqs_easy_hit_flag"].fillna(False).astype(bool)
    base["hmm_easy_hit_flag"] = base["hmm_easy_hit_flag"].fillna(False).astype(bool)
    base["easy"] = base["mmseqs_easy_hit_flag"] | base["hmm_easy_hit_flag"]

    if not mm_best.empty:
        mm1 = (
            mm_best.sort_values(["query_id", "bits"], ascending=[True, False])
            .drop_duplicates("query_id")
            .rename(
                columns={
                    "query_id": "candidate_id",
                    "target_id": "mmseqs_best_target",
                    "fident": "mmseqs_best_fident",
                    "qcov": "mmseqs_best_qcov",
                    "tcov": "mmseqs_best_tcov",
                    "evalue": "mmseqs_best_evalue",
                    "bits": "mmseqs_best_bits",
                }
            )
        )
        base = base.merge(
            mm1[
                [
                    "candidate_id",
                    "mmseqs_best_target",
                    "mmseqs_best_fident",
                    "mmseqs_best_qcov",
                    "mmseqs_best_tcov",
                    "mmseqs_best_evalue",
                    "mmseqs_best_bits",
                ]
            ],
            on="candidate_id",
            how="left",
        )

    if not hmm_best.empty:
        h1 = hmm_best.rename(columns={"candidate_id": "candidate_id"})
        base = base.merge(h1, on="candidate_id", how="left")

    base["mm_strength"] = base.get("mmseqs_best_bits", pd.Series(dtype=float)).fillna(0.0) + base.get(
        "mmseqs_best_evalue", pd.Series(dtype=float)
    ).fillna(1.0).map(_safe_loge)
    base["hmm_strength"] = base.get("hmm_best_bitscore", pd.Series(dtype=float)).fillna(0.0) + base.get(
        "hmm_best_evalue", pd.Series(dtype=float)
    ).fillna(1.0).map(_safe_loge)
    base["easy_score"] = 0.6 * base["mm_strength"] + 0.4 * base["hmm_strength"]

    easy_df = base[base["easy"]].copy().sort_values("easy_score", ascending=False).reset_index(drop=True)
    easy_df["easy_or_missed"] = "easy"
    easy_df["final_score"] = easy_df["easy_score"]
    easy_df["scoring_mode"] = "easy_strength"
    easy_df["confidence_tier"] = "high"
    easy_df["dominant_signal_type"] = "mmseqs/hmm"
    easy_df["reason_for_high_rank"] = "strong_easy_hit"
    easy_df["flags"] = ""
    return easy_df


def build_missed_candidate_features(
    missed_ids: list[str],
    unlabeled_ids: list[str],
    unlabeled_emb: np.ndarray,
    positive_ids: list[str],
    positive_emb: np.ndarray,
    positive_labels: np.ndarray,
    lengths: dict[str, int],
    medoids_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None,
    mm_best: pd.DataFrame,
    hmm_best: pd.DataFrame,
    cfg: dict[str, Any],
) -> FeatureBuildResult:
    """Build experiment-oriented candidate-level features for missed candidates."""
    if len(missed_ids) == 0:
        return FeatureBuildResult(features=pd.DataFrame())

    id_to_idx_u = {sid: i for i, sid in enumerate(unlabeled_ids)}
    valid_ids = [x for x in missed_ids if x in id_to_idx_u]
    miss_idx = [id_to_idx_u[x] for x in valid_ids]
    miss_emb = unlabeled_emb[miss_idx]

    # Positive cluster prototypes from current fold/run embedding.
    p_df = pd.DataFrame({"positive_id": positive_ids, "cluster": positive_labels})
    proto_ids = sorted(p_df["cluster"].unique().tolist())
    prototypes = []
    cluster_member_idx: dict[int, np.ndarray] = {}
    for c in proto_ids:
        idx = np.where(positive_labels == c)[0]
        cluster_member_idx[int(c)] = idx
        prototypes.append(positive_emb[idx].mean(axis=0))
    proto_mat = np.vstack(prototypes)

    affinity_metric = cfg.get("affinity_metric", cfg.get("distance_metric", "cosine"))
    dmat = pairwise_distances(miss_emb, proto_mat, metric=affinity_metric)
    nearest_proto_idx = np.argmin(dmat, axis=1)
    nearest_cluster = [int(proto_ids[i]) for i in nearest_proto_idx]
    emb_dist = dmat[np.arange(len(valid_ids)), nearest_proto_idx]
    emb_sim = np.exp(-emb_dist)

    # Positive support: combine top-k in-cluster similarity + medoid + prototype support.
    support_topk = int(cfg.get("support_topk", 5))
    id_to_pos_idx = {pid: i for i, pid in enumerate(positive_ids)}

    pos_support = []
    for i, cid in enumerate(nearest_cluster):
        idx = cluster_member_idx[int(cid)]
        cand = miss_emb[i : i + 1]

        sims = 1.0 - pairwise_distances(cand, positive_emb[idx], metric="cosine").reshape(-1)
        sims = np.sort(sims)[::-1]
        k = min(len(sims), max(1, support_topk))
        topk_support = float(np.mean(sims[:k]))

        proto_support = float(np.clip(1.0 - emb_dist[i], 0.0, 1.0))

        medoid_support = topk_support
        mid = medoid_map.get(int(cid))
        if mid in id_to_pos_idx:
            m_i = id_to_pos_idx[mid]
            medoid_support = float(
                np.clip(
                    1.0 - pairwise_distances(cand, positive_emb[m_i : m_i + 1], metric="cosine")[0, 0],
                    0.0,
                    1.0,
                )
            )

        comb = 0.60 * np.clip(topk_support, 0.0, 1.0) + 0.25 * medoid_support + 0.15 * proto_support
        pos_support.append(float(np.clip(comb, 0.0, 1.0)))

    # Local density in unlabeled space (auxiliary signal only).
    k_den = int(cfg.get("density_knn_k", cfg.get("knn_k", 20)))
    nn = NearestNeighbors(n_neighbors=min(k_den + 1, len(miss_emb)), metric="cosine")
    nn.fit(miss_emb)
    dist, _ = nn.kneighbors(miss_emb)
    mean_neighbor_dist = dist[:, 1:].mean(axis=1) if dist.shape[1] > 1 else np.ones(len(miss_emb))
    local_density = np.exp(-mean_neighbor_dist)

    family_map = {}
    if metadata_df is not None and {"positive_id", "gold_family"}.issubset(set(metadata_df.columns)):
        family_map = dict(zip(metadata_df["positive_id"], metadata_df["gold_family"]))

    feats = pd.DataFrame(
        {
            "candidate_id": valid_ids,
            "length": [lengths.get(x, -1) for x in valid_ids],
            "nearest_positive_cluster": nearest_cluster,
            "nearest_positive_id": [medoid_map.get(c, "NA") for c in nearest_cluster],
            "embedding_distance": emb_dist,
            "embedding_similarity": emb_sim,
            "positive_support_score": pos_support,
            "local_density_score": local_density,
            "easy_or_missed": "missed",
        }
    )
    feats["nearest_positive_family"] = feats["nearest_positive_id"].map(lambda x: family_map.get(x, "NA"))

    # MMseqs/HMM joins.
    mm_map = pd.DataFrame()
    if not mm_best.empty:
        mm_map = (
            mm_best.sort_values(["query_id", "bits"], ascending=[True, False])
            .drop_duplicates("query_id")
            .rename(
                columns={
                    "query_id": "candidate_id",
                    "target_id": "mmseqs_best_target",
                    "fident": "mmseqs_best_fident",
                    "qcov": "mmseqs_best_qcov",
                    "tcov": "mmseqs_best_tcov",
                    "evalue": "mmseqs_best_evalue",
                    "bits": "mmseqs_best_bits",
                }
            )
        )
        feats = feats.merge(
            mm_map[
                [
                    "candidate_id",
                    "mmseqs_best_target",
                    "mmseqs_best_fident",
                    "mmseqs_best_qcov",
                    "mmseqs_best_tcov",
                    "mmseqs_best_evalue",
                    "mmseqs_best_bits",
                ]
            ],
            on="candidate_id",
            how="left",
        )

    if not hmm_best.empty:
        hmm = hmm_best.rename(
            columns={
                "candidate_id": "candidate_id",
                "hmm_best_cluster": "hmm_best_cluster",
                "hmm_best_evalue": "hmm_best_evalue",
                "hmm_best_bitscore": "hmm_best_bitscore",
            }
        )
        if "hmm_best_target" not in hmm.columns:
            hmm["hmm_best_target"] = np.nan
        feats = feats.merge(
            hmm[["candidate_id", "hmm_best_cluster", "hmm_best_target", "hmm_best_evalue", "hmm_best_bitscore"]],
            on="candidate_id",
            how="left",
        )

    # Novelty as calibrated small correction: high when not too-close and still supported.
    mm_close = (
        0.5 * feats.get("mmseqs_best_fident", pd.Series(dtype=float)).fillna(0.0)
        + 0.2 * feats.get("mmseqs_best_qcov", pd.Series(dtype=float)).fillna(0.0)
        + 0.2 * feats.get("mmseqs_best_tcov", pd.Series(dtype=float)).fillna(0.0)
        + 0.1 * (feats.get("mmseqs_best_evalue", pd.Series(dtype=float)).fillna(1.0).map(_safe_loge) / 200.0)
    )
    hmm_close = (
        feats.get("hmm_best_evalue", pd.Series(dtype=float)).fillna(1.0).map(_safe_loge) / 200.0
        + feats.get("hmm_best_bitscore", pd.Series(dtype=float)).fillna(0.0) / 500.0
    )
    close_score = np.clip(0.5 * mm_close + 0.5 * hmm_close, 0.0, 1.0)
    novelty = np.clip((1.0 - close_score) * (0.4 + 0.6 * feats["positive_support_score"]), 0.0, cfg.get("novelty_cap", 0.9))
    feats["novelty_score"] = novelty

    # Risk/diagnostic flags.
    flags = []
    reasons = []
    dominant = []
    for _, r in feats.iterrows():
        f = []
        if r["positive_support_score"] < 0.25 and r["local_density_score"] > 0.75:
            f.append("generic_mco_risk")
        if r["length"] < 300 or r["length"] > 1800:
            f.append("length_outlier")
        if pd.notna(r.get("mmseqs_best_fident")) and float(r.get("mmseqs_best_fident")) > 0.45:
            f.append("too_close_to_easy_hit")

        sig = {
            "embedding": float(r["embedding_similarity"]),
            "support": float(r["positive_support_score"]),
            "novelty": float(r["novelty_score"]),
            "density": float(r["local_density_score"]),
        }
        dom = max(sig, key=sig.get)
        dominant.append(dom)
        reasons.append(f"dominant={dom};support={r['positive_support_score']:.3f};aff={r['embedding_similarity']:.3f};nov={r['novelty_score']:.3f}")
        flags.append(";".join(f) if f else "")

    feats["dominant_signal_type"] = dominant
    feats["flags"] = flags
    feats["reason_for_high_rank"] = reasons

    # explicit risk diagnostics for experimental triage
    generic_score = np.clip(0.7 * feats["local_density_score"] + 0.3 * (1.0 - feats["positive_support_score"]), 0.0, 1.0)
    fp_risk = np.clip(0.5 * generic_score + 0.5 * (1.0 - feats["embedding_similarity"]), 0.0, 1.0)
    feats["generic_mco_risk_score"] = generic_score
    feats["false_positive_risk"] = fp_risk

    return FeatureBuildResult(features=feats)
