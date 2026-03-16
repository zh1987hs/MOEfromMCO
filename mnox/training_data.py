from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .features import build_missed_candidate_features


@dataclass
class TrainingDataResult:
    """Container for learned scorer training data."""

    features: pd.DataFrame
    labels: np.ndarray
    n_pos: int
    n_neg: int


def build_learned_training_data(
    *,
    positive_ids: list[str],
    positive_emb: np.ndarray,
    positive_labels: np.ndarray,
    medoids_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None,
    mm_best: pd.DataFrame,
    hmm_best: pd.DataFrame,
    lengths: dict[str, int],
    retrieval_cfg: dict[str, Any],
    random_seed: int,
    background_ids: list[str],
    background_emb: np.ndarray,
) -> TrainingDataResult:
    """Build symmetric train data for learned scorer (positives vs background/hard negatives)."""
    if len(positive_ids) == 0:
        return TrainingDataResult(pd.DataFrame(), np.array([], dtype=int), 0, 0)

    # Positives: leave-one-out query construction for better train/inference symmetry.
    # This avoids trivially matching each positive to itself in support/nearest-anchor features.
    pos_feat = _build_positive_training_features_leave_one_out(
        positive_ids=positive_ids,
        positive_emb=positive_emb,
        positive_labels=positive_labels,
        medoids_df=medoids_df,
        metadata_df=metadata_df,
        mm_best=mm_best,
        hmm_best=hmm_best,
        lengths=lengths,
        retrieval_cfg=retrieval_cfg,
    )

    if len(background_ids) == 0:
        feats = pos_feat.copy()
        y = np.ones(len(feats), dtype=int)
        return TrainingDataResult(feats, y, len(pos_feat), 0)

    neg_n = min(int(retrieval_cfg.get("learned", {}).get("negative_background_n", 5000)), len(background_ids))
    rng = np.random.default_rng(random_seed)

    # Optional hard-negative preference: high similarity to positive prototypes but not easy-hit.
    use_hard = bool(retrieval_cfg.get("learned", {}).get("use_hard_negatives", True))
    if use_hard and len(background_ids) > neg_n:
        proto = []
        for c in sorted(set(positive_labels.tolist())):
            idx = np.where(positive_labels == c)[0]
            proto.append(positive_emb[idx].mean(axis=0))
        proto_m = np.vstack(proto)
        sim = 1.0 - _pairwise_cosine(background_emb, proto_m).min(axis=1)
        # select top hard pool then random sample for stability
        hard_pool_n = min(len(background_ids), max(neg_n * 3, neg_n))
        hard_idx = np.argsort(sim)[::-1][:hard_pool_n]
        sampled_idx = rng.choice(hard_idx, size=neg_n, replace=False)
    else:
        sampled_idx = rng.choice(np.arange(len(background_ids)), size=neg_n, replace=False)

    neg_ids = [background_ids[i] for i in sampled_idx]
    neg_emb = background_emb[sampled_idx]

    neg_feat = build_missed_candidate_features(
        missed_ids=neg_ids,
        unlabeled_ids=neg_ids,
        unlabeled_emb=neg_emb,
        positive_ids=positive_ids,
        positive_emb=positive_emb,
        positive_labels=positive_labels,
        lengths=lengths,
        medoids_df=medoids_df,
        metadata_df=metadata_df,
        mm_best=mm_best,
        hmm_best=hmm_best,
        cfg=retrieval_cfg,
    ).features

    feats = pd.concat([pos_feat, neg_feat], ignore_index=True, sort=False)
    y = np.array([1] * len(pos_feat) + [0] * len(neg_feat), dtype=int)
    return TrainingDataResult(feats, y, len(pos_feat), len(neg_feat))


def _build_positive_training_features_leave_one_out(
    *,
    positive_ids: list[str],
    positive_emb: np.ndarray,
    positive_labels: np.ndarray,
    medoids_df: pd.DataFrame,
    metadata_df: pd.DataFrame | None,
    mm_best: pd.DataFrame,
    hmm_best: pd.DataFrame,
    lengths: dict[str, int],
    retrieval_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Build positive-class training features using leave-one-out anchors."""
    if len(positive_ids) <= 1:
        return build_missed_candidate_features(
            missed_ids=positive_ids,
            unlabeled_ids=positive_ids,
            unlabeled_emb=positive_emb,
            positive_ids=positive_ids,
            positive_emb=positive_emb,
            positive_labels=positive_labels,
            lengths=lengths,
            medoids_df=medoids_df,
            metadata_df=metadata_df,
            mm_best=mm_best,
            hmm_best=hmm_best,
            cfg=retrieval_cfg,
        ).features

    rows: list[pd.DataFrame] = []
    for i, pid in enumerate(positive_ids):
        keep_idx = [j for j in range(len(positive_ids)) if j != i]
        if not keep_idx:
            continue
        medoids_sub = medoids_df
        if not medoids_df.empty and {"medoid_id"}.issubset(set(medoids_df.columns)):
            medoids_sub = medoids_df[medoids_df["medoid_id"] != pid].copy()

        sub = build_missed_candidate_features(
            missed_ids=[pid],
            unlabeled_ids=[pid],
            unlabeled_emb=positive_emb[i : i + 1],
            positive_ids=[positive_ids[j] for j in keep_idx],
            positive_emb=positive_emb[keep_idx],
            positive_labels=positive_labels[keep_idx],
            lengths=lengths,
            medoids_df=medoids_sub,
            metadata_df=metadata_df,
            mm_best=mm_best,
            hmm_best=hmm_best,
            cfg=retrieval_cfg,
        ).features
        rows.append(sub)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)


def _pairwise_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return 1.0 - (a_n @ b_n.T)
