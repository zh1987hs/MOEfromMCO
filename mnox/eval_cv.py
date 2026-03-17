from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.SeqRecord import SeqRecord
from sklearn.metrics import pairwise_distances

from .cluster import compute_prototypes_and_medoids, pick_best_clustering
from .features import build_missed_candidate_features
from .hmmer import build_cluster_hmms, make_hmm_easy_flags, run_hmmsearch_for_clusters
from .io_fasta import write_fasta_records
from .mmseqs import make_easy_hit_flags, run_mmseqs_search
from .retrieval import build_easy_and_missed_sets, rank_candidates_with_policy
from .scoring import apply_heuristic_scorer
from .training_data import build_learned_training_data
from .utils import ensure_dir


def _recall_at_k(sorted_ids: list[str], true_ids: set[str], k: int) -> float:
    top = sorted_ids[:k]
    hit = sum(1 for i in top if i in true_ids)
    return hit / max(1, len(true_ids))


def _mrr(sorted_ids: list[str], true_ids: set[str]) -> float:
    for i, sid in enumerate(sorted_ids, 1):
        if sid in true_ids:
            return 1.0 / i
    return 0.0


def _ef_at_k(sorted_ids: list[str], true_ids: set[str], k: int) -> float:
    top = sorted_ids[:k]
    hits = sum(1 for i in top if i in true_ids)
    expected = k * (len(true_ids) / max(1, len(sorted_ids)))
    return hits / max(expected, 1e-9)


def _pick_fold_train_labels(
    train_ids: list[str],
    all_pos_ids: list[str],
    all_pos_emb: np.ndarray,
    cfg: dict,
    logger: logging.Logger,
) -> np.ndarray:
    mode = cfg["positive_clustering"]["mode"]
    if mode == "none":
        return np.zeros(len(train_ids), dtype=int)

    if mode == "auto":
        idx = [all_pos_ids.index(x) for x in train_ids]
        emb = all_pos_emb[idx]
        cl_cfg = cfg["positive_clustering"]
        out = pick_best_clustering(
            emb,
            cl_cfg["methods"],
            cl_cfg["k_min"],
            cl_cfg["k_max"],
            cfg["random_seed"],
            logger=logger,
        )
        return out.labels

    # fixed mode in CV: subset of precomputed labels from discovery positives ordering
    # cfg injected at caller with positive_labels_map
    lmap = cfg.get("_positive_labels_map", {})
    return np.array([int(lmap[x]) for x in train_ids], dtype=int)


def _build_mm_hmm_for_fold(
    train_ids: list[str],
    query_ids: list[str],
    pos_id_to_rec: dict[str, SeqRecord],
    unl_id_to_rec: dict[str, SeqRecord],
    train_labels: np.ndarray,
    fold_dir: Path,
    cfg: dict,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_fasta = fold_dir / "train_positives.fasta"
    query_fasta = fold_dir / "query.fasta"
    write_fasta_records([pos_id_to_rec[x] for x in train_ids], train_fasta)

    q_records = [unl_id_to_rec[x] for x in query_ids if x in unl_id_to_rec] + [pos_id_to_rec[x] for x in query_ids if x in pos_id_to_rec]
    write_fasta_records(q_records, query_fasta)

    mm_df = run_mmseqs_search(train_fasta, query_fasta, fold_dir, cfg["mmseqs"], logger)
    mm_flags = make_easy_hit_flags(mm_df, cfg["easy_hit"]["mmseqs"])

    cluster_to_records: dict[int, list[SeqRecord]] = {}
    for c in sorted(set(train_labels.tolist())):
        members = [pid for pid, lab in zip(train_ids, train_labels) if int(lab) == int(c)]
        cluster_to_records[int(c)] = [pos_id_to_rec[x] for x in members]

    hmm_paths = build_cluster_hmms(cluster_to_records, fold_dir, cfg["hmmer"]["msa_tool"], cfg["hmmer"]["threads"], logger)
    hmm_best = run_hmmsearch_for_clusters(hmm_paths, query_fasta, fold_dir, cfg["hmmer"], logger)
    hmm_flags = make_hmm_easy_flags(hmm_best, cfg["easy_hit"]["hmm"])

    return mm_df, mm_flags, hmm_best, hmm_flags


def _evaluate_rank(
    rank_ids: list[str],
    true_set: set[str],
    k_values: list[int],
) -> dict[str, float]:
    row = {"mrr": _mrr(rank_ids, true_set)}
    for k in k_values:
        row[f"recall@{k}"] = _recall_at_k(rank_ids, true_set, k)
        row[f"ef@{k}"] = _ef_at_k(rank_ids, true_set, k)
    return row


def _run_fold_aligned(
    fold_name: str,
    train_ids: list[str],
    holdout_ids: list[str],
    positive_ids: list[str],
    positive_emb: np.ndarray,
    metadata_df: pd.DataFrame,
    unlabeled_ids: list[str],
    unlabeled_emb: np.ndarray,
    pos_id_to_rec: dict[str, SeqRecord],
    unl_id_to_rec: dict[str, SeqRecord],
    cfg: dict,
    fold_dir: Path,
    logger: logging.Logger,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    k_values = cfg["cv"]["k_values"]

    bg_n = min(cfg["cv"]["background_n"], len(unlabeled_ids))
    bg_ids = rng.choice(unlabeled_ids, size=bg_n, replace=False).tolist()
    query_ids = bg_ids + holdout_ids

    train_labels = _pick_fold_train_labels(train_ids, positive_ids, positive_emb, cfg, logger)

    mm_df, mm_flags, hmm_best, hmm_flags = _build_mm_hmm_for_fold(
        train_ids,
        query_ids,
        pos_id_to_rec,
        unl_id_to_rec,
        train_labels,
        fold_dir,
        cfg,
        logger,
    )

    easy_ids, missed_ids = build_easy_and_missed_sets(query_ids, mm_flags, hmm_flags)

    # fold prototypes/medoids from train only
    train_idx = [positive_ids.index(x) for x in train_ids]
    train_emb = positive_emb[train_idx]
    _, _, medoids = compute_prototypes_and_medoids(train_ids, train_emb, train_labels)

    lengths = {**{k: len(v.seq) for k, v in pos_id_to_rec.items()}, **{k: len(v.seq) for k, v in unl_id_to_rec.items()}}

    feat_res = build_missed_candidate_features(
        missed_ids=missed_ids,
        unlabeled_ids=query_ids,
        unlabeled_emb=np.vstack([
            unlabeled_emb[unlabeled_ids.index(q)] if q in unlabeled_ids else positive_emb[positive_ids.index(q)] for q in query_ids
        ]),
        positive_ids=train_ids,
        positive_emb=train_emb,
        positive_labels=train_labels,
        lengths=lengths,
        medoids_df=medoids,
        metadata_df=metadata_df,
        mm_best=mm_df,
        hmm_best=hmm_best,
        cfg=cfg["retrieval"],
    )

    # Baselines and hybrid variants
    true_set = set(holdout_ids)
    rows: list[dict[str, Any]] = []

    mm_rank = (
        mm_df.sort_values(["bits", "evalue"], ascending=[False, True])["query_id"].drop_duplicates().tolist()
        if not mm_df.empty
        else []
    )
    hm_rank = (
        hmm_best.sort_values(["hmm_best_bitscore", "hmm_best_evalue"], ascending=[False, True])["candidate_id"].drop_duplicates().tolist()
        if not hmm_best.empty
        else []
    )
    # embedding-only over all query by nearest train mean
    q_emb = np.vstack([
        unlabeled_emb[unlabeled_ids.index(q)] if q in unlabeled_ids else positive_emb[positive_ids.index(q)] for q in query_ids
    ])
    d = pairwise_distances(q_emb, train_emb.mean(axis=0, keepdims=True), metric="cosine").reshape(-1)
    emb_rank = [x for _, x in sorted(zip(d, query_ids), key=lambda t: t[0])]

    for method, rids in [
        ("mmseqs_only", mm_rank),
        ("hmmer_only", hm_rank),
        ("embedding_only", emb_rank),
    ]:
        met = _evaluate_rank(rids, true_set, k_values)
        rows.append({"method": method, "scoring_mode": method, **met})

    # aligned hybrid
    ranked_h, _, _ = rank_candidates_with_policy(
        candidate_ids=query_ids,
        easy_ids=easy_ids,
        missed_ids=missed_ids,
        mm_best=mm_df,
        hmm_best=hmm_best,
        mm_flags=mm_flags,
        hmm_flags=hmm_flags,
        feature_result=feat_res,
        retrieval_cfg=cfg["retrieval"],
        scorer_mode="heuristic",
    )
    rows.append({"method": "hybrid_heuristic", "scoring_mode": "heuristic", **_evaluate_rank(ranked_h["candidate_id"].tolist(), true_set, k_values)})

    # ablations
    for variant, mname in [("no_novelty", "hybrid_no_novelty"), ("no_support", "hybrid_no_support")]:
        scored = apply_heuristic_scorer(feat_res.features, cfg["retrieval"], variant=variant).scored
        # keep easy prepend behavior
        easy_rank = ranked_h[ranked_h.get("easy_or_missed", "") == "easy"]["candidate_id"].tolist() if "easy_or_missed" in ranked_h else []
        rank_ids = easy_rank + scored["candidate_id"].tolist()
        rows.append({"method": mname, "scoring_mode": f"heuristic_{variant}", **_evaluate_rank(rank_ids, true_set, k_values)})

    # no easy split
    all_feat = build_missed_candidate_features(
        missed_ids=query_ids,
        unlabeled_ids=query_ids,
        unlabeled_emb=q_emb,
        positive_ids=train_ids,
        positive_emb=train_emb,
        positive_labels=train_labels,
        lengths=lengths,
        medoids_df=medoids,
        metadata_df=metadata_df,
        mm_best=mm_df,
        hmm_best=hmm_best,
        cfg=cfg["retrieval"],
    ).features
    no_easy = apply_heuristic_scorer(all_feat, cfg["retrieval"]).scored
    rows.append({"method": "hybrid_no_easy_split", "scoring_mode": "heuristic", **_evaluate_rank(no_easy["candidate_id"].tolist(), true_set, k_values)})

    # learned (fold-local) with train positives vs sampled background
    learned_enabled = bool(cfg["retrieval"].get("learned", {}).get("enabled", True))
    if learned_enabled and len(feat_res.features) > 5 and len(train_ids) > 3:
        try:
            bg_emb = np.vstack([unlabeled_emb[unlabeled_ids.index(q)] for q in bg_ids]) if len(bg_ids) else np.empty((0, train_emb.shape[1]))
            t_res = build_learned_training_data(
                positive_ids=train_ids,
                positive_emb=train_emb,
                positive_labels=train_labels,
                medoids_df=medoids,
                metadata_df=metadata_df,
                mm_best=mm_df,
                hmm_best=hmm_best,
                lengths=lengths,
                retrieval_cfg=cfg["retrieval"],
                random_seed=cfg["random_seed"],
                background_ids=bg_ids,
                background_emb=bg_emb,
            )
            ranked_l, _, _ = rank_candidates_with_policy(
                candidate_ids=query_ids,
                easy_ids=easy_ids,
                missed_ids=missed_ids,
                mm_best=mm_df,
                hmm_best=hmm_best,
                mm_flags=mm_flags,
                hmm_flags=hmm_flags,
                feature_result=feat_res,
                retrieval_cfg=cfg["retrieval"],
                scorer_mode="learned",
                train_feature_df=t_res.features,
                train_labels=t_res.labels,
            )
            rows.append({"method": "hybrid_learned", "scoring_mode": "learned", **_evaluate_rank(ranked_l["candidate_id"].tolist(), true_set, k_values)})
        except Exception:
            pass

    diag = {
        "fold": fold_name,
        "n_easy": len(easy_ids),
        "n_missed": len(missed_ids),
        "n_train_pos": len(train_ids),
        "n_holdout_pos": len(holdout_ids),
    }
    for r in rows:
        r.update(diag)
    return rows, diag


def run_loco_cv(
    positive_records: list[SeqRecord],
    positive_ids: list[str],
    positive_labels: np.ndarray,
    positive_emb: np.ndarray,
    metadata_df: pd.DataFrame,
    unlabeled_records: list[SeqRecord],
    unlabeled_ids: list[str],
    unlabeled_emb: np.ndarray,
    cfg: dict,
    run_dir: str | Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """LOCO CV aligned to discovery flow (easy/missed + hybrid rerank)."""
    cv_dir = ensure_dir(Path(run_dir) / "cv_per_fold_details")
    rng = np.random.default_rng(cfg["random_seed"])

    pos_id_to_rec = {r.id: r for r in positive_records}
    unl_id_to_rec = {r.id: r for r in unlabeled_records}

    cfg = dict(cfg)
    cfg["_positive_labels_map"] = {pid: int(lab) for pid, lab in zip(positive_ids, positive_labels)}

    rows_all: list[dict[str, Any]] = []
    diags = {"folds": []}
    for c in sorted(set(positive_labels.tolist())):
        hold = [pid for pid, lab in zip(positive_ids, positive_labels) if int(lab) == int(c)]
        train = [pid for pid, lab in zip(positive_ids, positive_labels) if int(lab) != int(c)]
        fold_dir = ensure_dir(cv_dir / f"fold_cluster_{c}")
        rows, diag = _run_fold_aligned(
            fold_name=f"cluster_{c}",
            train_ids=train,
            holdout_ids=hold,
            positive_ids=positive_ids,
            positive_emb=positive_emb,
            metadata_df=metadata_df,
            unlabeled_ids=unlabeled_ids,
            unlabeled_emb=unlabeled_emb,
            pos_id_to_rec=pos_id_to_rec,
            unl_id_to_rec=unl_id_to_rec,
            cfg=cfg,
            fold_dir=fold_dir,
            logger=logger,
            rng=rng,
        )
        for r in rows:
            r["evaluation_mode"] = "loco"
            r["fold_cluster"] = c
            r["easy_hit_policy"] = cfg["retrieval"].get("easy_hit_policy", "prepend")
            r["leakage_guard"] = cfg["evaluation"].get("leakage_guard", True)
        rows_all.extend(rows)
        diags["folds"].append(diag)

    return pd.DataFrame(rows_all), diags


def run_leave_one_gold_family_out_cv(
    positive_records: list[SeqRecord],
    positive_ids: list[str],
    positive_emb: np.ndarray,
    positive_labels: np.ndarray,
    metadata_df: pd.DataFrame,
    unlabeled_records: list[SeqRecord],
    unlabeled_ids: list[str],
    unlabeled_emb: np.ndarray,
    cfg: dict,
    run_dir: str | Path,
    logger: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Family-aware CV aligned to discovery flow and leakage-safe by default."""
    cv_dir = ensure_dir(Path(run_dir) / "cv_per_fold_details")
    rng = np.random.default_rng(cfg["random_seed"])

    pos_id_to_rec = {r.id: r for r in positive_records}
    unl_id_to_rec = {r.id: r for r in unlabeled_records}

    cfg = dict(cfg)
    cfg["_positive_labels_map"] = {pid: int(lab) for pid, lab in zip(positive_ids, positive_labels)}

    leakage_guard = bool(cfg.get("evaluation", {}).get("leakage_guard", True))
    rows_all: list[dict[str, Any]] = []
    diags = {"folds": []}

    for fam in sorted(metadata_df["gold_family"].dropna().unique().tolist()):
        hold = metadata_df.loc[metadata_df["gold_family"] == fam, "positive_id"].tolist()
        if leakage_guard:
            train = metadata_df.loc[metadata_df["gold_family"] != fam, "positive_id"].tolist()
        else:
            train = [x for x in positive_ids if x not in set(hold)]

        hold = [x for x in hold if x in pos_id_to_rec]
        train = [x for x in train if x in pos_id_to_rec]
        fold_dir = ensure_dir(cv_dir / f"fold_gold_family_{fam}")

        rows, diag = _run_fold_aligned(
            fold_name=f"gold_family_{fam}",
            train_ids=train,
            holdout_ids=hold,
            positive_ids=positive_ids,
            positive_emb=positive_emb,
            metadata_df=metadata_df,
            unlabeled_ids=unlabeled_ids,
            unlabeled_emb=unlabeled_emb,
            pos_id_to_rec=pos_id_to_rec,
            unl_id_to_rec=unl_id_to_rec,
            cfg=cfg,
            fold_dir=fold_dir,
            logger=logger,
            rng=rng,
        )
        for r in rows:
            r["evaluation_mode"] = "leave_one_gold_family_out"
            r["fold_gold_family"] = fam
            r["easy_hit_policy"] = cfg["retrieval"].get("easy_hit_policy", "prepend")
            r["leakage_guard"] = leakage_guard
        rows_all.extend(rows)
        diags["folds"].append(diag)

    return pd.DataFrame(rows_all), diags
