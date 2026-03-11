from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.SeqRecord import SeqRecord
from sklearn.metrics import pairwise_distances

from .hmmer import build_cluster_hmms, run_hmmsearch_for_clusters
from .io_fasta import write_fasta_records
from .mmseqs import run_mmseqs_search
from .retrieval import rank_missed_candidates
from .utils import ensure_dir, write_dataframe


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


def _run_single_fold(
    fold_name: str,
    train_ids: list[str],
    holdout_ids: list[str],
    pos_id_to_rec: dict[str, SeqRecord],
    unl_id_to_rec: dict[str, SeqRecord],
    positive_ids: list[str],
    positive_emb: np.ndarray,
    unlabeled_ids: list[str],
    unlabeled_emb: np.ndarray,
    cfg: dict,
    fold_dir: Path,
    logger: logging.Logger,
    k_values: list[int],
    rng: np.random.Generator,
) -> list[dict]:
    cv_cfg = cfg["cv"]
    lengths = {**{k: len(v.seq) for k, v in pos_id_to_rec.items()}, **{k: len(v.seq) for k, v in unl_id_to_rec.items()}}

    bg_n = min(cv_cfg["background_n"], len(unlabeled_ids))
    bg_ids = rng.choice(unlabeled_ids, size=bg_n, replace=False).tolist()
    query_ids = bg_ids + holdout_ids

    train_fasta = fold_dir / "train_positives.fasta"
    query_fasta = fold_dir / "query.fasta"
    write_fasta_records([pos_id_to_rec[x] for x in train_ids], train_fasta)
    q_records = [unl_id_to_rec[x] for x in bg_ids] + [pos_id_to_rec[x] for x in holdout_ids]
    write_fasta_records(q_records, query_fasta)

    mm_df = run_mmseqs_search(train_fasta, query_fasta, fold_dir, cfg["mmseqs"], logger)
    hmm_paths = build_cluster_hmms(
        {0: [pos_id_to_rec[x] for x in train_ids]},
        fold_dir,
        cfg["hmmer"]["msa_tool"],
        cfg["hmmer"]["threads"],
        logger,
    )
    hmm_df = run_hmmsearch_for_clusters(hmm_paths, query_fasta, fold_dir, cfg["hmmer"], logger)

    train_idx = [positive_ids.index(x) for x in train_ids]
    proto = {0: positive_emb[train_idx].mean(axis=0)}

    query_emb = []
    for qid in query_ids:
        if qid in unlabeled_ids:
            query_emb.append(unlabeled_emb[unlabeled_ids.index(qid)])
        else:
            query_emb.append(positive_emb[positive_ids.index(qid)])
    query_emb = np.vstack(query_emb)

    d = pairwise_distances(query_emb, np.vstack([proto[0]]), metric="cosine").reshape(-1)
    emb_rank = [x for _, x in sorted(zip(d, query_ids), key=lambda t: t[0])]

    mm_rank = (
        mm_df.sort_values(["bits", "evalue"], ascending=[False, True])["query_id"].drop_duplicates().tolist()
        if not mm_df.empty
        else []
    )
    hmm_rank = (
        hmm_df.sort_values(["hmm_best_bitscore", "hmm_best_evalue"], ascending=[False, True])["candidate_id"].drop_duplicates().tolist()
        if not hmm_df.empty
        else []
    )

    ranked_hybrid = rank_missed_candidates(
        missed_ids=query_ids,
        unlabeled_ids=query_ids,
        unlabeled_emb=query_emb,
        lengths=lengths,
        prototypes=proto,
        medoids_df=pd.DataFrame([{"cluster": 0, "medoid_id": train_ids[0]}]),
        mm_best=mm_df,
        hmm_best=hmm_df,
        score_cfg=cfg["retrieval"],
    )
    hybrid_rank = ranked_hybrid["candidate_id"].tolist()

    true_set = set(holdout_ids)
    rows = []
    for name, rank_ids in [
        ("mmseqs_only", mm_rank),
        ("hmmer_only", hmm_rank),
        ("embedding_only", emb_rank),
        ("hybrid", hybrid_rank),
    ]:
        row = {
            "fold": fold_name,
            "method": name,
            "mrr": _mrr(rank_ids, true_set),
        }
        for k in k_values:
            row[f"recall@{k}"] = _recall_at_k(rank_ids, true_set, k)
            row[f"ef@{k}"] = _ef_at_k(rank_ids, true_set, k)
        rows.append(row)

    write_dataframe(ranked_hybrid, fold_dir / "hybrid_ranked", logger)
    return rows


def run_loco_cv(
    positive_records: list[SeqRecord],
    positive_ids: list[str],
    positive_labels: np.ndarray,
    positive_emb: np.ndarray,
    unlabeled_records: list[SeqRecord],
    unlabeled_ids: list[str],
    unlabeled_emb: np.ndarray,
    cfg: dict,
    run_dir: str | Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Run leave-one-cluster-out validation with baseline and hybrid rankers."""
    cv_dir = ensure_dir(Path(run_dir) / "cv_per_fold_details")
    rng = np.random.default_rng(cfg["random_seed"])
    cv_cfg = cfg["cv"]
    k_values = cv_cfg["k_values"]

    pos_id_to_rec = {r.id: r for r in positive_records}
    unl_id_to_rec = {r.id: r for r in unlabeled_records}

    fold_rows: list[dict] = []
    clusters = sorted(set(positive_labels.tolist()))
    for fold_cluster in clusters:
        fold_dir = ensure_dir(cv_dir / f"fold_cluster_{fold_cluster}")
        holdout_ids = [pid for pid, c in zip(positive_ids, positive_labels) if int(c) == int(fold_cluster)]
        train_ids = [pid for pid, c in zip(positive_ids, positive_labels) if int(c) != int(fold_cluster)]

        rows = _run_single_fold(
            fold_name=f"cluster_{fold_cluster}",
            train_ids=train_ids,
            holdout_ids=holdout_ids,
            pos_id_to_rec=pos_id_to_rec,
            unl_id_to_rec=unl_id_to_rec,
            positive_ids=positive_ids,
            positive_emb=positive_emb,
            unlabeled_ids=unlabeled_ids,
            unlabeled_emb=unlabeled_emb,
            cfg=cfg,
            fold_dir=fold_dir,
            logger=logger,
            k_values=k_values,
            rng=rng,
        )
        for r in rows:
            r["evaluation_mode"] = "loco"
            r["fold_cluster"] = fold_cluster
        fold_rows.extend(rows)

    return pd.DataFrame(fold_rows)


def run_leave_one_gold_family_out_cv(
    positive_records: list[SeqRecord],
    positive_ids: list[str],
    positive_emb: np.ndarray,
    unlabeled_records: list[SeqRecord],
    unlabeled_ids: list[str],
    unlabeled_emb: np.ndarray,
    metadata_df: pd.DataFrame,
    cfg: dict,
    run_dir: str | Path,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Run leave-one-gold-family-out CV with leakage guard over gold+silver positives."""
    cv_dir = ensure_dir(Path(run_dir) / "cv_per_fold_details")
    rng = np.random.default_rng(cfg["random_seed"])
    cv_cfg = cfg["cv"]
    k_values = cv_cfg["k_values"]

    eval_cfg = cfg.get("evaluation", {})
    leakage_guard = bool(eval_cfg.get("leakage_guard", True))

    pos_id_to_rec = {r.id: r for r in positive_records}
    unl_id_to_rec = {r.id: r for r in unlabeled_records}

    required = {"positive_id", "gold_family"}
    if not required.issubset(metadata_df.columns):
        raise RuntimeError(f"Metadata missing required columns for family-aware CV: {required - set(metadata_df.columns)}")

    families = sorted(metadata_df["gold_family"].dropna().unique().tolist())
    fold_rows: list[dict] = []

    for fam in families:
        fold_dir = ensure_dir(cv_dir / f"fold_gold_family_{fam}")
        holdout_ids = metadata_df.loc[metadata_df["gold_family"] == fam, "positive_id"].tolist()

        if leakage_guard:
            train_ids = metadata_df.loc[metadata_df["gold_family"] != fam, "positive_id"].tolist()
        else:
            train_ids = [x for x in positive_ids if x not in set(holdout_ids)]

        holdout_ids = [x for x in holdout_ids if x in pos_id_to_rec]
        train_ids = [x for x in train_ids if x in pos_id_to_rec]

        rows = _run_single_fold(
            fold_name=f"gold_family_{fam}",
            train_ids=train_ids,
            holdout_ids=holdout_ids,
            pos_id_to_rec=pos_id_to_rec,
            unl_id_to_rec=unl_id_to_rec,
            positive_ids=positive_ids,
            positive_emb=positive_emb,
            unlabeled_ids=unlabeled_ids,
            unlabeled_emb=unlabeled_emb,
            cfg=cfg,
            fold_dir=fold_dir,
            logger=logger,
            k_values=k_values,
            rng=rng,
        )
        for r in rows:
            r["evaluation_mode"] = "leave_one_gold_family_out"
            r["fold_gold_family"] = fam
            r["leakage_guard"] = leakage_guard
        fold_rows.extend(rows)

    return pd.DataFrame(fold_rows)
