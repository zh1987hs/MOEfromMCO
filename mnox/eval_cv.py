from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Seq import Seq
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
    unl_seqs = {r.id: str(r.seq) for r in unlabeled_records}
    lengths = {r.id: len(r.seq) for r in unlabeled_records}

    fold_rows: list[dict] = []
    clusters = sorted(set(positive_labels.tolist()))
    for fold_cluster in clusters:
        fold_dir = ensure_dir(cv_dir / f"fold_cluster_{fold_cluster}")
        holdout_ids = [pid for pid, c in zip(positive_ids, positive_labels) if int(c) == int(fold_cluster)]
        train_ids = [pid for pid, c in zip(positive_ids, positive_labels) if int(c) != int(fold_cluster)]

        bg_n = min(cv_cfg["background_n"], len(unlabeled_ids))
        bg_ids = rng.choice(unlabeled_ids, size=bg_n, replace=False).tolist()
        query_ids = bg_ids + holdout_ids

        train_fasta = fold_dir / "train_positives.fasta"
        query_fasta = fold_dir / "query.fasta"
        write_fasta_records([pos_id_to_rec[x] for x in train_ids], train_fasta)
        q_records = [unl_id_to_rec[x] for x in bg_ids] + [pos_id_to_rec[x] for x in holdout_ids]
        write_fasta_records(q_records, query_fasta)

        mm_df = run_mmseqs_search(train_fasta, query_fasta, fold_dir, cfg["mmseqs"], logger)
        hmm_paths = build_cluster_hmms({0: [pos_id_to_rec[x] for x in train_ids]}, fold_dir, cfg["hmmer"]["msa_tool"], cfg["hmmer"]["threads"], logger)
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
        for name, rank_ids in [
            ("mmseqs_only", mm_rank),
            ("hmmer_only", hmm_rank),
            ("embedding_only", emb_rank),
            ("hybrid", hybrid_rank),
        ]:
            row = {"fold_cluster": fold_cluster, "method": name, "mrr": _mrr(rank_ids, true_set)}
            for k in k_values:
                row[f"recall@{k}"] = _recall_at_k(rank_ids, true_set, k)
                row[f"ef@{k}"] = _ef_at_k(rank_ids, true_set, k)
            fold_rows.append(row)

        write_dataframe(ranked_hybrid, fold_dir / "hybrid_ranked", logger)

    summary = pd.DataFrame(fold_rows)
    return summary
