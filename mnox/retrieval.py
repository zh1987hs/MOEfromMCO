from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors

from .io_fasta import write_fasta_records


def build_easy_and_missed_sets(
    candidate_ids: list[str], mm_flags: pd.DataFrame, hmm_flags: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """Build easy and missed candidate ID sets."""
    flags = pd.DataFrame({"candidate_id": candidate_ids})
    flags = flags.merge(mm_flags, on="candidate_id", how="left")
    flags = flags.merge(hmm_flags, on="candidate_id", how="left")
    flags["mmseqs_easy_hit_flag"] = flags["mmseqs_easy_hit_flag"].fillna(False)
    flags["hmm_easy_hit_flag"] = flags["hmm_easy_hit_flag"].fillna(False)
    flags["easy"] = flags["mmseqs_easy_hit_flag"] | flags["hmm_easy_hit_flag"]

    easy_ids = flags.loc[flags["easy"], "candidate_id"].tolist()
    missed_ids = flags.loc[~flags["easy"], "candidate_id"].tolist()
    return easy_ids, missed_ids


def rank_missed_candidates(
    missed_ids: list[str],
    unlabeled_ids: list[str],
    unlabeled_emb: np.ndarray,
    lengths: dict[str, int],
    prototypes: dict[int, np.ndarray],
    medoids_df: pd.DataFrame,
    mm_best: pd.DataFrame,
    hmm_best: pd.DataFrame,
    score_cfg: dict,
) -> pd.DataFrame:
    """Score and rank missed candidates via affinity/novelty/local support."""
    id_to_idx = {sid: i for i, sid in enumerate(unlabeled_ids)}
    miss_idx = [id_to_idx[mid] for mid in missed_ids if mid in id_to_idx]
    miss_emb = unlabeled_emb[miss_idx]

    proto_ids = sorted(prototypes.keys())
    proto_mat = np.vstack([prototypes[k] for k in proto_ids])
    dmat = pairwise_distances(miss_emb, proto_mat, metric=score_cfg.get("distance_metric", "cosine"))
    nearest_proto_idx = np.argmin(dmat, axis=1)
    nearest_proto = [proto_ids[i] for i in nearest_proto_idx]
    emb_dist = dmat[np.arange(len(miss_idx)), nearest_proto_idx]
    emb_sim = np.exp(-emb_dist)

    mm_map = mm_best.set_index("query_id") if not mm_best.empty else pd.DataFrame()
    hmm_map = hmm_best.set_index("candidate_id") if not hmm_best.empty else pd.DataFrame()

    novelty = []
    for cid in missed_ids:
        mm_novel = 1.0
        hmm_novel = 1.0

        if not mm_best.empty and cid in mm_map.index:
            mm_fident = float(mm_map.loc[cid, "fident"])
            mm_qcov = float(mm_map.loc[cid, "qcov"])
            mm_novel = max(0.0, 1.0 - 0.6 * mm_fident - 0.4 * mm_qcov)

        if not hmm_best.empty and cid in hmm_map.index:
            hv = float(hmm_map.loc[cid, "hmm_best_evalue"])
            hmm_novel = min(1.0, np.log10(max(hv, 1e-300) + 1.0))
            hmm_novel = max(0.0, hmm_novel)

        novelty.append(min(score_cfg.get("novelty_cap", 0.95), 0.5 * mm_novel + 0.5 * hmm_novel))

    k = int(score_cfg["knn_k"])
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(miss_emb)), metric="cosine")
    nn.fit(miss_emb)
    dist, _ = nn.kneighbors(miss_emb)
    mean_neighbor_dist = dist[:, 1:].mean(axis=1) if dist.shape[1] > 1 else np.ones(len(miss_emb))
    local_support = np.exp(-mean_neighbor_dist)

    medoid_map = medoids_df.set_index("cluster")["medoid_id"].to_dict()
    df = pd.DataFrame(
        {
            "candidate_id": missed_ids,
            "length": [lengths.get(cid, -1) for cid in missed_ids],
            "nearest_positive_cluster": nearest_proto,
            "nearest_positive_id": [medoid_map.get(c, "NA") for c in nearest_proto],
            "embedding_distance": emb_dist,
            "embedding_similarity": emb_sim,
            "novelty_score": novelty,
            "local_support_score": local_support,
        }
    )

    mm_cols = ["query_id", "target_id", "fident", "qcov", "evalue"]
    if not mm_best.empty:
        mm_meta = mm_best[mm_cols].rename(
            columns={
                "query_id": "candidate_id",
                "target_id": "mmseqs_best_target",
                "fident": "mmseqs_best_fident",
                "qcov": "mmseqs_best_qcov",
                "evalue": "mmseqs_best_evalue",
            }
        )
        df = df.merge(mm_meta, on="candidate_id", how="left")

    if not hmm_best.empty:
        df = df.merge(hmm_best, on="candidate_id", how="left")

    if "hmm_best_cluster" in df:
        df = df.rename(
            columns={
                "hmm_best_cluster": "hmm_best_cluster",
                "hmm_best_bitscore": "hmm_best_bitscore",
                "hmm_best_evalue": "hmm_best_evalue",
            }
        )

    w1 = score_cfg["w_affinity"]
    w2 = score_cfg["w_novelty"]
    w3 = score_cfg["w_local_support"]
    df["final_score"] = w1 * df["embedding_similarity"] + w2 * df["novelty_score"] + w3 * df["local_support_score"]
    df = df.sort_values("final_score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    return df


def export_top_candidates(
    ranked_df: pd.DataFrame,
    seqs: dict[str, str],
    out_dir: str | Path,
    top_n: int,
) -> None:
    """Export top candidates FASTA globally and by nearest cluster."""
    out_dir = Path(out_dir)
    top = ranked_df.head(top_n)

    records = [SeqRecord(Seq(seqs[cid]), id=cid, description="") for cid in top["candidate_id"] if cid in seqs]
    write_fasta_records(records, out_dir / "top_candidates.fasta")

    by_cluster_dir = out_dir / "top_candidates_by_cluster"
    by_cluster_dir.mkdir(parents=True, exist_ok=True)
    for cluster_id, sub in top.groupby("nearest_positive_cluster"):
        recs = [SeqRecord(Seq(seqs[cid]), id=cid, description="") for cid in sub["candidate_id"] if cid in seqs]
        write_fasta_records(recs, by_cluster_dir / f"cluster_{cluster_id}.fasta")
