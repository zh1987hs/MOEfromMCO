from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from .features import FeatureBuildResult, build_easy_rank_table, build_missed_candidate_features
from .io_fasta import write_fasta_records
from .scoring import ScoringResult, apply_heuristic_scorer, apply_learned_scorer


def build_easy_and_missed_sets(
    candidate_ids: list[str], mm_flags: pd.DataFrame, hmm_flags: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """Build easy and missed candidate ID sets."""
    flags = pd.DataFrame({"candidate_id": candidate_ids})
    flags = flags.merge(mm_flags, on="candidate_id", how="left")
    flags = flags.merge(hmm_flags, on="candidate_id", how="left")
    flags["mmseqs_easy_hit_flag"] = flags["mmseqs_easy_hit_flag"].fillna(False).astype(bool)
    flags["hmm_easy_hit_flag"] = flags["hmm_easy_hit_flag"].fillna(False).astype(bool)
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
    """Backward-compatible missed ranking entry (heuristic-only fallback)."""
    if len(missed_ids) == 0:
        return pd.DataFrame()

    # Build pseudo positive labels from prototypes if needed.
    proto_ids = sorted(prototypes.keys())
    positive_ids = medoids_df["medoid_id"].tolist() if not medoids_df.empty else [f"cluster_{x}" for x in proto_ids]
    positive_emb = np.vstack([prototypes[k] for k in proto_ids])
    positive_labels = np.arange(len(proto_ids), dtype=int)

    feat_res = build_missed_candidate_features(
        missed_ids=missed_ids,
        unlabeled_ids=unlabeled_ids,
        unlabeled_emb=unlabeled_emb,
        positive_ids=positive_ids,
        positive_emb=positive_emb,
        positive_labels=positive_labels,
        lengths=lengths,
        medoids_df=medoids_df,
        metadata_df=None,
        mm_best=mm_best,
        hmm_best=hmm_best,
        cfg=score_cfg,
    )
    return apply_heuristic_scorer(feat_res.features, score_cfg).scored


def rank_candidates_with_policy(
    candidate_ids: list[str],
    easy_ids: list[str],
    missed_ids: list[str],
    mm_best: pd.DataFrame,
    hmm_best: pd.DataFrame,
    mm_flags: pd.DataFrame,
    hmm_flags: pd.DataFrame,
    feature_result: FeatureBuildResult,
    retrieval_cfg: dict[str, Any],
    scorer_mode: str = "heuristic",
    train_feature_df: pd.DataFrame | None = None,
    train_labels: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Rank easy + missed candidates with configurable scorer and merge policy."""
    # Respect upstream split explicitly; avoid recomputing semantic split silently.
    cand_set = set(candidate_ids)
    easy_ids = [x for x in easy_ids if x in cand_set]
    missed_ids = [x for x in missed_ids if x in cand_set]

    easy_df = build_easy_rank_table(easy_ids, mm_best, hmm_best, mm_flags, hmm_flags)

    missed_feature_df = feature_result.features.copy()
    if not missed_feature_df.empty:
        missed_feature_df = missed_feature_df[missed_feature_df["candidate_id"].isin(missed_ids)].copy()

    if missed_feature_df.empty:
        missed_df = pd.DataFrame(columns=["candidate_id", "final_score", "rank", "scoring_mode", "easy_or_missed"])
        fi = None
    else:
        if scorer_mode == "learned" and train_feature_df is not None and train_labels is not None:
            try:
                scored_res: ScoringResult = apply_learned_scorer(
                    missed_feature_df,
                    train_feature_df,
                    train_labels,
                    retrieval_cfg,
                )
            except Exception:
                scored_res = apply_heuristic_scorer(missed_feature_df, retrieval_cfg)
        else:
            scored_res = apply_heuristic_scorer(missed_feature_df, retrieval_cfg)

        missed_df = scored_res.scored.copy()
        missed_df["easy_or_missed"] = "missed"
        fi = scored_res.feature_importance

    policy = retrieval_cfg.get("easy_hit_policy", "prepend")
    if policy == "merge":
        # normalize score spaces before merge
        e = easy_df.copy()
        if len(e) > 0:
            e["final_score"] = (e["easy_score"] - e["easy_score"].min()) / (e["easy_score"].max() - e["easy_score"].min() + 1e-9)
            e["scoring_mode"] = "easy_strength"
            e["confidence_tier"] = "high"
            e["reason_for_high_rank"] = "strong_easy_hit"
            e["dominant_signal_type"] = "mmseqs/hmm"
            e["flags"] = ""
        combined = pd.concat([e, missed_df], ignore_index=True, sort=False).sort_values("final_score", ascending=False)
    else:
        combined = pd.concat([easy_df, missed_df], ignore_index=True, sort=False)

    for col, default in [
        ("false_positive_risk", 0.0),
        ("generic_mco_risk_score", 0.0),
        ("embedding_similarity", np.nan),
        ("positive_support_score", np.nan),
        ("local_density_score", np.nan),
        ("novelty_score", np.nan),
    ]:
        if col not in combined.columns:
            combined[col] = default

    combined = combined.drop_duplicates("candidate_id", keep="first").reset_index(drop=True)
    combined["rank"] = np.arange(1, len(combined) + 1)

    # Experimental priority rank: score-aware but risk-penalized triage index.
    pcfg = retrieval_cfg.get("experimental_priority", {}) if isinstance(retrieval_cfg, dict) else {}
    w_fp = float(pcfg.get("w_false_positive_risk", 0.20))
    w_gr = float(pcfg.get("w_generic_mco_risk", 0.10))
    if "final_score" in combined.columns:
        exp_priority = combined["final_score"].fillna(0.0).astype(float)
        if "false_positive_risk" in combined.columns:
            exp_priority = exp_priority - w_fp * combined["false_positive_risk"].fillna(0.0).astype(float)
        if "generic_mco_risk_score" in combined.columns:
            exp_priority = exp_priority - w_gr * combined["generic_mco_risk_score"].fillna(0.0).astype(float)
        combined["experimental_priority_score"] = exp_priority
        combined["experimental_priority_rank"] = (
            combined["experimental_priority_score"].rank(method="first", ascending=False).astype(int)
        )
    else:
        combined["experimental_priority_rank"] = combined["rank"]

    return combined, missed_feature_df, fi


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
    cluster_col = "nearest_positive_cluster" if "nearest_positive_cluster" in top.columns else None
    if cluster_col:
        for cluster_id, sub in top.groupby(cluster_col):
            recs = [SeqRecord(Seq(seqs[cid]), id=cid, description="") for cid in sub["candidate_id"] if cid in seqs]
            write_fasta_records(recs, by_cluster_dir / f"cluster_{cluster_id}.fasta")
