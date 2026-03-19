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


def _series_bool(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=bool)
    return df[col].fillna(default).astype(bool)


def _series_float(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _profile_name(easy_cfg: dict[str, Any]) -> str:
    return str(easy_cfg.get("profile", "balanced")).lower()


def _normalize_easy_scores(easy_df: pd.DataFrame) -> pd.DataFrame:
    """Map easy-hit strength onto a 0..1 score range for merged policies."""
    e = easy_df.copy()
    if len(e) == 0:
        return e
    denom = e["easy_score"].max() - e["easy_score"].min() + 1e-9
    e["final_score"] = (e["easy_score"] - e["easy_score"].min()) / denom
    e["scoring_mode"] = "easy_strength"
    e["confidence_tier"] = "high"
    e["reason_for_high_rank"] = "strong_easy_hit"
    e["dominant_signal_type"] = "mmseqs/hmm"
    e["flags"] = ""
    return e


def _merge_easy_and_missed(easy_df: pd.DataFrame, missed_df: pd.DataFrame, retrieval_cfg: dict[str, Any]) -> pd.DataFrame:
    """Merge easy and missed tables using configured policy."""
    policy = retrieval_cfg.get("easy_hit_policy", "prepend")
    if policy == "merge":
        e = _normalize_easy_scores(easy_df)
        return pd.concat([e, missed_df], ignore_index=True, sort=False).sort_values("final_score", ascending=False)

    if policy == "interleave":
        e = _normalize_easy_scores(easy_df).reset_index(drop=True)
        m = missed_df.copy().reset_index(drop=True)
        rows: list[pd.Series] = []
        max_len = max(len(e), len(m))
        for i in range(max_len):
            if i < len(e):
                rows.append(e.iloc[i])
            if i < len(m):
                rows.append(m.iloc[i])
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=pd.Index([]))

    if policy == "prepend_with_cap":
        cap = int(retrieval_cfg.get("easy_prepend_cap", retrieval_cfg.get("top_n_export", 200)))
        e = easy_df.copy()
        head = e.head(cap)
        tail = e.iloc[cap:]
        return pd.concat([head, missed_df, tail], ignore_index=True, sort=False)

    return pd.concat([easy_df, missed_df], ignore_index=True, sort=False)


def _add_split_view_columns(combined: pd.DataFrame, easy_df: pd.DataFrame, missed_df: pd.DataFrame, retrieval_cfg: dict[str, Any]) -> pd.DataFrame:
    """Add missed/easy-local ranks and experimental triage diagnostics."""
    out = combined.copy()

    easy_rank_map: dict[str, int] = {}
    if not easy_df.empty:
        e = easy_df.copy().reset_index(drop=True)
        e["easy_rank"] = np.arange(1, len(e) + 1)
        easy_rank_map = dict(zip(e["candidate_id"], e["easy_rank"]))

    missed_rank_map: dict[str, int] = {}
    missed_exp_map: dict[str, int] = {}
    if not missed_df.empty:
        m = missed_df.copy().reset_index(drop=True)
        m["missed_rank"] = np.arange(1, len(m) + 1)
        if "experimental_priority_score" in m.columns:
            m["missed_experimental_priority_rank"] = (
                m["experimental_priority_score"].rank(method="first", ascending=False).astype(int)
            )
        else:
            m["missed_experimental_priority_rank"] = m["missed_rank"]
        missed_rank_map = dict(zip(m["candidate_id"], m["missed_rank"]))
        missed_exp_map = dict(zip(m["candidate_id"], m["missed_experimental_priority_rank"]))

    out["easy_rank"] = out["candidate_id"].map(easy_rank_map)
    out["missed_rank"] = out["candidate_id"].map(missed_rank_map)
    out["missed_experimental_priority_rank"] = out["candidate_id"].map(missed_exp_map)
    easy_or_missed = out["easy_or_missed"] if "easy_or_missed" in out.columns else pd.Series("", index=out.index)

    out["rank_shift_due_to_easy"] = np.where(
        easy_or_missed.eq("missed"),
        out["experimental_priority_rank"].fillna(out["rank"]) - out["missed_experimental_priority_rank"].fillna(np.nan),
        0,
    )
    top_k = int(retrieval_cfg.get("top_n_export", 200))
    out["is_top_in_missed"] = out["missed_experimental_priority_rank"].fillna(np.inf).le(top_k)
    out["is_top_in_merged"] = out["experimental_priority_rank"].fillna(np.inf).le(top_k)
    out["why_hidden_by_easy"] = np.where(
        easy_or_missed.eq("missed") & out["is_top_in_missed"] & ~out["is_top_in_merged"],
        (
            "top_missed_but_hidden_by_easy:"
            + out["rank_shift_due_to_easy"].fillna(0).astype(int).astype(str)
            + "_positions"
        ),
        "",
    )
    return out


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


def decide_easy_hits(
    candidate_ids: list[str],
    mm_flags: pd.DataFrame,
    hmm_flags: pd.DataFrame,
    easy_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Decide easy hits with rule/profile/fraction guard while preserving legacy mode."""
    base = pd.DataFrame({"candidate_id": candidate_ids})
    base = base.merge(mm_flags, on="candidate_id", how="left")
    base = base.merge(hmm_flags, on="candidate_id", how="left")

    profile = _profile_name(easy_cfg)
    rule = str(easy_cfg.get("decision_rule", "or")).lower()
    max_easy_fraction = float(easy_cfg.get("max_easy_fraction", 1.0))
    fallback = str(easy_cfg.get("fallback_if_too_many_easy", "warn_only")).lower()
    calibrate = bool(easy_cfg.get("calibrate", False)) or rule == "calibrated"
    ordered_profiles = ["loose", "balanced", "strict"]

    def apply_rule(profile_name: str) -> pd.Series:
        mm_prof = _series_bool(base, f"mmseqs_hit_{profile_name}", False)
        hmm_prof = _series_bool(base, f"hmm_hit_{profile_name}", False)
        mm_strict = _series_bool(base, "mmseqs_hit_strict", False)
        hmm_strict = _series_bool(base, "hmm_hit_strict", False)
        if rule == "and":
            return mm_prof & hmm_prof
        if rule in {"consensus", "calibrated"}:
            return (mm_prof & hmm_prof) | (mm_strict & ~hmm_prof) | (hmm_strict & ~mm_prof)
        return mm_prof | hmm_prof

    candidate_profiles = ordered_profiles[ordered_profiles.index(profile):] if profile in ordered_profiles else [profile]
    selected_profile = profile if profile in ordered_profiles else "balanced"
    easy = apply_rule(selected_profile)

    if calibrate or easy.mean() > max_easy_fraction:
        for prof in candidate_profiles:
            trial = apply_rule(prof)
            if trial.mean() <= float(easy_cfg.get("target_easy_fraction_max", max_easy_fraction)):
                selected_profile = prof
                easy = trial
                break
        else:
            if fallback == "tighten":
                selected_profile = "strict"
                easy = apply_rule("strict")

    if easy.mean() > max_easy_fraction and fallback == "demote_to_missed":
        strength = np.maximum(
            _series_float(base, f"mmseqs_strength_{selected_profile}", 0.0),
            _series_float(base, f"hmm_strength_{selected_profile}", 0.0),
        )
        keep_n = int(np.floor(max_easy_fraction * len(base)))
        keep_n = max(0, min(keep_n, int(easy.sum())))
        if keep_n < int(easy.sum()):
            keep_ids = base.loc[easy].assign(_strength=strength[easy].values).sort_values("_strength", ascending=False).head(keep_n)["candidate_id"]
            easy = base["candidate_id"].isin(keep_ids)

    mm_prof = _series_bool(base, f"mmseqs_hit_{selected_profile}", False)
    hmm_prof = _series_bool(base, f"hmm_hit_{selected_profile}", False)
    mm_strict = _series_bool(base, "mmseqs_hit_strict", False)
    hmm_strict = _series_bool(base, "hmm_hit_strict", False)

    base["mmseqs_easy_hit_flag"] = mm_prof
    base["hmm_easy_hit_flag"] = hmm_prof
    base["easy_hit_flag"] = easy.astype(bool)
    base["easy_confidence_class"] = np.select(
        [mm_prof & hmm_prof, mm_strict & ~hmm_prof, hmm_strict & ~mm_prof],
        ["both_support", "mmseqs_strict_only", "hmm_strict_only"],
        default="not_easy",
    )
    base["easy_reason"] = np.select(
        [mm_prof & hmm_prof, mm_strict & ~hmm_prof, hmm_strict & ~mm_prof],
        [
            f"{rule}:{selected_profile}:both",
            f"{rule}:strict:mmseqs_only",
            f"{rule}:strict:hmm_only",
        ],
        default=f"{rule}:{selected_profile}:missed",
    )
    base["easy_strength_score"] = np.maximum(
        _series_float(base, f"mmseqs_strength_{selected_profile}", 0.0),
        _series_float(base, f"hmm_strength_{selected_profile}", 0.0),
    )

    both_easy = int((easy & mm_prof & hmm_prof).sum())
    mm_only_easy = int((easy & mm_strict & ~hmm_prof).sum())
    hmm_only_easy = int((easy & hmm_strict & ~mm_prof).sum())
    diagnostics = {
        "decision_rule": rule,
        "profile_requested": profile,
        "profile_used": selected_profile,
        "total_candidates": int(len(base)),
        "easy_count": int(easy.sum()),
        "missed_count": int((~easy).sum()),
        "easy_fraction": float(easy.mean()) if len(base) else 0.0,
        "mmseqs_only_easy": mm_only_easy,
        "hmm_only_easy": hmm_only_easy,
        "both_easy": both_easy,
        "exceeds_max_easy_fraction": bool((float(easy.mean()) if len(base) else 0.0) > max_easy_fraction),
        "fallback_if_too_many_easy": fallback,
        "calibrate": calibrate,
    }
    return base, diagnostics


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

    combined = _merge_easy_and_missed(easy_df, missed_df, retrieval_cfg)

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
    easy_bonus = float(pcfg.get("easy_hit_bonus", 0.03))
    high_conf_bonus = float(pcfg.get("high_confidence_bonus", 0.02))
    if "final_score" in combined.columns:
        exp_priority = combined["final_score"].fillna(0.0).astype(float)
        if "false_positive_risk" in combined.columns:
            exp_priority = exp_priority - w_fp * combined["false_positive_risk"].fillna(0.0).astype(float)
        if "generic_mco_risk_score" in combined.columns:
            exp_priority = exp_priority - w_gr * combined["generic_mco_risk_score"].fillna(0.0).astype(float)
        if "easy_or_missed" in combined.columns:
            exp_priority = exp_priority + easy_bonus * combined["easy_or_missed"].fillna("").eq("easy").astype(float)
        if "confidence_tier" in combined.columns:
            exp_priority = exp_priority + high_conf_bonus * combined["confidence_tier"].fillna("").eq("high").astype(float)
        exp_priority = pd.Series(exp_priority, index=combined.index)
        pmin, pmax = float(exp_priority.min()), float(exp_priority.max())
        if pmax - pmin > 1e-12:
            exp_priority = (exp_priority - pmin) / (pmax - pmin)
        else:
            exp_priority = pd.Series(np.clip(exp_priority, 0.0, 1.0), index=combined.index)
        combined["experimental_priority_score"] = exp_priority
        combined["experimental_priority_rank"] = (
            combined["experimental_priority_score"].rank(method="first", ascending=False).astype(int)
        )
    else:
        combined["experimental_priority_score"] = np.nan
        combined["experimental_priority_rank"] = combined["rank"]

    combined = _add_split_view_columns(combined, easy_df, missed_df, retrieval_cfg)
    return combined, missed_feature_df, fi


def export_top_candidates(
    ranked_df: pd.DataFrame,
    seqs: dict[str, str],
    out_dir: str | Path,
    top_n: int,
) -> None:
    """Export top candidates FASTA globally and by nearest cluster."""
    out_dir = Path(out_dir)
    if "experimental_priority_rank" in ranked_df.columns:
        top = ranked_df.sort_values("experimental_priority_rank", ascending=True).head(top_n)
    elif "rank" in ranked_df.columns:
        top = ranked_df.sort_values("rank", ascending=True).head(top_n)
    else:
        top = ranked_df.sort_values("final_score", ascending=False).head(top_n)

    records = [SeqRecord(Seq(seqs[cid]), id=cid, description="") for cid in top["candidate_id"] if cid in seqs]
    write_fasta_records(records, out_dir / "top_candidates.fasta")
    top.to_csv(out_dir / "top_candidates_experimental.csv", index=False)

    by_cluster_dir = out_dir / "top_candidates_by_cluster"
    by_cluster_dir.mkdir(parents=True, exist_ok=True)
    cluster_col = "nearest_positive_cluster" if "nearest_positive_cluster" in top.columns else None
    if cluster_col:
        for cluster_id, sub in top.groupby(cluster_col):
            recs = [SeqRecord(Seq(seqs[cid]), id=cid, description="") for cid in sub["candidate_id"] if cid in seqs]
            write_fasta_records(recs, by_cluster_dir / f"cluster_{cluster_id}.fasta")
