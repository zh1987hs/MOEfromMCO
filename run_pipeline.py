from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd

from mnox.cluster import (
    compute_prototypes_and_medoids,
    load_fixed_clusters,
    pick_best_clustering,
)
from mnox.esm_embed import ESMEmbedder, write_runtime_log
from mnox.eval_cv import run_leave_one_gold_family_out_cv, run_loco_cv
from mnox.features import build_missed_candidate_features
from mnox.hmmer import build_cluster_hmms, make_hmm_easy_flags, run_hmmsearch_for_clusters
from mnox.io_fasta import id_to_seq_dict, read_fasta_records, write_fasta_records
from mnox.mmseqs import make_easy_hit_flags, run_mmseqs_search
from mnox.plots import plot_cv_metrics
from mnox.qc import dedupe_and_filter, qc_summary_dict, save_duplicate_map
from mnox.retrieval import (
    decide_easy_hits,
    export_top_candidates,
    rank_candidates_with_policy,
)
from mnox.training_data import build_learned_training_data
from mnox.utils import (
    check_external_tools,
    check_python_dependencies,
    ensure_dir,
    is_wsl,
    load_config,
    make_run_dir,
    save_json,
    setup_logger,
    write_dataframe,
    write_lines,
)


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    """Coerce mixed-type flag series to boolean safely."""
    if s.dtype == bool:
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float).ne(0)
    t = s.fillna("").astype(str).str.strip().str.lower()
    return t.isin({"1", "true", "t", "yes", "y"})


def _normalize_easy_hit_flags(
    ranked: pd.DataFrame,
    mm_flags: pd.DataFrame,
    hmm_flags: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize easy-hit flags to canonical bool columns without _x/_y leakage.

    Handles cases where ranked already has canonical columns, only suffixed columns,
    or no flag columns at all.
    """

    out = ranked.copy()
    for src, base_col in [
        (mm_flags, "mmseqs_easy_hit_flag"),
        (hmm_flags, "hmm_easy_hit_flag"),
    ]:
        related = [c for c in out.columns if c == base_col or c.startswith(f"{base_col}_")]

        # Merge only when no related column exists yet.
        if not related and base_col in src.columns:
            out = out.merge(src[["candidate_id", base_col]], on="candidate_id", how="left")
            related = [c for c in out.columns if c == base_col or c.startswith(f"{base_col}_")]

        if not related:
            out[base_col] = False
            continue

        combined = pd.Series(False, index=out.index)
        for c in related:
            combined = combined | _coerce_bool_series(out[c])

        out[base_col] = combined.fillna(False).astype(bool)

        # Drop non-canonical suffix variants to avoid downstream confusion.
        drop_cols = [c for c in related if c != base_col]
        if drop_cols:
            out = out.drop(columns=drop_cols)

    return out


def _validate_split_rank_outputs(ranked: pd.DataFrame) -> None:
    """Lightweight runtime sanity checks for merged + missed-only ranking outputs."""
    if ranked.empty or "easy_or_missed" not in ranked.columns:
        return
    missed = ranked[ranked["easy_or_missed"].eq("missed")].copy()
    if missed.empty:
        return
    if "missed_rank" not in missed.columns or "missed_experimental_priority_rank" not in missed.columns:
        raise RuntimeError("Missing missed-only ranking columns in ranked output.")
    mr = missed["missed_rank"].dropna().astype(int)
    mepr = missed["missed_experimental_priority_rank"].dropna().astype(int)
    if len(mr) and (mr.min() != 1 or mr.nunique() != len(mr)):
        raise RuntimeError("missed_rank sanity check failed: expected unique contiguous missed-only ranks.")
    if len(mepr) and (mepr.min() != 1 or mepr.nunique() != len(mepr)):
        raise RuntimeError("missed_experimental_priority_rank sanity check failed.")


def _resolve_config(cfg: dict) -> dict:
    """Backwards-compatible config normalization."""
    if "positives" not in cfg:
        cfg["positives"] = {
            "fasta": cfg["input"]["positives_fasta"],
            "metadata_csv": None,
        }
    if "input" in cfg and "unlabeled_fasta" not in cfg["input"]:
        cfg["input"]["unlabeled_fasta"] = cfg["input"].get("unlabeled_mco_fasta")

    if "positive_clustering" not in cfg:
        old = cfg.get("clustering", {})
        cfg["positive_clustering"] = {
            "mode": "auto",
            "fixed_cluster_csv": None,
            "methods": old.get("methods", ["agglomerative", "kmeans"]),
            "k_min": old.get("k_min", 3),
            "k_max": old.get("k_max", 10),
        }

    if "prototype" not in cfg:
        cfg["prototype"] = {
            "representative": "prototype",
            "use_tier_weights": False,
            "gold_weight": 1.0,
            "silver_weight": 1.0,
        }

    if "evaluation" not in cfg:
        cfg["evaluation"] = {"mode": "loco", "leakage_guard": True}

    if "cv" not in cfg:
        cfg["cv"] = {"background_n": 5000, "k_values": [10, 20, 50, 100]}

    if "easy_hit" not in cfg:
        cfg["easy_hit"] = {}
    e = cfg["easy_hit"]
    e.setdefault("decision_rule", "consensus")
    e.setdefault("profile", "balanced")
    e.setdefault("max_easy_fraction", 0.30)
    e.setdefault("fallback_if_too_many_easy", "tighten")
    e.setdefault("calibrate", True)
    e.setdefault("calibration_objective", "precision_at_easy")
    e.setdefault("target_easy_precision", 0.95)
    e.setdefault("target_easy_fraction_max", e.get("max_easy_fraction", 0.30))
    e.setdefault("negative_source", "background")
    mm_old = e.get("mmseqs", {})
    if not {"loose", "balanced", "strict"}.issubset(mm_old.keys()):
        mm_base = {
            "fident_min": mm_old.get("fident_min", 0.30),
            "qcov_min": mm_old.get("qcov_min", 0.70),
            "tcov_min": mm_old.get("tcov_min", mm_old.get("qcov_min", 0.70)),
            "evalue_max": mm_old.get("evalue_max", 1e-5),
        }
        e["mmseqs"] = {
            "loose": dict(mm_base),
            "balanced": {"fident_min": 0.35, "qcov_min": 0.80, "tcov_min": 0.80, "evalue_max": 1e-10},
            "strict": {"fident_min": 0.40, "qcov_min": 0.80, "tcov_min": 0.80, "evalue_max": 1e-20},
        }
    hmm_old = e.get("hmm", {})
    if not {"loose", "balanced", "strict"}.issubset(hmm_old.keys()):
        base_eval = hmm_old.get("evalue_max", 1e-5)
        base_bits = hmm_old.get("bitscore_min", 50.0)
        base_cov = hmm_old.get("hmm_cov_min", 0.35)
        e["hmm"] = {
            "use_curated_cutoffs_if_present": True,
            "loose": {"full_seq_evalue_max": base_eval, "bitscore_min": base_bits, "hmm_cov_min": base_cov},
            "balanced": {"full_seq_evalue_max": 1e-10, "bitscore_min": 80.0, "hmm_cov_min": 0.35},
            "strict": {"full_seq_evalue_max": 1e-15, "bitscore_min": 100.0, "hmm_cov_min": 0.50},
        }
    e["mmseqs"]["profile"] = e.get("profile", "balanced")
    e["hmm"]["profile"] = e.get("profile", "balanced")
    e["hmm"].setdefault("use_curated_cutoffs_if_present", True)

    if "retrieval" not in cfg:
        cfg["retrieval"] = {}
    r = cfg["retrieval"]
    r.setdefault("scorer", "heuristic")
    r.setdefault("affinity_metric", r.get("distance_metric", "cosine"))
    r.setdefault("easy_hit_policy", "prepend")
    r.setdefault("experimental_view_mode", "merged")
    r.setdefault("cv_view_mode", "both")
    r.setdefault("export_easy_only", True)
    r.setdefault("export_missed_only", True)
    r.setdefault("easy_prepend_cap", r.get("top_n_export", 200))
    r.setdefault("support_topk", 5)
    r.setdefault("density_knn_k", r.get("knn_k", 20))
    r.setdefault("novelty_cap", 0.9)
    r.setdefault(
        "risk",
        {
            "w_generic_density": 0.45,
            "w_generic_low_support": 0.35,
            "w_generic_weak_homology": 0.20,
            "w_fp_generic": 0.40,
            "w_fp_low_support": 0.25,
            "w_fp_novelty_mismatch": 0.20,
            "w_fp_low_affinity": 0.15,
            "flag_generic_boost": 0.10,
            "flag_length_boost": 0.08,
            "flag_close_boost": 0.06,
        },
    )
    r.setdefault(
        "experimental_priority",
        {
            "w_false_positive_risk": 0.20,
            "w_generic_mco_risk": 0.10,
            "easy_hit_bonus": 0.03,
            "high_confidence_bonus": 0.02,
        },
    )
    r.setdefault(
        "heuristic",
        {
            "w_affinity": r.get("w_affinity", 0.75),
            "w_positive_support": 0.15,
            "w_local_density": r.get("w_local_support", 0.08),
            "w_novelty": r.get("w_novelty", 0.02),
        },
    )
    r.setdefault(
        "learned",
        {
            "enabled": True,
            "model_type": "logistic",
            "negative_background_n": 5000,
            "standardize_features": True,
            "use_hard_negatives": True,
        },
    )
    return cfg


def _easy_overlap_summary(
    easy_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize easy-hit overlap globally and by inferred family."""
    out_rows: list[dict[str, object]] = []
    if easy_df.empty:
        return pd.DataFrame(columns=["group", "group_value", "total_candidates", "easy_count", "easy_fraction"])

    target_family = {}
    if {"positive_id", "gold_family"}.issubset(metadata_df.columns):
        target_family = dict(zip(metadata_df["positive_id"], metadata_df["gold_family"]))

    work = easy_df.copy()
    work["support_family"] = work.get("mmseqs_best_target", pd.Series(index=work.index)).map(target_family).fillna("unknown")
    out_rows.append(
        {
            "group": "global",
            "group_value": "all",
            "total_candidates": int(len(work)),
            "easy_count": int(work["easy_hit_flag"].sum()),
            "easy_fraction": float(work["easy_hit_flag"].mean()) if len(work) else 0.0,
            "mmseqs_only_easy": int((work["easy_confidence_class"] == "mmseqs_strict_only").sum()),
            "hmm_only_easy": int((work["easy_confidence_class"] == "hmm_strict_only").sum()),
            "both_easy": int((work["easy_confidence_class"] == "both_support").sum()),
        }
    )
    for fam, sub in work.groupby("support_family"):
        out_rows.append(
            {
                "group": "family",
                "group_value": fam,
                "total_candidates": int(len(sub)),
                "easy_count": int(sub["easy_hit_flag"].sum()),
                "easy_fraction": float(sub["easy_hit_flag"].mean()) if len(sub) else 0.0,
                "mmseqs_only_easy": int((sub["easy_confidence_class"] == "mmseqs_strict_only").sum()),
                "hmm_only_easy": int((sub["easy_confidence_class"] == "hmm_strict_only").sum()),
                "both_easy": int((sub["easy_confidence_class"] == "both_support").sum()),
            }
        )
    return pd.DataFrame(out_rows)


def _load_positive_metadata(path: str | None, kept_positive_ids: list[str]) -> pd.DataFrame:
    """Load metadata CSV or create compatibility fallback metadata."""
    if path is None:
        return pd.DataFrame(
            {
                "positive_id": kept_positive_ids,
                "tier": ["gold"] * len(kept_positive_ids),
                "gold_family": ["FAM00"] * len(kept_positive_ids),
                "seed_gold_id": kept_positive_ids,
            }
        )

    meta = pd.read_csv(path)
    required = {"positive_id", "tier", "gold_family", "seed_gold_id"}
    if not required.issubset(meta.columns):
        raise RuntimeError(f"metadata_csv missing required columns: {required - set(meta.columns)}")

    meta = meta[meta["positive_id"].isin(kept_positive_ids)].copy()
    missing = set(kept_positive_ids) - set(meta["positive_id"].tolist())
    if missing:
        raise RuntimeError(f"Missing metadata for positives IDs (first 10): {list(sorted(missing))[:10]}")
    return meta


def main() -> None:
    cfg = _resolve_config(load_config("config.yaml"))
    run_dir = make_run_dir(cfg["output"]["base_dir"])
    logger = setup_logger(run_dir)

    logger.info("Run directory: %s", run_dir)
    logger.info("Runtime platform: %s", platform.platform())
    if is_wsl():
        logger.info("WSL detected: using Linux toolchain path.")
        logger.info("For large data, keep FASTA/outputs inside WSL filesystem (e.g. ~/projects), not /mnt/c.")

    logger.info("positive_clustering.mode=%s", cfg["positive_clustering"]["mode"])
    logger.info(
        "evaluation.mode=%s leakage_guard=%s",
        cfg["evaluation"]["mode"],
        cfg["evaluation"].get("leakage_guard", True),
    )
    logger.info("retrieval.scorer=%s easy_hit_policy=%s", cfg["retrieval"]["scorer"], cfg["retrieval"]["easy_hit_policy"])
    logger.info(
        "retrieval.experimental_view_mode=%s cv_view_mode=%s",
        cfg["retrieval"]["experimental_view_mode"],
        cfg["retrieval"]["cv_view_mode"],
    )

    check_python_dependencies()
    check_external_tools(cfg["hmmer"]["msa_tool"])

    # Step 1
    pos_raw = read_fasta_records(cfg["positives"]["fasta"])
    unl_raw = read_fasta_records(cfg["input"]["unlabeled_fasta"])

    qc_cfg = cfg["qc"]
    pos_qc = dedupe_and_filter(pos_raw, qc_cfg["min_len"], qc_cfg["max_len"], qc_cfg["check_motif"])
    unl_qc = dedupe_and_filter(unl_raw, qc_cfg["min_len"], qc_cfg["max_len"], qc_cfg["check_motif"])

    metadata_df = _load_positive_metadata(cfg["positives"].get("metadata_csv"), pos_qc.kept_ids)
    logger.info(
        "positive tier counts: gold=%d silver=%d families=%d",
        int((metadata_df["tier"].str.lower() == "gold").sum()),
        int((metadata_df["tier"].str.lower() == "silver").sum()),
        int(metadata_df["gold_family"].nunique()),
    )

    write_fasta_records(pos_qc.kept_records, run_dir / "positives_qc.fasta")
    write_fasta_records(unl_qc.kept_records, run_dir / "unlabeled_qc.fasta")
    write_lines(pos_qc.kept_ids, run_dir / "kept_ids_positives.txt")
    write_lines(unl_qc.kept_ids, run_dir / "kept_ids_unlabeled.txt")
    save_duplicate_map(pos_qc.duplicate_map, run_dir / "positives_duplicate_map.tsv")
    save_duplicate_map(unl_qc.duplicate_map, run_dir / "unlabeled_duplicate_map.tsv")
    save_json(qc_summary_dict(pos_qc, unl_qc), run_dir / "qc_summary.json")
    metadata_df.to_csv(run_dir / "positive_metadata_used.csv", index=False)

    # Step 2
    emb_dir = ensure_dir(run_dir / "embeddings")
    runtime_lines = []
    embedder = ESMEmbedder(
        model_name=cfg["esm"]["model_name"],
        batch_size=cfg["esm"]["batch_size"],
        pooling=cfg["esm"]["pooling"],
        device=cfg["esm"].get("device"),
        cache_dir=emb_dir,
        logger=logger,
    )

    t0 = time.time()
    pos_ids_emb, pos_emb = embedder.embed_records_with_cache(pos_qc.kept_records, "positives", cfg["esm"]["shard_size"])
    runtime_lines.append(f"positives_embedding_seconds={time.time() - t0:.2f}")
    np.savez_compressed(emb_dir / "positives_embeddings.npz", ids=np.array(pos_ids_emb), embeddings=pos_emb)

    t1 = time.time()
    unl_ids_emb, unl_emb = embedder.embed_records_with_cache(unl_qc.kept_records, "unlabeled", cfg["esm"]["shard_size"])
    runtime_lines.append(f"unlabeled_embedding_seconds={time.time() - t1:.2f}")
    np.savez_compressed(emb_dir / "unlabeled_embeddings_full.npz", ids=np.array(unl_ids_emb), embeddings=unl_emb)
    write_runtime_log(run_dir / "embedding_runtime_log.txt", runtime_lines)

    # Step 3 positive clustering/prototypes
    cl_cfg = cfg["positive_clustering"]
    mode = cl_cfg["mode"]
    if mode == "fixed":
        labels = load_fixed_clusters(cl_cfg["fixed_cluster_csv"], pos_ids_emb, metadata_df)
        cl_method, cl_k, cl_score, cl_source = "fixed", int(len(set(labels))), float("nan"), "fixed"
    elif mode == "none":
        labels = np.zeros(len(pos_ids_emb), dtype=int)
        cl_method, cl_k, cl_score, cl_source = "none", 1, float("nan"), "none"
    else:
        cl_out = pick_best_clustering(
            pos_emb,
            cl_cfg["methods"],
            cl_cfg["k_min"],
            cl_cfg["k_max"],
            cfg["random_seed"],
            logger=logger,
        )
        labels = cl_out.labels
        cl_method, cl_k, cl_score, cl_source = cl_out.method, cl_out.k, cl_out.score, "auto"

    proto_cfg = cfg["prototype"]
    cluster_df, prototypes, medoids_df = compute_prototypes_and_medoids(
        pos_ids_emb,
        pos_emb,
        labels,
        metadata_df=metadata_df,
        use_tier_weights=bool(proto_cfg.get("use_tier_weights", False)),
        gold_weight=float(proto_cfg.get("gold_weight", 1.0)),
        silver_weight=float(proto_cfg.get("silver_weight", 1.0)),
    )
    cluster_df = cluster_df.rename(columns={"id": "positive_id"}).merge(metadata_df, on="positive_id", how="left")
    cluster_df["method"] = cl_method
    cluster_df["k"] = cl_k
    cluster_df["silhouette"] = cl_score
    cluster_df["cluster_source"] = cl_source
    cluster_df.to_csv(run_dir / "positive_clusters.csv", index=False)
    np.savez_compressed(run_dir / "positive_prototypes.npz", **{f"cluster_{k}": v for k, v in prototypes.items()})
    medoids_df.to_csv(run_dir / "positive_medoids.csv", index=False)

    # Step 4/5 baseline retrieval
    mm_df = run_mmseqs_search(run_dir / "positives_qc.fasta", run_dir / "unlabeled_qc.fasta", run_dir, cfg["mmseqs"], logger)
    write_dataframe(mm_df, run_dir / "mmseqs_results", logger)
    mm_flags = make_easy_hit_flags(mm_df, cfg["easy_hit"]["mmseqs"])
    mm_flags.to_csv(run_dir / "mmseqs_easy_hit_flags.csv", index=False)

    pos_id_to_rec = {r.id: r for r in pos_qc.kept_records}
    cluster_to_records = {int(cid): [pos_id_to_rec[x] for x in sub["positive_id"].tolist()] for cid, sub in cluster_df.groupby("cluster")}
    hmm_paths = build_cluster_hmms(cluster_to_records, run_dir, cfg["hmmer"]["msa_tool"], cfg["hmmer"]["threads"], logger)
    hmm_best = run_hmmsearch_for_clusters(hmm_paths, run_dir / "unlabeled_qc.fasta", run_dir, cfg["hmmer"], logger)
    write_dataframe(hmm_best, run_dir / "hmm_results", logger)
    hmm_flags = make_hmm_easy_flags(hmm_best, cfg["easy_hit"]["hmm"])
    hmm_flags.to_csv(run_dir / "hmm_easy_hit_flags.csv", index=False)

    # Step 6 easy/missed split
    easy_detail_df, easy_diag = decide_easy_hits(unl_ids_emb, mm_flags, hmm_flags, cfg["easy_hit"])
    easy_ids = easy_detail_df.loc[easy_detail_df["easy_hit_flag"], "candidate_id"].tolist()
    missed_ids = easy_detail_df.loc[~easy_detail_df["easy_hit_flag"], "candidate_id"].tolist()
    write_lines(easy_ids, run_dir / "easy_ids.txt")
    write_lines(missed_ids, run_dir / "missed_ids.txt")
    save_json(
        {
            **easy_diag,
            "single_tool_easy_strength_mean": {
                "mmseqs_strict_only": float(
                    easy_detail_df.loc[easy_detail_df["easy_confidence_class"] == "mmseqs_strict_only", "easy_strength_score"].mean()
                )
                if pd.notna(easy_detail_df.loc[easy_detail_df["easy_confidence_class"] == "mmseqs_strict_only", "easy_strength_score"].mean())
                else 0.0,
                "hmm_strict_only": float(
                    easy_detail_df.loc[easy_detail_df["easy_confidence_class"] == "hmm_strict_only", "easy_strength_score"].mean()
                )
                if pd.notna(easy_detail_df.loc[easy_detail_df["easy_confidence_class"] == "hmm_strict_only", "easy_strength_score"].mean())
                else 0.0,
                "both_support": float(
                    easy_detail_df.loc[easy_detail_df["easy_confidence_class"] == "both_support", "easy_strength_score"].mean()
                )
                if pd.notna(easy_detail_df.loc[easy_detail_df["easy_confidence_class"] == "both_support", "easy_strength_score"].mean())
                else 0.0,
            },
        },
        run_dir / "easy_hit_diagnostics.json",
    )
    _easy_overlap_summary(easy_detail_df, metadata_df).to_csv(run_dir / "easy_hit_overlap_summary.csv", index=False)
    used_profile = str(easy_diag.get("profile_used", cfg["easy_hit"]["profile"]))
    save_json(
        {
            "decision_rule": cfg["easy_hit"]["decision_rule"],
            "profile_requested": cfg["easy_hit"]["profile"],
            "profile_used": used_profile,
            "max_easy_fraction": cfg["easy_hit"]["max_easy_fraction"],
            "fallback_if_too_many_easy": cfg["easy_hit"]["fallback_if_too_many_easy"],
            "mmseqs_thresholds": cfg["easy_hit"]["mmseqs"].get(used_profile, {}),
            "mmseqs_strict_thresholds": cfg["easy_hit"]["mmseqs"].get("strict", {}),
            "hmm_thresholds": cfg["easy_hit"]["hmm"].get(used_profile, {}),
            "hmm_strict_thresholds": cfg["easy_hit"]["hmm"].get("strict", {}),
        },
        run_dir / "easy_hit_thresholds_used.json",
    )

    # Step 7 features + scoring + final merge
    lengths = {r.id: len(r.seq) for r in unl_qc.kept_records}
    feat_res = build_missed_candidate_features(
        missed_ids=missed_ids,
        unlabeled_ids=unl_ids_emb,
        unlabeled_emb=unl_emb,
        positive_ids=pos_ids_emb,
        positive_emb=pos_emb,
        positive_labels=labels,
        lengths=lengths,
        medoids_df=medoids_df,
        metadata_df=metadata_df,
        mm_best=mm_df,
        hmm_best=hmm_best,
        cfg=cfg["retrieval"],
    )

    # discovery stage defaults to heuristic unless explicitly requested
    scorer_mode = cfg["retrieval"].get("scorer", "heuristic")

    train_feature_df = None
    train_labels = None
    if scorer_mode == "learned" and bool(cfg["retrieval"].get("learned", {}).get("enabled", True)):
        try:
            lengths_all = {**{r.id: len(r.seq) for r in pos_qc.kept_records}, **{r.id: len(r.seq) for r in unl_qc.kept_records}}
            t_res = build_learned_training_data(
                positive_ids=pos_ids_emb,
                positive_emb=pos_emb,
                positive_labels=labels,
                medoids_df=medoids_df,
                metadata_df=metadata_df,
                mm_best=mm_df,
                hmm_best=hmm_best,
                lengths=lengths_all,
                retrieval_cfg=cfg["retrieval"],
                random_seed=cfg["random_seed"],
                background_ids=unl_ids_emb,
                background_emb=unl_emb,
            )
            train_feature_df = t_res.features
            train_labels = t_res.labels
            logger.info("Discovery learned scorer training enabled: n_pos=%d n_neg=%d", t_res.n_pos, t_res.n_neg)
        except Exception as exc:
            logger.warning("Discovery learned scorer training failed; fallback to heuristic. reason=%s", exc)
            train_feature_df = None
            train_labels = None

    ranked, feat_df, fi_df = rank_candidates_with_policy(
        candidate_ids=unl_ids_emb,
        easy_ids=easy_ids,
        missed_ids=missed_ids,
        mm_best=mm_df,
        hmm_best=hmm_best,
        mm_flags=mm_flags,
        hmm_flags=hmm_flags,
        feature_result=feat_res,
        retrieval_cfg=cfg["retrieval"],
        scorer_mode=scorer_mode,
        train_feature_df=train_feature_df,
        train_labels=train_labels,
    )

    ranked = _normalize_easy_hit_flags(ranked, mm_flags, hmm_flags)
    ranked = ranked.merge(
        easy_detail_df[
            [
                "candidate_id",
                "easy_hit_flag",
                "easy_confidence_class",
                "easy_reason",
                "easy_strength_score",
                "mmseqs_easy_hit_flag",
                "hmm_easy_hit_flag",
            ]
        ],
        on="candidate_id",
        how="left",
        suffixes=("", "_easy"),
    )
    if "mmseqs_easy_hit_flag_easy" in ranked.columns:
        ranked["mmseqs_easy_hit_flag"] = ranked["mmseqs_easy_hit_flag"] | ranked["mmseqs_easy_hit_flag_easy"].fillna(False).astype(bool)
        ranked = ranked.drop(columns=["mmseqs_easy_hit_flag_easy"])
    if "hmm_easy_hit_flag_easy" in ranked.columns:
        ranked["hmm_easy_hit_flag"] = ranked["hmm_easy_hit_flag"] | ranked["hmm_easy_hit_flag_easy"].fillna(False).astype(bool)
        ranked = ranked.drop(columns=["hmm_easy_hit_flag_easy"])
    ranked["easy_confidence_class"] = ranked["easy_confidence_class"].fillna("not_easy")
    ranked["easy_reason"] = ranked["easy_reason"].fillna("")
    ranked["easy_strength_score"] = pd.to_numeric(ranked["easy_strength_score"], errors="coerce").fillna(0.0)
    _validate_split_rank_outputs(ranked)

    write_dataframe(ranked, run_dir / "ranked_candidates", logger)
    feat_df.to_csv(run_dir / "ranked_candidates_features.csv", index=False)

    if bool(cfg["retrieval"].get("export_missed_only", True)):
        missed_mask = ranked["easy_or_missed"].eq("missed") if "easy_or_missed" in ranked.columns else pd.Series(False, index=ranked.index)
        missed_only = ranked[missed_mask].copy()
        if "missed_experimental_priority_rank" in missed_only.columns:
            missed_only = missed_only.sort_values("missed_experimental_priority_rank", ascending=True)
        elif "missed_rank" in missed_only.columns:
            missed_only = missed_only.sort_values("missed_rank", ascending=True)
        missed_only.to_csv(run_dir / "ranked_candidates_missed_only.csv", index=False)

    if bool(cfg["retrieval"].get("export_easy_only", True)):
        easy_mask = ranked["easy_or_missed"].eq("easy") if "easy_or_missed" in ranked.columns else pd.Series(False, index=ranked.index)
        easy_only = ranked[easy_mask].copy()
        if "easy_rank" in easy_only.columns:
            easy_only = easy_only.sort_values("easy_rank", ascending=True)
        easy_only.to_csv(run_dir / "ranked_candidates_easy_only.csv", index=False)

    experimental_cols = [
        "candidate_id",
        "final_score",
        "experimental_priority_score",
        "rank",
        "experimental_priority_rank",
        "missed_rank",
        "missed_experimental_priority_rank",
        "rank_shift_due_to_easy",
        "is_top_in_missed",
        "is_top_in_merged",
        "why_hidden_by_easy",
        "easy_rank",
        "easy_confidence_class",
        "easy_reason",
        "easy_strength_score",
        "easy_or_missed",
        "scoring_mode",
        "nearest_positive_id",
        "nearest_positive_cluster",
        "nearest_positive_family",
        "embedding_similarity",
        "positive_support_score",
        "local_density_score",
        "novelty_score",
        "false_positive_risk",
        "generic_mco_risk_score",
        "dominant_signal_type",
        "confidence_tier",
        "flags",
        "reason_for_high_rank",
    ]
    exp_view = ranked[[c for c in experimental_cols if c in ranked.columns]].copy()
    exp_view = exp_view.rename(columns={"dominant_signal_type": "main_supporting_signals", "flags": "risk_flags"})
    experimental_view_mode = cfg["retrieval"].get("experimental_view_mode", "merged")
    if experimental_view_mode == "missed_only":
        exp_missed_mask = exp_view["easy_or_missed"].eq("missed") if "easy_or_missed" in exp_view.columns else pd.Series(False, index=exp_view.index)
        exp_view = exp_view[exp_missed_mask].copy()
        if "missed_experimental_priority_rank" in exp_view.columns:
            exp_view = exp_view.sort_values("missed_experimental_priority_rank", ascending=True).reset_index(drop=True)
    else:
        if "experimental_priority_rank" in exp_view.columns:
            exp_view = exp_view.sort_values("experimental_priority_rank", ascending=True).reset_index(drop=True)
    exp_view.to_csv(run_dir / "ranked_candidates_experimental_view.csv", index=False)

    if fi_df is not None and not fi_df.empty:
        fi_df.to_csv(run_dir / "feature_importance.csv", index=False)

    unl_seqs = id_to_seq_dict(run_dir / "unlabeled_qc.fasta")
    export_top_candidates(ranked, unl_seqs, run_dir, cfg["retrieval"]["top_n_export"])

    # Step 8 CV (aligned with discovery flow)
    eval_mode = cfg["evaluation"]["mode"]
    if eval_mode == "leave_one_gold_family_out":
        cv_df, diag = run_leave_one_gold_family_out_cv(
            positive_records=pos_qc.kept_records,
            positive_ids=pos_ids_emb,
            positive_emb=pos_emb,
            positive_labels=labels,
            metadata_df=metadata_df,
            unlabeled_records=unl_qc.kept_records,
            unlabeled_ids=unl_ids_emb,
            unlabeled_emb=unl_emb,
            cfg=cfg,
            run_dir=run_dir,
            logger=logger,
        )
    else:
        cv_df, diag = run_loco_cv(
            positive_records=pos_qc.kept_records,
            positive_ids=pos_ids_emb,
            positive_labels=labels,
            positive_emb=pos_emb,
            metadata_df=metadata_df,
            unlabeled_records=unl_qc.kept_records,
            unlabeled_ids=unl_ids_emb,
            unlabeled_emb=unl_emb,
            cfg=cfg,
            run_dir=run_dir,
            logger=logger,
        )

    cv_view_mode = cfg["retrieval"].get("cv_view_mode", "both")
    if cv_view_mode == "merged":
        eval_view = cv_df["evaluation_view"] if "evaluation_view" in cv_df.columns else pd.Series("overall_merged", index=cv_df.index)
        cv_df = cv_df[eval_view == "overall_merged"].copy()
    elif cv_view_mode == "missed_only":
        eval_view = cv_df["evaluation_view"] if "evaluation_view" in cv_df.columns else pd.Series("", index=cv_df.index)
        cv_df = cv_df[eval_view == "missed_only"].copy()

    required_cv_cols = [
        "evaluation_mode", "evaluation_view", "fold_gold_family", "leakage_guard", "method", "scoring_mode", "easy_hit_policy",
        "mrr", "recall@10", "recall@20", "recall@50", "recall@100", "ef@10", "ef@20", "ef@50", "ef@100",
        "n_easy", "n_missed", "n_train_pos", "n_holdout_pos",
    ]
    for c in required_cv_cols:
        if c not in cv_df.columns:
            cv_df[c] = np.nan
    cv_df.to_csv(run_dir / "cv_summary.csv", index=False)
    save_json(diag, run_dir / "cv_fold_diagnostics.json")
    with (run_dir / "cv_method_config_used.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "evaluation": cfg["evaluation"],
                "positive_clustering": cfg["positive_clustering"],
                "retrieval": cfg["retrieval"],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # simple ablation summary
    ablation = cv_df.groupby(
        ["evaluation_view", "method", "scoring_mode"], as_index=False
    )[[c for c in cv_df.columns if c.startswith("recall@") or c.startswith("ef@") or c == "mrr"]].mean(numeric_only=True)
    ablation.to_csv(run_dir / "ablation_summary.csv", index=False)

    plot_cv_metrics(cv_df, run_dir / "plots")
    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
