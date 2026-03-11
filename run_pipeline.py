from __future__ import annotations

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
from mnox.hmmer import build_cluster_hmms, make_hmm_easy_flags, run_hmmsearch_for_clusters
from mnox.io_fasta import id_to_seq_dict, read_fasta_records, write_fasta_records
from mnox.mmseqs import make_easy_hit_flags, run_mmseqs_search
from mnox.plots import plot_cv_metrics
from mnox.qc import dedupe_and_filter, qc_summary_dict, save_duplicate_map
from mnox.retrieval import build_easy_and_missed_sets, export_top_candidates, rank_missed_candidates
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

    return cfg


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
    logger.info("evaluation.mode=%s leakage_guard=%s", cfg["evaluation"]["mode"], cfg["evaluation"].get("leakage_guard", True))

    # Step 0
    check_python_dependencies()
    check_external_tools(cfg["hmmer"]["msa_tool"])

    # Step 1
    pos_raw = read_fasta_records(cfg["positives"]["fasta"])
    unl_raw = read_fasta_records(cfg["input"]["unlabeled_fasta"])

    qc_cfg = cfg["qc"]
    pos_qc = dedupe_and_filter(
        pos_raw,
        min_len=qc_cfg["min_len"],
        max_len=qc_cfg["max_len"],
        check_motif=qc_cfg["check_motif"],
    )
    unl_qc = dedupe_and_filter(
        unl_raw,
        min_len=qc_cfg["min_len"],
        max_len=qc_cfg["max_len"],
        check_motif=qc_cfg["check_motif"],
    )

    metadata_df = _load_positive_metadata(cfg["positives"].get("metadata_csv"), pos_qc.kept_ids)
    gold_n = int((metadata_df["tier"].str.lower() == "gold").sum())
    silver_n = int((metadata_df["tier"].str.lower() == "silver").sum())
    family_n = int(metadata_df["gold_family"].nunique())
    logger.info("positive tier counts: gold=%d silver=%d families=%d", gold_n, silver_n, family_n)

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
    pos_ids_emb, pos_emb = embedder.embed_records_with_cache(
        pos_qc.kept_records, "positives", cfg["esm"]["shard_size"]
    )
    runtime_lines.append(f"positives_embedding_seconds={time.time() - t0:.2f}")
    np.savez_compressed(emb_dir / "positives_embeddings.npz", ids=np.array(pos_ids_emb), embeddings=pos_emb)

    t1 = time.time()
    unl_ids_emb, unl_emb = embedder.embed_records_with_cache(
        unl_qc.kept_records, "unlabeled", cfg["esm"]["shard_size"]
    )
    runtime_lines.append(f"unlabeled_embedding_seconds={time.time() - t1:.2f}")
    np.savez_compressed(emb_dir / "unlabeled_embeddings_full.npz", ids=np.array(unl_ids_emb), embeddings=unl_emb)
    write_runtime_log(run_dir / "embedding_runtime_log.txt", runtime_lines)

    # Step 3
    cl_cfg = cfg["positive_clustering"]
    mode = cl_cfg["mode"]
    if mode == "fixed":
        if not cl_cfg.get("fixed_cluster_csv"):
            raise RuntimeError("positive_clustering.mode=fixed requires fixed_cluster_csv")
        labels = load_fixed_clusters(cl_cfg["fixed_cluster_csv"], pos_ids_emb, metadata_df)
        cl_method, cl_k, cl_score, cl_source = "fixed", int(len(set(labels))), float("nan"), "fixed"
    elif mode == "none":
        labels = np.zeros(len(pos_ids_emb), dtype=int)
        cl_method, cl_k, cl_score, cl_source = "none", 1, float("nan"), "none"
    else:
        cl_out = pick_best_clustering(
            pos_emb,
            methods=cl_cfg["methods"],
            k_min=cl_cfg["k_min"],
            k_max=cl_cfg["k_max"],
            random_state=cfg["random_seed"],
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
    cluster_df = cluster_df.rename(columns={"id": "positive_id"})
    cluster_df = cluster_df.merge(metadata_df, on="positive_id", how="left")
    cluster_df["method"] = cl_method
    cluster_df["k"] = cl_k
    cluster_df["silhouette"] = cl_score
    cluster_df["cluster_source"] = cl_source
    cluster_df.to_csv(run_dir / "positive_clusters.csv", index=False)
    np.savez_compressed(run_dir / "positive_prototypes.npz", **{f"cluster_{k}": v for k, v in prototypes.items()})
    medoids_df.to_csv(run_dir / "positive_medoids.csv", index=False)

    # Step 4
    mm_df = run_mmseqs_search(
        run_dir / "positives_qc.fasta",
        run_dir / "unlabeled_qc.fasta",
        run_dir,
        cfg["mmseqs"],
        logger,
    )
    mm_path = write_dataframe(mm_df, run_dir / "mmseqs_results", logger)
    logger.info("MMseqs results -> %s", mm_path)
    mm_flags = make_easy_hit_flags(mm_df, cfg["easy_hit"]["mmseqs"])
    mm_flags.to_csv(run_dir / "mmseqs_easy_hit_flags.csv", index=False)

    # Step 5
    pos_id_to_rec = {r.id: r for r in pos_qc.kept_records}
    cluster_to_records = {
        int(cid): [pos_id_to_rec[x] for x in sub["positive_id"].tolist()]
        for cid, sub in cluster_df.groupby("cluster")
    }
    hmm_paths = build_cluster_hmms(
        cluster_to_records,
        run_dir,
        msa_tool=cfg["hmmer"]["msa_tool"],
        threads=cfg["hmmer"]["threads"],
        logger=logger,
    )
    hmm_best = run_hmmsearch_for_clusters(hmm_paths, run_dir / "unlabeled_qc.fasta", run_dir, cfg["hmmer"], logger)
    hmm_path = write_dataframe(hmm_best, run_dir / "hmm_results", logger)
    logger.info("HMM results -> %s", hmm_path)
    hmm_flags = make_hmm_easy_flags(hmm_best, cfg["easy_hit"]["hmm"])
    hmm_flags.to_csv(run_dir / "hmm_easy_hit_flags.csv", index=False)

    # Step 6
    easy_ids, missed_ids = build_easy_and_missed_sets(unl_ids_emb, mm_flags, hmm_flags)
    write_lines(easy_ids, run_dir / "easy_ids.txt")
    write_lines(missed_ids, run_dir / "missed_ids.txt")

    # Step 7
    lengths = {r.id: len(r.seq) for r in unl_qc.kept_records}
    ranked = rank_missed_candidates(
        missed_ids=missed_ids,
        unlabeled_ids=unl_ids_emb,
        unlabeled_emb=unl_emb,
        lengths=lengths,
        prototypes=prototypes,
        medoids_df=medoids_df,
        mm_best=mm_df,
        hmm_best=hmm_best,
        score_cfg=cfg["retrieval"],
    )
    ranked = ranked.merge(mm_flags, on="candidate_id", how="left")
    ranked = ranked.merge(hmm_flags, on="candidate_id", how="left")
    ranked["mmseqs_easy_hit_flag"] = ranked["mmseqs_easy_hit_flag"].fillna(False).astype(bool)
    ranked["hmm_easy_hit_flag"] = ranked["hmm_easy_hit_flag"].fillna(False).astype(bool)
    rank_path = write_dataframe(ranked, run_dir / "ranked_candidates", logger)
    logger.info("Ranked candidates -> %s", rank_path)

    unl_seqs = id_to_seq_dict(run_dir / "unlabeled_qc.fasta")
    export_top_candidates(ranked, unl_seqs, run_dir, cfg["retrieval"]["top_n_export"])

    # Step 8
    eval_mode = cfg["evaluation"]["mode"]
    if eval_mode == "leave_one_gold_family_out":
        cv_df = run_leave_one_gold_family_out_cv(
            positive_records=pos_qc.kept_records,
            positive_ids=pos_ids_emb,
            positive_emb=pos_emb,
            unlabeled_records=unl_qc.kept_records,
            unlabeled_ids=unl_ids_emb,
            unlabeled_emb=unl_emb,
            metadata_df=metadata_df,
            cfg=cfg,
            run_dir=run_dir,
            logger=logger,
        )
    else:
        cv_df = run_loco_cv(
            positive_records=pos_qc.kept_records,
            positive_ids=pos_ids_emb,
            positive_labels=labels,
            positive_emb=pos_emb,
            unlabeled_records=unl_qc.kept_records,
            unlabeled_ids=unl_ids_emb,
            unlabeled_emb=unl_emb,
            cfg=cfg,
            run_dir=run_dir,
            logger=logger,
        )

    cv_df.to_csv(run_dir / "cv_summary.csv", index=False)
    plot_cv_metrics(cv_df, run_dir / "plots")

    logger.info("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
