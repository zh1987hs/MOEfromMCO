from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .clustering import build_prototypes, cluster_positives_embedding, cluster_positives_sequence
from .config import Config
from .embedding import compute_embeddings
from .evaluation import evaluate_ranking, summarize_cv
from .external_tools import detect_tools
from .io_utils import read_fasta, save_json, save_table, write_fasta
from .plots import plot_embedding_projection, plot_rank_distribution, plot_recall_curve
from .qc import quality_control
from .retrieval import run_retrieval
from .simulate import simulate_dataset


def load_or_simulate(sim_dir: Path, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    p_csv = sim_dir / "positives.csv"
    u_csv = sim_dir / "unlabeled.csv"
    if p_csv.exists() and u_csv.exists():
        return pd.read_csv(p_csv), pd.read_csv(u_csv)
    sim = cfg.get("simulation", default={})
    res = simulate_dataset(sim_dir, **sim, seed=cfg.get("seed", default=42))
    return res.positives, res.unlabeled


def _cluster_mode(cfg: Config, pos_df: pd.DataFrame, emb_pos: np.ndarray) -> np.ndarray:
    mode = cfg.get("clustering", "mode", default="embedding")
    n_clusters = cfg.get("clustering", "n_clusters", default=5)
    if mode == "sequence":
        return cluster_positives_sequence(pos_df, n_clusters=n_clusters, seed=cfg.get("seed", default=42))
    return cluster_positives_embedding(emb_pos, n_clusters=n_clusters, seed=cfg.get("seed", default=42))


def run_once(positives: pd.DataFrame, unlabeled: pd.DataFrame, cfg: Config, out_dir: Path) -> tuple[pd.DataFrame, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    positives = quality_control(positives)
    unlabeled = quality_control(unlabeled)

    emb_cfg = cfg.get("embedding", default={})
    emb_pos = compute_embeddings(positives, **emb_cfg)
    emb_unl = compute_embeddings(unlabeled, **emb_cfg)

    labels = _cluster_mode(cfg, positives, emb_pos)
    positives = positives.copy()
    positives["cluster"] = labels
    protos = build_prototypes(positives, emb_pos, labels)

    rank_df = run_retrieval(
        positives,
        positives,
        unlabeled,
        emb_pos,
        emb_unl,
        protos,
        cfg.get("scoring", default={}),
        cfg.get("baseline", default={}),
        cfg.get("external", default={}),
        cfg.get("scoring", "local_support_neighbors", default=15),
    )
    hidden_ids = set(unlabeled.loc[unlabeled.get("is_hidden_positive", 0) == 1, "id"].tolist()) if "is_hidden_positive" in unlabeled.columns else set()
    metrics = evaluate_ranking(rank_df, hidden_ids, cfg.get("evaluation", "topk", default=[10, 20, 50, 100]))

    save_table(rank_df, out_dir / "ranked_candidates.csv")
    save_json(metrics, out_dir / "evaluation_summary.json")
    save_json(detect_tools(), out_dir / "tool_detection.json")
    cfg.dump(out_dir / "config_used.yaml")

    top_records = rank_df.head(100)[["candidate_id"]].merge(unlabeled[["id", "sequence"]], left_on="candidate_id", right_on="id")
    write_fasta([(r["candidate_id"], r["sequence"]) for _, r in top_records.iterrows()], out_dir / "top_candidates.fasta")

    plot_rank_distribution(rank_df, out_dir / "rank_distribution.png")
    comb_labels = ["positive"] * len(positives) + ["unlabeled"] * len(unlabeled)
    plot_embedding_projection(np.vstack([emb_pos, emb_unl]), comb_labels, out_dir / "embedding_pca.png")
    return rank_df, metrics


def cross_validate(positives: pd.DataFrame, unlabeled: pd.DataFrame, cfg: Config, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    emb_cfg = cfg.get("embedding", default={})
    emb_pos_all = compute_embeddings(positives, **emb_cfg)
    labels = _cluster_mode(cfg, positives, emb_pos_all)
    positives = positives.copy()
    positives["cluster"] = labels

    fold_metrics: list[dict] = []
    fold_rankings: list[pd.DataFrame] = []
    for fold in sorted(positives["cluster"].unique()):
        train = positives[positives["cluster"] != fold].reset_index(drop=True)
        hold = positives[positives["cluster"] == fold].reset_index(drop=True)
        mixed = pd.concat([unlabeled.assign(is_hidden_positive=0), hold[["id", "sequence"]].assign(is_hidden_positive=1)], ignore_index=True)

        emb_train = compute_embeddings(train, **emb_cfg)
        emb_mixed = compute_embeddings(mixed, **emb_cfg)
        tr_labels = train["cluster"].to_numpy()
        protos = build_prototypes(train, emb_train, tr_labels)

        rank_df = run_retrieval(
            train,
            positives,
            mixed,
            emb_train,
            emb_mixed,
            protos,
            cfg.get("scoring", default={}),
            cfg.get("baseline", default={}),
            cfg.get("external", default={}),
            cfg.get("scoring", "local_support_neighbors", default=15),
        )
        hidden_ids = set(hold["id"].tolist())
        met = evaluate_ranking(rank_df, hidden_ids, cfg.get("evaluation", "topk", default=[10, 20, 50, 100]))
        met["fold"] = int(fold)
        fold_metrics.append(met)
        rank_df["fold"] = int(fold)
        fold_rankings.append(rank_df)

    mdf = pd.DataFrame(fold_metrics)
    sdf = summarize_cv(fold_metrics)
    all_rank = pd.concat(fold_rankings, ignore_index=True)
    save_table(mdf, out_dir / "cv_metrics_by_fold.csv")
    save_table(sdf, out_dir / "cv_summary.csv")
    save_table(all_rank, out_dir / "cv_rankings.csv")
    plot_recall_curve(mdf, out_dir / "cv_recall_curve.png")
    return mdf, sdf


def load_real_fasta(positive_fasta: str, unlabeled_fasta: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    pos = pd.DataFrame([{"id": rid, "sequence": seq} for rid, seq in read_fasta(positive_fasta)])
    unl = pd.DataFrame([{"id": rid, "sequence": seq} for rid, seq in read_fasta(unlabeled_fasta)])
    return pos, unl
