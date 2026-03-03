from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def plot_recall_curve(metrics_df: pd.DataFrame, out_path: str | Path) -> None:
    ks = sorted({int(c.split("@")[1]) for c in metrics_df.columns if c.startswith("Recall@")})
    vals = [metrics_df[f"Recall@{k}"].mean() for k in ks]
    plt.figure(figsize=(6, 4))
    plt.plot(ks, vals, marker="o")
    plt.xlabel("K")
    plt.ylabel("Recall@K")
    plt.title("Recall curve")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_rank_distribution(rank_df: pd.DataFrame, out_path: str | Path) -> None:
    hidden = rank_df[rank_df["candidate_id"].str.contains("unl_hidden_", na=False)]["rank"]
    plt.figure(figsize=(6, 4))
    plt.hist(hidden, bins=30)
    plt.xlabel("Rank")
    plt.ylabel("Count")
    plt.title("Hidden positive rank distribution")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_embedding_projection(emb: np.ndarray, labels: list[str], out_path: str | Path) -> None:
    pca = PCA(n_components=2, random_state=42)
    xy = pca.fit_transform(emb)
    plt.figure(figsize=(6, 4))
    uniq = sorted(set(labels))
    for u in uniq:
        idx = [i for i, x in enumerate(labels) if x == u]
        plt.scatter(xy[idx, 0], xy[idx, 1], s=12, label=u, alpha=0.7)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
