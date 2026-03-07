from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_cv_metrics(cv_df: pd.DataFrame, out_dir: str | Path) -> None:
    """Create Recall@K and MRR plots from CV summary dataframe."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    recall_cols = [c for c in cv_df.columns if c.startswith("recall@")]
    if recall_cols:
        plt.figure(figsize=(8, 5))
        for method, sub in cv_df.groupby("method"):
            means = [sub[c].mean() for c in recall_cols]
            plt.plot(recall_cols, means, marker="o", label=method)
        plt.ylabel("Recall")
        plt.xlabel("K")
        plt.title("LOCO Recall@K")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "recall_at_k.png", dpi=200)
        plt.close()

    if "mrr" in cv_df.columns:
        plt.figure(figsize=(7, 5))
        data = [sub["mrr"].values for _, sub in cv_df.groupby("method")]
        labels = [m for m, _ in cv_df.groupby("method")]
        plt.boxplot(data, labels=labels)
        plt.ylabel("MRR")
        plt.title("LOCO MRR distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "mrr_box.png", dpi=200)
        plt.close()
