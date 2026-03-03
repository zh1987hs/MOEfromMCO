from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_ranking(rank_df: pd.DataFrame, hidden_ids: set[str], topk: list[int]) -> dict:
    rank_df = rank_df.copy()
    rank_df["is_hidden"] = rank_df["candidate_id"].isin(hidden_ids).astype(int)
    out: dict[str, float] = {}
    total_pos = max(1, len(hidden_ids))
    ranks = rank_df.loc[rank_df["is_hidden"] == 1, "rank"].sort_values().tolist()
    out["MRR"] = float(np.mean([1 / r for r in ranks])) if ranks else 0.0
    for k in topk:
        top = rank_df.head(k)
        hit = int(top["is_hidden"].sum())
        out[f"Recall@{k}"] = hit / total_pos
        out[f"Precision@{k}"] = hit / max(1, k)
        out[f"HitRate@{k}"] = 1.0 if hit > 0 else 0.0
        expected = (k / len(rank_df)) * total_pos
        out[f"EF@{k}"] = (hit / max(1, expected))
    return out


def summarize_cv(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    metric_cols = [c for c in df.columns if c != "fold"]
    summary = pd.DataFrame({"metric": metric_cols, "mean": [df[c].mean() for c in metric_cols], "std": [df[c].std(ddof=0) for c in metric_cols]})
    return summary
