from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


def build_cluster_profiles(positives: pd.DataFrame, k: int = 3) -> dict[int, Counter[str]]:
    profiles: dict[int, Counter[str]] = {}
    for c, grp in positives.groupby("cluster"):
        bag: Counter[str] = Counter()
        for seq in grp["sequence"]:
            bag.update(seq[i : i + k] for i in range(max(0, len(seq) - k + 1)))
        profiles[int(c)] = bag
    return profiles


def hmm_like_search(positives: pd.DataFrame, candidates: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    profiles = build_cluster_profiles(positives, k=k)
    rows: list[dict] = []
    for c in candidates.itertuples():
        ckm = Counter(c.sequence[i : i + k] for i in range(max(0, len(c.sequence) - k + 1)))
        best_cluster = -1
        best_score = -1.0
        for cid, prof in profiles.items():
            shared = sum((ckm & prof).values())
            denom = max(1, sum(ckm.values()))
            score = shared / denom
            if score > best_score:
                best_score = score
                best_cluster = cid
        rows.append({"candidate_id": c.id, "best_hmm_cluster": best_cluster, "hmm_like_score": float(np.clip(best_score, 0, 1))})
    return pd.DataFrame(rows)


def rank_by_hmm(hmm_df: pd.DataFrame) -> pd.DataFrame:
    out = hmm_df.sort_values("hmm_like_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out
