from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .external_tools import ExternalToolError, phmmer_search_real, which


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


def hmm_search_dispatch(positives: pd.DataFrame, candidates: pd.DataFrame, baseline_cfg: dict, external_cfg: dict) -> pd.DataFrame:
    mode = external_cfg.get("hmm_mode", "auto")  # auto|real|fallback
    if mode in {"auto", "real"} and which(external_cfg.get("phmmer_bin", "phmmer")):
        try:
            return phmmer_search_real(positives, candidates, phmmer_bin=external_cfg.get("phmmer_bin", "phmmer"))
        except ExternalToolError:
            if mode == "real":
                raise
    return hmm_like_search(positives, candidates, k=int(baseline_cfg.get("hmm_window", 3)))
