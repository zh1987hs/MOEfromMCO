from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd


def _kmers(seq: str, k: int) -> Counter[str]:
    return Counter(seq[i : i + k] for i in range(max(0, len(seq) - k + 1)))


def approx_seq_identity(a: str, b: str) -> tuple[float, float]:
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0.0
    matches = sum(1 for i in range(n) if a[i] == b[i])
    return matches / n, n / max(len(a), len(b))


def blast_like_search(positives: pd.DataFrame, candidates: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    pos_km = {r.id: _kmers(r.sequence, k) for r in positives.itertuples()}
    rows: list[dict] = []
    for c in candidates.itertuples():
        ck = _kmers(c.sequence, k)
        best = {"score": -1.0, "id": None, "identity": 0.0, "coverage": 0.0}
        for p in positives.itertuples():
            shared = sum((ck & pos_km[p.id]).values())
            denom = max(1, sum(ck.values()) + sum(pos_km[p.id].values()))
            jacc = 2.0 * shared / denom
            ident, cov = approx_seq_identity(c.sequence, p.sequence)
            score = 0.55 * jacc + 0.35 * ident + 0.10 * cov
            if score > best["score"]:
                best = {"score": score, "id": p.id, "identity": ident, "coverage": cov}
        rows.append(
            {
                "candidate_id": c.id,
                "best_hit_positive_id": best["id"],
                "approx_identity": best["identity"],
                "approx_coverage": best["coverage"],
                "blast_like_score": float(np.clip(best["score"], 0, 1)),
            }
        )
    return pd.DataFrame(rows)


def rank_by_blast(blast_df: pd.DataFrame) -> pd.DataFrame:
    out = blast_df.sort_values("blast_like_score", ascending=False).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1)
    return out
