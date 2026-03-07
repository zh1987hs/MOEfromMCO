from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .external_tools import ExternalToolError, blastp_search_real, which


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
                "best_hit_positive_id": best["id"] or "NA",
                "approx_identity": best["identity"],
                "approx_coverage": best["coverage"],
                "blast_like_score": float(np.clip(best["score"], 0, 1)),
            }
        )
    return pd.DataFrame(rows)


def blast_search_dispatch(positives: pd.DataFrame, candidates: pd.DataFrame, baseline_cfg: dict, external_cfg: dict) -> pd.DataFrame:
    mode = external_cfg.get("blast_mode", "auto")  # auto|real|fallback
    if mode in {"auto", "real"} and which(external_cfg.get("blastp_bin", "blastp")) and which(external_cfg.get("makeblastdb_bin", "makeblastdb")):
        try:
            return blastp_search_real(
                positives,
                candidates,
                blastp_bin=external_cfg.get("blastp_bin", "blastp"),
                makeblastdb_bin=external_cfg.get("makeblastdb_bin", "makeblastdb"),
                threads=int(external_cfg.get("threads", 1)),
            )
        except ExternalToolError:
            if mode == "real":
                raise
    return blast_like_search(positives, candidates, k=int(baseline_cfg.get("blast_kmer_k", 3)))
