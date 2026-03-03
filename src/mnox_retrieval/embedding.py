from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import HashingVectorizer

AA = "ACDEFGHIKLMNPQRSTVWY"
HYDRO = set("AVILMFWY")
AROM = set("FWYH")
BASIC = set("KRH")
ACIDIC = set("DE")


def _fallback_features(seq: str) -> np.ndarray:
    length = max(1, len(seq))
    aa_freq = np.array([seq.count(a) / length for a in AA], dtype=float)
    hydro = sum(a in HYDRO for a in seq) / length
    arom = sum(a in AROM for a in seq) / length
    basic = sum(a in BASIC for a in seq) / length
    acidic = sum(a in ACIDIC for a in seq) / length
    motifs = np.array([float(m in seq) for m in ["DDE", "EEDD", "DEDE", "HWH", "HCH"]], dtype=float)
    windows = []
    win = 25
    for i in range(0, len(seq), win):
        chunk = seq[i : i + win]
        if chunk:
            windows.append(sum(a in ACIDIC for a in chunk) / len(chunk))
    stats = np.array([np.mean(windows) if windows else 0.0, np.std(windows) if windows else 0.0])
    return np.concatenate([aa_freq, np.array([hydro, arom, basic, acidic]), motifs, stats])


def compute_embeddings(
    df: pd.DataFrame,
    mode: str = "auto",
    dim: int = 128,
    esm_model_name: str | None = None,
    esm_model_path: str | None = None,
) -> np.ndarray:
    if mode in {"esm", "auto"}:
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            model_ref = esm_model_path or esm_model_name
            if model_ref is None:
                raise ValueError("ESM mode requested but no model provided")
            tok = AutoTokenizer.from_pretrained(model_ref)
            mdl = AutoModel.from_pretrained(model_ref)
            mdl.eval()
            embeds = []
            for seq in df["sequence"]:
                with torch.no_grad():
                    t = tok(seq, return_tensors="pt", truncation=True)
                    out = mdl(**t).last_hidden_state.mean(dim=1).squeeze(0).cpu().numpy()
                embeds.append(out)
            arr = np.vstack(embeds)
            if arr.shape[1] > dim:
                arr = PCA(n_components=dim, random_state=42).fit_transform(arr)
            return arr
        except Exception:
            if mode == "esm":
                raise
    feats = np.vstack([_fallback_features(s) for s in df["sequence"]])
    hv = HashingVectorizer(analyzer="char", ngram_range=(2, 3), n_features=512, alternate_sign=False, norm=None)
    text = [" ".join([s[i : i + 3] for i in range(max(0, len(s) - 2))]) for s in df["sequence"]]
    hashed = hv.transform(text).toarray()
    concat = np.hstack([feats, hashed])
    d = min(dim, concat.shape[1], len(df))
    emb = PCA(n_components=max(2, d), random_state=42).fit_transform(concat)
    return emb
