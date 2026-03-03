from __future__ import annotations

import hashlib

import pandas as pd


AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def clean_sequence(seq: str) -> str:
    return "".join([aa for aa in seq.upper() if aa in AA20])


def quality_control(df: pd.DataFrame, min_len: int = 50, max_x_ratio: float = 0.05) -> pd.DataFrame:
    clean = df.copy()
    clean["sequence"] = clean["sequence"].map(clean_sequence)
    clean["sequence_length"] = clean["sequence"].str.len()
    clean = clean[(clean["sequence_length"] >= min_len)]
    clean = deduplicate(clean)
    return clean.reset_index(drop=True)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["seq_hash"] = out["sequence"].map(lambda s: hashlib.md5(s.encode("utf-8")).hexdigest())
    out = out.drop_duplicates(subset=["seq_hash"]).drop(columns=["seq_hash"])
    return out
