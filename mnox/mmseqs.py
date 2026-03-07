from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pandas as pd

from .utils import ensure_dir, run_command


def _params_hash(params: dict) -> str:
    return hashlib.md5(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()


def run_mmseqs_search(
    positives_fasta: str | Path,
    query_fasta: str | Path,
    run_dir: str | Path,
    cfg: dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Run MMseqs2 createdb/search/convertalis and parse top-N hits."""
    run_dir = Path(run_dir)
    mm_dir = ensure_dir(run_dir / "mmseqs")
    tmp_dir = ensure_dir(run_dir / cfg["tmp_dir"])

    qdb = mm_dir / "query_db"
    tdb = mm_dir / "target_db"
    resdb = mm_dir / "result_db"
    out_tsv = mm_dir / "result.tsv"
    marker = mm_dir / "marker.json"

    params = {
        "threads": cfg["threads"],
        "sensitivity": cfg["sensitivity"],
        "max_seqs": cfg["max_seqs"],
        "evalue": cfg["evalue"],
        "top_n": cfg["top_n"],
    }

    if marker.exists() and out_tsv.exists():
        prev = json.loads(marker.read_text(encoding="utf-8"))
        if prev.get("hash") == _params_hash(params):
            logger.info("MMseqs2 outputs already exist with same params; reusing.")
            return parse_mmseqs_tsv(out_tsv, cfg["top_n"])

    run_command(["mmseqs", "createdb", str(query_fasta), str(qdb)], logger)
    run_command(["mmseqs", "createdb", str(positives_fasta), str(tdb)], logger)

    run_command(
        [
            "mmseqs",
            "search",
            str(qdb),
            str(tdb),
            str(resdb),
            str(tmp_dir),
            "-s",
            str(cfg["sensitivity"]),
            "--max-seqs",
            str(cfg["max_seqs"]),
            "-e",
            str(cfg["evalue"]),
            "--threads",
            str(cfg["threads"]),
        ],
        logger,
    )

    fmt = "query,target,fident,alnlen,qcov,tcov,evalue,bits"
    run_command(
        [
            "mmseqs",
            "convertalis",
            str(qdb),
            str(tdb),
            str(resdb),
            str(out_tsv),
            "--format-output",
            fmt,
        ],
        logger,
    )

    marker.write_text(json.dumps({"hash": _params_hash(params), "params": params}), encoding="utf-8")
    return parse_mmseqs_tsv(out_tsv, cfg["top_n"])


def parse_mmseqs_tsv(tsv_path: str | Path, top_n: int = 1) -> pd.DataFrame:
    """Parse mmseqs convertalis TSV and retain top N hits per query."""
    cols = ["query_id", "target_id", "fident", "alnlen", "qcov", "tcov", "evalue", "bits"]
    df = pd.read_csv(tsv_path, sep="\t", names=cols)
    if df.empty:
        return df
    df = df.sort_values(["query_id", "bits"], ascending=[True, False])
    return df.groupby("query_id", as_index=False).head(top_n).reset_index(drop=True)


def make_easy_hit_flags(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create easy-hit flag dataframe from MMseqs hit table."""
    if df.empty:
        return pd.DataFrame(columns=["candidate_id", "mmseqs_easy_hit_flag"])

    best = df.sort_values(["query_id", "bits"], ascending=[True, False]).groupby("query_id", as_index=False).first()
    flag = (
        ((best["fident"] >= cfg["fident_min"]) & (best["qcov"] >= cfg["qcov_min"]))
        | (best["evalue"] <= cfg["evalue_max"])
    )
    return pd.DataFrame({"candidate_id": best["query_id"], "mmseqs_easy_hit_flag": flag.astype(bool)})
