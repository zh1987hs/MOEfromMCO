from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from Bio.SeqRecord import SeqRecord

from .io_fasta import write_fasta_records
from .utils import ensure_dir, run_command


def build_cluster_hmms(
    cluster_to_records: dict[int, list[SeqRecord]],
    run_dir: str | Path,
    msa_tool: str,
    threads: int,
    logger: logging.Logger,
) -> dict[int, Path]:
    """Build HMM profile per positive cluster via MSA + hmmbuild."""
    hmm_dir = ensure_dir(Path(run_dir) / "hmm")
    hmm_paths: dict[int, Path] = {}

    for cluster_id, records in cluster_to_records.items():
        cluster_dir = ensure_dir(hmm_dir / f"cluster_{cluster_id}")
        in_fasta = cluster_dir / "cluster.fasta"
        aln_fasta = cluster_dir / "cluster.aln.fasta"
        hmm_path = cluster_dir / "cluster.hmm"

        write_fasta_records(records, in_fasta)

        if msa_tool == "mafft":
            cmd = ["mafft", "--thread", str(threads), str(in_fasta)]
            logger.info("RUN: %s", " ".join(cmd))
            import subprocess

            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(f"MAFFT failed for cluster {cluster_id}: {proc.stderr}")
            aln_fasta.write_text(proc.stdout, encoding="utf-8")
        else:
            run_command(
                ["muscle", "-align", str(in_fasta), "-output", str(aln_fasta)], logger
            )

        run_command(["hmmbuild", str(hmm_path), str(aln_fasta)], logger)
        hmm_paths[cluster_id] = hmm_path

    return hmm_paths


def run_hmmsearch_for_clusters(
    hmm_paths: dict[int, Path],
    query_fasta: str | Path,
    run_dir: str | Path,
    cfg: dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    """Run hmmsearch per cluster and aggregate best hit per query."""
    out_dir = ensure_dir(Path(run_dir) / "hmmsearch")
    rows: list[dict] = []

    for cluster_id, hmm_path in hmm_paths.items():
        tbl = out_dir / f"cluster_{cluster_id}.tbl"
        run_command(
            [
                "hmmsearch",
                "--tblout",
                str(tbl),
                "-E",
                str(cfg["evalue"]),
                "--cpu",
                str(cfg["threads"]),
                str(hmm_path),
                str(query_fasta),
            ],
            logger,
        )
        rows.extend(parse_hmm_tbl(tbl, cluster_id))

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["candidate_id", "hmm_best_cluster", "hmm_best_bitscore", "hmm_best_evalue"]
        )

    best = df.sort_values(["candidate_id", "bitscore"], ascending=[True, False]).groupby("candidate_id", as_index=False).first()
    best = best.rename(
        columns={
            "cluster": "hmm_best_cluster",
            "bitscore": "hmm_best_bitscore",
            "evalue": "hmm_best_evalue",
        }
    )
    best["hmm_best_target"] = best["hmm_best_cluster"].map(lambda x: f"cluster_{int(x)}")
    return best


def parse_hmm_tbl(path: str | Path, cluster_id: int) -> list[dict]:
    """Parse HMMER --tblout format rows."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            target_name = parts[0]
            evalue = float(parts[4])
            bits = float(parts[5])
            rows.append(
                {
                    "candidate_id": target_name,
                    "cluster": int(cluster_id),
                    "evalue": evalue,
                    "bitscore": bits,
                }
            )
    return rows


def make_hmm_easy_flags(best_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create easy-hit flag for HMMER hits."""
    if best_df.empty:
        return pd.DataFrame(columns=["candidate_id", "hmm_easy_hit_flag"])

    flag = (best_df["hmm_best_evalue"] <= cfg["evalue_max"]) | (
        best_df["hmm_best_bitscore"] >= cfg["bitscore_min"]
    )
    return pd.DataFrame(
        {"candidate_id": best_df["candidate_id"], "hmm_easy_hit_flag": flag.astype(bool)}
    )
