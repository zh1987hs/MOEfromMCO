from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
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
    cutoff_rows: list[dict] = []

    for cluster_id, hmm_path in hmm_paths.items():
        tbl = out_dir / f"cluster_{cluster_id}.tbl"
        domtbl = out_dir / f"cluster_{cluster_id}.domtbl"
        run_command(
            [
                "hmmsearch",
                "--tblout",
                str(tbl),
                "--domtblout",
                str(domtbl),
                "-E",
                str(cfg["evalue"]),
                "--cpu",
                str(cfg["threads"]),
                str(hmm_path),
                str(query_fasta),
            ],
            logger,
        )
        rows.extend(parse_hmm_domtbl(domtbl, cluster_id))
        cutoff_rows.append({"cluster": int(cluster_id), **_read_hmm_cutoffs(hmm_path)})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(
            columns=["candidate_id", "hmm_best_cluster", "hmm_best_bitscore", "hmm_best_evalue", "hmm_best_cov"]
        )

    best = df.sort_values(["candidate_id", "bitscore"], ascending=[True, False]).groupby("candidate_id", as_index=False).first()
    best = best.rename(
        columns={
            "cluster": "hmm_best_cluster",
            "bitscore": "hmm_best_bitscore",
            "evalue": "hmm_best_evalue",
            "hmm_cov": "hmm_best_cov",
        }
    )
    best["hmm_best_target"] = best["hmm_best_cluster"].map(lambda x: f"cluster_{int(x)}")
    cutoff_df = pd.DataFrame(cutoff_rows)
    if not cutoff_df.empty:
        best = best.merge(cutoff_df, left_on="hmm_best_cluster", right_on="cluster", how="left").drop(columns=["cluster"])
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


def parse_hmm_domtbl(path: str | Path, cluster_id: int) -> list[dict]:
    """Parse HMMER --domtblout rows with HMM coverage."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 23:
                continue
            target_name = parts[0]
            qlen = float(parts[5])
            full_evalue = float(parts[6])
            full_score = float(parts[7])
            hmm_from = float(parts[15])
            hmm_to = float(parts[16])
            hmm_cov = ((hmm_to - hmm_from + 1.0) / max(qlen, 1.0)) if qlen > 0 else 0.0
            rows.append(
                {
                    "candidate_id": target_name,
                    "cluster": int(cluster_id),
                    "evalue": full_evalue,
                    "bitscore": full_score,
                    "hmm_cov": float(np.clip(hmm_cov, 0.0, 1.0)),
                }
            )
    return rows


def _read_hmm_cutoffs(hmm_path: str | Path) -> dict[str, float]:
    """Read GA/TC cutoffs from HMM file if present."""
    cutoffs: dict[str, float] = {}
    with Path(hmm_path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("GA"):
                parts = line.replace(";", " ").split()
                if len(parts) >= 2:
                    cutoffs["ga_bitscore"] = float(parts[1])
            if line.startswith("TC"):
                parts = line.replace(";", " ").split()
                if len(parts) >= 2:
                    cutoffs["tc_bitscore"] = float(parts[1])
    return cutoffs


def _profile_thresholds(cfg: dict) -> dict[str, dict]:
    if {"loose", "balanced", "strict"}.issubset(cfg.keys()):
        return {k: cfg[k] for k in ["loose", "balanced", "strict"]}
    base = {
        "full_seq_evalue_max": cfg.get("evalue_max", 1e-5),
        "bitscore_min": cfg.get("bitscore_min", 50.0),
        "hmm_cov_min": cfg.get("hmm_cov_min", 0.35),
    }
    return {"loose": dict(base), "balanced": dict(base), "strict": dict(base)}


def make_hmm_easy_flags(best_df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Create easy-hit flag for HMMER hits."""
    if best_df.empty:
        return pd.DataFrame(columns=["candidate_id", "hmm_easy_hit_flag"])

    prof = _profile_thresholds(cfg)
    out = pd.DataFrame({"candidate_id": best_df["candidate_id"]})
    use_curated = bool(cfg.get("use_curated_cutoffs_if_present", True))
    ga = best_df.get("ga_bitscore", pd.Series(np.nan, index=best_df.index))
    tc = best_df.get("tc_bitscore", pd.Series(np.nan, index=best_df.index))
    curated_bits = ga.fillna(tc)

    for name, thr in prof.items():
        bits_min = float(thr.get("bitscore_min", 0.0))
        if use_curated:
            bits_req = curated_bits.fillna(bits_min)
        else:
            bits_req = pd.Series(bits_min, index=best_df.index)
        flag = (
            (best_df["hmm_best_evalue"] <= float(thr.get("full_seq_evalue_max", 1.0)))
            & (best_df["hmm_best_bitscore"] >= bits_req)
            & (best_df.get("hmm_best_cov", pd.Series(0.0, index=best_df.index)) >= float(thr.get("hmm_cov_min", 0.35)))
        )
        out[f"hmm_hit_{name}"] = flag.astype(bool)
        out[f"hmm_strength_{name}"] = np.clip(
            0.45 * (best_df["hmm_best_bitscore"] / np.maximum(bits_req, 1.0))
            + 0.30 * (
                -np.log10(best_df["hmm_best_evalue"].clip(lower=1e-300))
                / max(-np.log10(float(thr.get("full_seq_evalue_max", 1e-300))), 1e-9)
            )
            + 0.25 * (best_df.get("hmm_best_cov", pd.Series(0.0, index=best_df.index)) / max(float(thr.get("hmm_cov_min", 0.35)), 1e-9)),
            0.0,
            1.0,
        )

    profile = cfg.get("profile", "balanced")
    out["hmm_easy_hit_flag"] = out.get(f"hmm_hit_{profile}", out["hmm_hit_balanced"]).astype(bool)
    out["hmm_best_target"] = best_df.get("hmm_best_target")
    out["hmm_best_bitscore"] = best_df.get("hmm_best_bitscore")
    out["hmm_best_evalue"] = best_df.get("hmm_best_evalue")
    out["hmm_best_cov"] = best_df.get("hmm_best_cov")
    return out
