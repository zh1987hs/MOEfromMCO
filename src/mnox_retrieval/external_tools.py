from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


class ExternalToolError(RuntimeError):
    """Raised when an external tool invocation fails."""


def which(binary: str) -> str | None:
    return shutil.which(binary)


def run_command(cmd: list[str], cwd: str | Path | None = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise ExternalToolError(
            f"Command failed ({' '.join(cmd)}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )


def blastp_search_real(
    positives: "pd.DataFrame",
    candidates: "pd.DataFrame",
    blastp_bin: str = "blastp",
    makeblastdb_bin: str = "makeblastdb",
    threads: int = 1,
) -> "pd.DataFrame":
    """Run real BLASTP and return standardized columns."""
    import numpy as np
    import pandas as pd

    from .io_utils import write_fasta

    with tempfile.TemporaryDirectory(prefix="mnox_blast_") as td:
        tdir = Path(td)
        pos_fa = tdir / "positives.fasta"
        cand_fa = tdir / "candidates.fasta"
        db_prefix = tdir / "pos_db"
        out_tsv = tdir / "blast.tsv"

        write_fasta([(r.id, r.sequence) for r in positives.itertuples()], pos_fa)
        write_fasta([(r.id, r.sequence) for r in candidates.itertuples()], cand_fa)
        run_command([makeblastdb_bin, "-in", str(pos_fa), "-dbtype", "prot", "-out", str(db_prefix)])
        run_command(
            [
                blastp_bin,
                "-query",
                str(cand_fa),
                "-db",
                str(db_prefix),
                "-outfmt",
                "6 qseqid sseqid pident length qlen slen bitscore",
                "-max_target_seqs",
                "1",
                "-num_threads",
                str(threads),
                "-out",
                str(out_tsv),
            ]
        )
        rows: list[dict] = []
        if out_tsv.exists() and out_tsv.stat().st_size > 0:
            raw = pd.read_csv(
                out_tsv,
                sep="\t",
                header=None,
                names=["qseqid", "sseqid", "pident", "length", "qlen", "slen", "bitscore"],
            )
            best = raw.sort_values("bitscore", ascending=False).drop_duplicates("qseqid")
            bit_max = float(best["bitscore"].max()) if len(best) else 1.0
            for r in best.itertuples():
                cov = float(r.length) / max(1, float(min(r.qlen, r.slen)))
                rows.append(
                    {
                        "candidate_id": r.qseqid,
                        "best_hit_positive_id": r.sseqid,
                        "approx_identity": float(r.pident) / 100.0,
                        "approx_coverage": float(np.clip(cov, 0, 1)),
                        "blast_like_score": float(np.clip(r.bitscore / max(1e-8, bit_max), 0, 1)),
                    }
                )
        result = pd.DataFrame(rows)
    if result.empty:
        result = pd.DataFrame({"candidate_id": candidates["id"]})
    return _fill_missing_blast(result, candidates)


def _fill_missing_blast(df: "pd.DataFrame", candidates: "pd.DataFrame") -> "pd.DataFrame":
    import pandas as pd

    base = pd.DataFrame({"candidate_id": candidates["id"].tolist()})
    merged = base.merge(df, on="candidate_id", how="left")
    merged["best_hit_positive_id"] = merged["best_hit_positive_id"].fillna("NA")
    merged["approx_identity"] = merged["approx_identity"].fillna(0.0)
    merged["approx_coverage"] = merged["approx_coverage"].fillna(0.0)
    merged["blast_like_score"] = merged["blast_like_score"].fillna(0.0)
    return merged


def phmmer_search_real(
    positives: "pd.DataFrame",
    candidates: "pd.DataFrame",
    phmmer_bin: str = "phmmer",
) -> "pd.DataFrame":
    """Run HMMER(phmmer) as real backend and map to hmm_like_score."""
    import numpy as np
    import pandas as pd

    from .io_utils import write_fasta

    with tempfile.TemporaryDirectory(prefix="mnox_hmmer_") as td:
        tdir = Path(td)
        pos_fa = tdir / "positives.fasta"
        cand_fa = tdir / "candidates.fasta"
        tbl = tdir / "phmmer.tbl"
        write_fasta([(r.id, r.sequence) for r in positives.itertuples()], pos_fa)
        write_fasta([(r.id, r.sequence) for r in candidates.itertuples()], cand_fa)

        run_command([phmmer_bin, "--tblout", str(tbl), str(pos_fa), str(cand_fa)])

        score_map: dict[str, float] = {}
        if tbl.exists():
            for line in tbl.read_text(encoding="utf-8", errors="ignore").splitlines():
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 6:
                    continue
                target = parts[0]
                try:
                    score = float(parts[5])
                except ValueError:
                    score = 0.0
                if target not in score_map or score > score_map[target]:
                    score_map[target] = score

        smax = max(score_map.values()) if score_map else 1.0
        rows = [
            {
                "candidate_id": cid,
                "best_hmm_cluster": -1,
                "hmm_like_score": float(np.clip(score_map.get(cid, 0.0) / max(1e-8, smax), 0, 1)),
            }
            for cid in candidates["id"].tolist()
        ]
        return pd.DataFrame(rows)


def detect_tools() -> dict[str, bool]:
    return {
        "blastp": which("blastp") is not None,
        "makeblastdb": which("makeblastdb") is not None,
        "phmmer": which("phmmer") is not None,
    }
