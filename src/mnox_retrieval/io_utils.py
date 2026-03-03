from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


AA20 = set("ACDEFGHIKLMNPQRSTVWY")


def read_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Stream FASTA records as (id, sequence)."""
    for rec in SeqIO.parse(str(path), "fasta"):
        seq = str(rec.seq).upper()
        yield rec.id, "".join([aa for aa in seq if aa in AA20])


def write_fasta(records: Iterable[tuple[str, str]], path: str | Path) -> None:
    seq_records = [SeqRecord(seq=seq, id=rid, description="") for rid, seq in records]
    # Bio.SeqRecord expects Seq; string accepted by SeqIO.write via SeqRecord.seq assignment handling
    from Bio.Seq import Seq

    fixed = [SeqRecord(Seq(str(r.seq)), id=r.id, description="") for r in seq_records]
    SeqIO.write(fixed, str(path), "fasta")


def save_json(data: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_table(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
