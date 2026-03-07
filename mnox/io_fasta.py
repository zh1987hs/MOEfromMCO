from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord


def read_fasta_records(path: str | Path) -> list[SeqRecord]:
    """Read FASTA file into SeqRecord list."""
    return list(SeqIO.parse(str(path), "fasta"))


def stream_fasta_records(path: str | Path) -> Iterator[SeqRecord]:
    """Stream FASTA records lazily."""
    yield from SeqIO.parse(str(path), "fasta")


def write_fasta_records(records: Iterable[SeqRecord], out_path: str | Path) -> None:
    """Write records to FASTA file."""
    SeqIO.write(list(records), str(out_path), "fasta")


def id_to_seq_dict(path: str | Path) -> dict[str, str]:
    """Return mapping from sequence id to sequence string."""
    out: dict[str, str] = {}
    for record in stream_fasta_records(path):
        out[record.id] = str(record.seq)
    return out
