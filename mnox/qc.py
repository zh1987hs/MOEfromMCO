from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from Bio.SeqRecord import SeqRecord


@dataclass
class QCResult:
    """Container for QC output records and metadata."""

    kept_records: list[SeqRecord]
    kept_ids: list[str]
    duplicate_map: dict[str, list[str]]
    removed_too_short: int
    removed_too_long: int
    removed_motif_fail: int


DEFAULT_MOTIFS = ["HWHG", "HXH", "HCH"]


def _motif_pass(seq: str) -> bool:
    seq_u = seq.upper()
    return ("HWHG" in seq_u) or (("HXH" in seq_u) and ("HCH" in seq_u))


def dedupe_and_filter(
    records: list[SeqRecord],
    min_len: int,
    max_len: int,
    check_motif: bool = False,
) -> QCResult:
    """Deduplicate identical sequences and apply length/motif filtering."""
    by_seq: dict[str, list[SeqRecord]] = defaultdict(list)
    for rec in records:
        by_seq[str(rec.seq)].append(rec)

    kept_records: list[SeqRecord] = []
    kept_ids: list[str] = []
    duplicate_map: dict[str, list[str]] = {}
    removed_too_short = 0
    removed_too_long = 0
    removed_motif_fail = 0

    for seq, recs in by_seq.items():
        canonical = recs[0]
        duplicates = [r.id for r in recs[1:]]
        duplicate_map[canonical.id] = duplicates

        length = len(seq)
        if length < min_len:
            removed_too_short += 1
            continue
        if length > max_len:
            removed_too_long += 1
            continue

        if check_motif and not _motif_pass(seq):
            removed_motif_fail += 1
            continue

        kept_records.append(canonical)
        kept_ids.append(canonical.id)

    return QCResult(
        kept_records=kept_records,
        kept_ids=kept_ids,
        duplicate_map=duplicate_map,
        removed_too_short=removed_too_short,
        removed_too_long=removed_too_long,
        removed_motif_fail=removed_motif_fail,
    )


def qc_summary_dict(pos_result: QCResult, unl_result: QCResult) -> dict:
    """Build JSON serializable QC summary payload."""
    return {
        "positives": {
            "kept": len(pos_result.kept_records),
            "removed_too_short": pos_result.removed_too_short,
            "removed_too_long": pos_result.removed_too_long,
            "removed_motif_fail": pos_result.removed_motif_fail,
            "dedupe_clusters": len(pos_result.duplicate_map),
        },
        "unlabeled": {
            "kept": len(unl_result.kept_records),
            "removed_too_short": unl_result.removed_too_short,
            "removed_too_long": unl_result.removed_too_long,
            "removed_motif_fail": unl_result.removed_motif_fail,
            "dedupe_clusters": len(unl_result.duplicate_map),
        },
    }


def save_duplicate_map(dup_map: dict[str, list[str]], out_path: str | Path) -> None:
    """Save duplicate map as TSV."""
    with Path(out_path).open("w", encoding="utf-8") as f:
        f.write("canonical_id\tduplicate_ids\n")
        for k, v in dup_map.items():
            f.write(f"{k}\t{';'.join(v)}\n")
