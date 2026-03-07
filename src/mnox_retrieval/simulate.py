from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .io_utils import save_table, write_fasta

AA = "ACDEFGHIKLMNPQRSTVWY"
MCO_MOTIFS = ["HXH", "HCH", "CXXH", "HXXH"]
CLUSTER_MOTIFS = ["DDE", "EEDD", "DEDE", "EDDD", "DDHE"]


@dataclass
class SimulationResult:
    positives: pd.DataFrame
    unlabeled: pd.DataFrame


def _mutate(seq: str, rate: float, rng: random.Random) -> str:
    arr = list(seq)
    n = max(1, int(len(arr) * rate))
    for _ in range(n):
        i = rng.randrange(len(arr))
        arr[i] = rng.choice(AA)
    return "".join(arr)


def _insert_motif(seq: str, motif: str, rng: random.Random) -> str:
    arr = list(seq)
    pos = rng.randrange(10, max(11, len(arr) - len(motif) - 10))
    arr[pos : pos + len(motif)] = list(motif)
    return "".join(arr)


def _acidic_patch(seq: str, rng: random.Random, strength: int = 8) -> str:
    arr = list(seq)
    start = rng.randrange(50, max(51, len(arr) - 30))
    for i in range(start, min(len(arr), start + 20)):
        if rng.random() < strength / 20:
            arr[i] = rng.choice("DE")
    return "".join(arr)


def _base_mco(length: int, rng: random.Random) -> str:
    seq = "".join(rng.choice(AA) for _ in range(length))
    for motif in rng.sample(MCO_MOTIFS, k=2):
        motif_real = motif.replace("X", rng.choice(AA))
        seq = _insert_motif(seq, motif_real, rng)
    return seq


def simulate_dataset(
    out_dir: str | Path,
    n_positives: int = 200,
    n_clusters: int = 5,
    unlabeled_size: int = 6000,
    easy_hits: int = 700,
    hidden_positives: int = 80,
    length_min: int = 450,
    length_max: int = 700,
    seed: int = 42,
) -> SimulationResult:
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    cluster_sizes = [n_positives // n_clusters] * n_clusters
    for i in range(n_positives % n_clusters):
        cluster_sizes[i] += 1

    positive_rows: list[dict] = []
    seeds: list[str] = []
    for c in range(n_clusters):
        core = _base_mco(rng.randint(length_min, length_max), rng)
        core = _insert_motif(core, CLUSTER_MOTIFS[c % len(CLUSTER_MOTIFS)], rng)
        core = _acidic_patch(core, rng, strength=10)
        seeds.append(core)
        for i in range(cluster_sizes[c]):
            seq = _mutate(core, rate=rng.uniform(0.04, 0.12), rng=rng)
            pid = f"pos_c{c}_{i:03d}"
            positive_rows.append({"id": pid, "sequence": seq, "cluster": c, "label": 1, "source": "positive"})

    unlabeled_rows: list[dict] = []
    normal_n = unlabeled_size - easy_hits - hidden_positives
    for i in range(max(0, normal_n)):
        seq = _base_mco(rng.randint(length_min, length_max), rng)
        if rng.random() < 0.35:
            seq = _acidic_patch(seq, rng, strength=4)
        uid = f"unl_norm_{i:05d}"
        unlabeled_rows.append({"id": uid, "sequence": seq, "is_hidden_positive": 0, "kind": "normal"})

    for i in range(easy_hits):
        core = rng.choice(seeds)
        seq = _mutate(core, rate=rng.uniform(0.02, 0.07), rng=rng)
        uid = f"unl_easy_{i:05d}"
        unlabeled_rows.append({"id": uid, "sequence": seq, "is_hidden_positive": 0, "kind": "easy_hit"})

    for i in range(hidden_positives):
        core = rng.choice(seeds)
        seq = _mutate(core, rate=rng.uniform(0.22, 0.35), rng=rng)
        seq = _insert_motif(seq, rng.choice(CLUSTER_MOTIFS), rng)
        seq = _acidic_patch(seq, rng, strength=9)
        uid = f"unl_hidden_{i:05d}"
        unlabeled_rows.append({"id": uid, "sequence": seq, "is_hidden_positive": 1, "kind": "distant_hidden_pos"})

    positives = pd.DataFrame(positive_rows)
    unlabeled = pd.DataFrame(unlabeled_rows)

    save_table(positives, out / "positives.csv")
    save_table(unlabeled, out / "unlabeled.csv")
    write_fasta([(r["id"], r["sequence"]) for _, r in positives.iterrows()], out / "positives.fasta")
    write_fasta([(r["id"], r["sequence"]) for _, r in unlabeled.iterrows()], out / "unlabeled.fasta")

    return SimulationResult(positives=positives, unlabeled=unlabeled)
