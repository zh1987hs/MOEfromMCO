from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import Config
from .io_utils import read_fasta, write_fasta
from .pipeline import cross_validate, load_or_simulate, load_real_fasta, run_once
from .simulate import simulate_dataset


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m mnox_retrieval.cli")
    p.add_argument("--config", default="config/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate-data")
    s.add_argument("--out-dir", default="outputs/sim_data")

    d = sub.add_parser("run-demo")
    d.add_argument("--sim-dir", default="outputs/sim_data")
    d.add_argument("--out-dir", default="outputs/demo")

    c = sub.add_parser("cross-validate")
    c.add_argument("--sim-dir", default="outputs/sim_data")
    c.add_argument("--out-dir", default="outputs/cv")

    r = sub.add_parser("rank-real")
    r.add_argument("--positive-fasta", required=True)
    r.add_argument("--unlabeled-fasta", required=True)
    r.add_argument("--out-dir", default="outputs/real")

    e = sub.add_parser("export-top")
    e.add_argument("--ranking-csv", required=True)
    e.add_argument("--source-fasta", required=True)
    e.add_argument("--top-n", type=int, default=100)
    e.add_argument("--out-fasta", default="outputs/top_candidates.fasta")

    pr = sub.add_parser("plot-report")
    pr.add_argument("--cv-metrics", required=True)
    pr.add_argument("--out-dir", default="outputs/plots")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = Config.from_file(args.config)

    if args.cmd == "simulate-data":
        sim = cfg.get("simulation", default={})
        simulate_dataset(args.out_dir, **sim, seed=cfg.get("seed", default=42))
        return
    if args.cmd == "run-demo":
        pos, unl = load_or_simulate(Path(args.sim_dir), cfg)
        run_once(pos, unl, cfg, Path(args.out_dir))
        return
    if args.cmd == "cross-validate":
        pos, unl = load_or_simulate(Path(args.sim_dir), cfg)
        cross_validate(pos, unl, cfg, Path(args.out_dir))
        return
    if args.cmd == "rank-real":
        pos, unl = load_real_fasta(args.positive_fasta, args.unlabeled_fasta)
        run_once(pos, unl, cfg, Path(args.out_dir))
        return
    if args.cmd == "export-top":
        rank = pd.read_csv(args.ranking_csv).head(args.top_n)
        ids = set(rank["candidate_id"].tolist())
        rec = [(rid, seq) for rid, seq in read_fasta(args.source_fasta) if rid in ids]
        write_fasta(rec, args.out_fasta)
        return
    if args.cmd == "plot-report":
        from .plots import plot_recall_curve

        m = pd.read_csv(args.cv_metrics)
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        plot_recall_curve(m, Path(args.out_dir) / "recall_curve.png")


if __name__ == "__main__":
    main()
