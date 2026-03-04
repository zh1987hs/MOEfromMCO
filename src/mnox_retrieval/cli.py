from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import Config
from .external_tools import detect_tools
from .installer import get_install_commands, run_install_commands
from .io_utils import read_fasta, write_fasta
from .pipeline import cross_validate, load_or_simulate, load_real_fasta, run_once
from .simulate import simulate_dataset


def _require_exists(path: str | Path, arg_name: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{arg_name} not found: {p}")
    return p


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m mnox_retrieval.cli")
    p.add_argument("--config", default="config/default.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate-data", help="Generate simulated positive/unlabeled FASTA+CSV")
    s.add_argument("--out-dir", default="outputs/sim_data")

    d = sub.add_parser("run-demo", help="Run full pipeline on simulated data")
    d.add_argument("--sim-dir", default="outputs/sim_data")
    d.add_argument("--out-dir", default="outputs/demo")

    c = sub.add_parser("cross-validate", help="Leave-one-cluster-out CV on simulated data")
    c.add_argument("--sim-dir", default="outputs/sim_data")
    c.add_argument("--out-dir", default="outputs/cv")

    r = sub.add_parser("rank-real", help="Rank candidates from real FASTA files")
    r.add_argument("--positive-fasta", required=True)
    r.add_argument("--unlabeled-fasta", required=True)
    r.add_argument("--out-dir", default="outputs/real")

    e = sub.add_parser("export-top", help="Export top-N candidates from ranking CSV into FASTA")
    e.add_argument("--ranking-csv", required=True)
    e.add_argument("--source-fasta", required=True)
    e.add_argument("--top-n", type=int, default=100)
    e.add_argument("--out-fasta", default="outputs/top_candidates.fasta")

    pr = sub.add_parser("plot-report", help="Plot recall curve from CV metrics")
    pr.add_argument("--cv-metrics", required=True)
    pr.add_argument("--out-dir", default="outputs/plots")

    doc = sub.add_parser("doctor", help="Show detected external tools and backend mode")
    doc.add_argument("--json", action="store_true")

    inst = sub.add_parser("install-tools", help="Print/install helper commands for BLAST+/HMMER")
    inst.add_argument("--method", choices=["conda", "mamba", "choco", "scoop"], default="conda")
    inst.add_argument("--execute", action="store_true", help="Actually run installer commands")
    return p


def _print_install_help(method: str) -> None:
    print("\n".join(get_install_commands(method)))
    print("# 离线 ESM: 手动下载模型目录后，在 config/default.yaml 中设置 embedding.esm_model_path")


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
        pos_fa = _require_exists(args.positive_fasta, "--positive-fasta")
        unl_fa = _require_exists(args.unlabeled_fasta, "--unlabeled-fasta")
        pos, unl = load_real_fasta(str(pos_fa), str(unl_fa))
        run_once(pos, unl, cfg, Path(args.out_dir))
        return
    if args.cmd == "export-top":
        ranking_csv = _require_exists(args.ranking_csv, "--ranking-csv")
        source_fasta = _require_exists(args.source_fasta, "--source-fasta")
        rank = pd.read_csv(ranking_csv).head(args.top_n)
        ids = set(rank["candidate_id"].tolist())
        rec = [(rid, seq) for rid, seq in read_fasta(source_fasta) if rid in ids]
        write_fasta(rec, args.out_fasta)
        return
    if args.cmd == "plot-report":
        from .plots import plot_recall_curve

        metrics_path = _require_exists(args.cv_metrics, "--cv-metrics")
        m = pd.read_csv(metrics_path)
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        plot_recall_curve(m, Path(args.out_dir) / "recall_curve.png")
        return
    if args.cmd == "doctor":
        tools = detect_tools()
        if args.json:
            import json

            print(json.dumps(tools, indent=2))
        else:
            for k, v in tools.items():
                print(f"{k}: {'OK' if v else 'MISSING'}")
            print("提示: external.blast_mode/hmm_mode=auto 时，工具存在则走真实后端，否则自动回退。")
        return
    if args.cmd == "install-tools":
        _print_install_help(args.method)
        if args.execute:
            for cmd, code in run_install_commands(args.method):
                print(f"[{code}] {cmd}")
        return


if __name__ == "__main__":
    main()
