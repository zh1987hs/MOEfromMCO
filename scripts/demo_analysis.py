from __future__ import annotations

from pathlib import Path

from mnox_retrieval.config import Config
from mnox_retrieval.pipeline import cross_validate, load_or_simulate, run_once
from mnox_retrieval.reports import generate_markdown_report


def main() -> None:
    cfg = Config.from_file("config/default.yaml")
    sim_dir = Path("outputs/sim_data")
    demo_dir = Path("outputs/demo")
    cv_dir = Path("outputs/cv")

    positives, unlabeled = load_or_simulate(sim_dir, cfg)
    run_once(positives, unlabeled, cfg, demo_dir)
    cross_validate(positives, unlabeled, cfg, cv_dir)
    generate_markdown_report(demo_dir / "ranked_candidates.csv", demo_dir / "evaluation_summary.json", demo_dir / "report.md")


if __name__ == "__main__":
    main()
