from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_markdown_report(rank_csv: str | Path, eval_json: str | Path, out_md: str | Path) -> None:
    rank = pd.read_csv(rank_csv)
    import json

    with Path(eval_json).open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    top = rank.head(20)
    lines = ["# MCO Mn-oxidation retrieval report", "", "## Metrics", ""]
    for k, v in metrics.items():
        lines.append(f"- **{k}**: {v:.4f}" if isinstance(v, (int, float)) else f"- **{k}**: {v}")
    lines += ["", "## Top 20 candidates", "", top.to_markdown(index=False)]
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
