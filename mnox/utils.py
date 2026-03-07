from __future__ import annotations

import importlib
import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass
class ToolCheckResult:
    """Result for an external tool check."""

    name: str
    found: bool
    path: str | None


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML configuration file."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_run_dir(base_output_dir: str | Path) -> Path:
    """Create timestamped run directory."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_output_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def setup_logger(run_dir: Path, name: str = "mnox") -> logging.Logger:
    """Create logger writing to console and file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(run_dir / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def ensure_dir(path: str | Path) -> Path:
    """Create directory if not existing."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def check_python_dependencies() -> None:
    """Check required Python packages and raise with install hints if missing."""
    required = [
        "Bio",
        "numpy",
        "pandas",
        "sklearn",
        "scipy",
        "yaml",
        "matplotlib",
        "torch",
    ]
    optional_esm_providers = ["transformers", "esm"]

    missing = []
    for module in required:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(module)

    if not any(_can_import(m) for m in optional_esm_providers):
        missing.append("transformers_or_fair-esm")

    if missing:
        hint = (
            "Missing Python packages: "
            f"{', '.join(missing)}\n"
            "Install suggestions:\n"
            "- conda: conda install biopython numpy pandas scikit-learn scipy pyyaml matplotlib pytorch -c pytorch\n"
            "- pip: pip install biopython numpy pandas scikit-learn scipy pyyaml matplotlib torch transformers\n"
            "- fair-esm optional: pip install fair-esm"
        )
        raise RuntimeError(hint)


def _can_import(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def check_external_tools(msa_tool: str) -> dict[str, ToolCheckResult]:
    """Check required external executables and raise detailed errors if missing."""
    tool_names = ["mmseqs", "hmmbuild", "hmmsearch", msa_tool]
    results: dict[str, ToolCheckResult] = {}

    missing = []
    for name in tool_names:
        path = shutil.which(name)
        found = path is not None
        results[name] = ToolCheckResult(name=name, found=found, path=path)
        if not found:
            missing.append(name)

    if missing:
        msg = (
            f"Missing external tools: {', '.join(missing)}\n"
            "Linux examples:\n"
            "- conda: conda install -c bioconda mmseqs2 hmmer mafft muscle\n"
            "- apt (partial): sudo apt-get install hmmer mafft muscle\n"
            "macOS examples:\n"
            "- brew: brew install mmseqs2 hmmer mafft muscle\n"
            "Windows (PowerShell) examples:\n"
            "- conda: conda install -c bioconda mmseqs2 hmmer mafft muscle\n"
            "- chocolatey: choco install hmmer mafft muscle\n"
            "- Prefer running MMseqs2/HMMER in WSL2 for best compatibility, or add native binaries to PATH."
        )
        raise RuntimeError(msg)

    return results


def run_command(
    cmd: list[str],
    logger: logging.Logger,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    """Run a subprocess command and fail with captured output if command returns non-zero."""
    logger.info("RUN: %s", " ".join(cmd))
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.stdout:
        logger.info(proc.stdout.strip())
    if proc.returncode != 0:
        err = proc.stderr.strip() if proc.stderr else "<no stderr>"
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{err}")


def save_json(data: dict[str, Any], out_path: str | Path) -> None:
    """Save dict as pretty JSON."""
    with Path(out_path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_lines(lines: Iterable[str], out_path: str | Path) -> None:
    """Write iterable lines to text file."""
    with Path(out_path).open("w", encoding="utf-8") as f:
        for line in lines:
            f.write(f"{line}\n")


def write_dataframe(df: Any, out_base: Path, logger: logging.Logger) -> Path:
    """Write dataframe to parquet when possible, else CSV."""
    try:
        import pyarrow  # noqa: F401

        out_path = out_base.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)
        return out_path
    except Exception:
        out_path = out_base.with_suffix(".csv")
        df.to_csv(out_path, index=False)
        logger.warning(
            "pyarrow missing/unavailable; wrote CSV instead. Install with: pip install pyarrow"
        )
        return out_path
