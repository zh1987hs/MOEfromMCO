from __future__ import annotations

import importlib
import json
import logging
import platform
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


def detect_linux_distro() -> str:
    """Best-effort Linux distro detection using /etc/os-release."""
    os_release = Path("/etc/os-release")
    if not os_release.exists():
        return "unknown"
    content = os_release.read_text(encoding="utf-8", errors="ignore").lower()
    if "centos" in content:
        return "centos"
    if "rhel" in content or "red hat" in content:
        return "rhel"
    if "rocky" in content:
        return "rocky"
    if "almalinux" in content:
        return "almalinux"
    if "ubuntu" in content:
        return "ubuntu"
    if "debian" in content:
        return "debian"
    return "linux"




def is_wsl() -> bool:
    """Detect whether current Linux runtime is WSL."""
    if platform.system().lower() != "linux":
        return False
    proc_version = Path("/proc/version")
    if proc_version.exists() and "microsoft" in proc_version.read_text(encoding="utf-8", errors="ignore").lower():
        return True
    os_release = Path("/etc/os-release")
    if os_release.exists() and "microsoft" in os_release.read_text(encoding="utf-8", errors="ignore").lower():
        return True
    return False


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
            "- Recommended (CentOS): conda install -c conda-forge -c bioconda biopython numpy pandas scikit-learn scipy pyyaml matplotlib transformers pyarrow && conda install -c pytorch pytorch cpuonly\n"
            "- pip (may compile and fail on old CentOS toolchains): pip install biopython numpy pandas scikit-learn scipy pyyaml matplotlib torch transformers\n"
            "- fair-esm optional: pip install fair-esm"
        )
        raise RuntimeError(hint)


def _can_import(module: str) -> bool:
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


def _external_tool_hints() -> str:
    """Build OS-specific tool installation hints."""
    sys_name = platform.system().lower()
    if sys_name == "linux":
        distro = detect_linux_distro()
        if is_wsl() and distro in {"ubuntu", "debian", "linux"}:
            return (
                "WSL Ubuntu/Debian examples:\n"
                "- sudo apt-get update && sudo apt-get install -y hmmer mafft muscle build-essential\n"
                "- MMseqs2 推荐用 conda: conda install -c conda-forge -c bioconda mmseqs2 hmmer mafft muscle\n"
                "- 性能建议: 数据放在 WSL Linux 文件系统（如 ~/projects），避免 /mnt/c 上的大规模 I/O"
            )
        if distro in {"centos", "rhel", "rocky", "almalinux"}:
            return (
                "CentOS/RHEL-like examples:\n"
                "- sudo dnf install -y epel-release\n"
                "- sudo dnf install -y hmmer mafft muscle\n"
                "- MMseqs2 推荐用 conda: conda install -c bioconda mmseqs2 hmmer mafft muscle\n"
                "- 若是 CentOS 7 用 yum: sudo yum install -y epel-release hmmer mafft muscle"
            )
        if distro in {"ubuntu", "debian"}:
            return (
                "Ubuntu/Debian examples:\n"
                "- sudo apt-get update && sudo apt-get install -y hmmer mafft muscle build-essential\n"
                "- conda: conda install -c conda-forge -c bioconda mmseqs2 hmmer mafft muscle"
            )
        return (
            "Linux examples:\n"
            "- conda: conda install -c bioconda mmseqs2 hmmer mafft muscle\n"
            "- apt (partial): sudo apt-get install hmmer mafft muscle"
        )
    if sys_name == "darwin":
        return "macOS examples:\n- brew install mmseqs2 hmmer mafft muscle"
    return (
        "Windows (PowerShell) examples:\n"
        "- conda: conda install -c bioconda mmseqs2 hmmer mafft muscle\n"
        "- chocolatey: choco install hmmer mafft muscle\n"
        "- Prefer WSL2 for MMseqs2/HMMER compatibility"
    )


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
            f"{_external_tool_hints()}"
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
