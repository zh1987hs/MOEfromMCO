from __future__ import annotations

import subprocess


def get_install_commands(method: str) -> list[str]:
    mapping = {
        "conda": ["conda install -c bioconda blast hmmer -y"],
        "mamba": ["mamba install -c bioconda blast hmmer -y"],
        "choco": ["choco install blast -y"],
        "scoop": ["scoop install ncbi-blast"],
    }
    return mapping[method]


def run_install_commands(method: str) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for cmd in get_install_commands(method):
        proc = subprocess.run(cmd, shell=True)
        out.append((cmd, proc.returncode))
    return out
