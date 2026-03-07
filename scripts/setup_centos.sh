#!/usr/bin/env bash
set -euo pipefail

# CentOS/RHEL-like bootstrap for MnOx pipeline dependencies.
# Usage:
#   bash scripts/setup_centos.sh

if command -v dnf >/dev/null 2>&1; then
  sudo dnf install -y epel-release
  sudo dnf install -y hmmer mafft muscle git wget
elif command -v yum >/dev/null 2>&1; then
  sudo yum install -y epel-release
  sudo yum install -y hmmer mafft muscle git wget
else
  echo "[WARN] Neither dnf nor yum found. Please install hmmer/mafft/muscle manually."
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[INFO] conda not found. Install Miniconda first:"
  echo "https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

conda create -n mnox python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mnox

# Prefer bioconda for mmseqs2 and bio tools consistency.
# IMPORTANT: install heavy scientific stack from conda to avoid local compilation issues
# on old CentOS toolchains (e.g., gcc 4.8 on CentOS 7).
conda install -c conda-forge -c bioconda -y \
  mmseqs2 hmmer mafft muscle \
  numpy pandas scipy scikit-learn matplotlib pyarrow biopython pyyaml transformers

# PyTorch: prefer conda binary packages.
# CPU-only:
conda install -c pytorch -y pytorch cpuonly
# If you need CUDA, replace with the command from https://pytorch.org/get-started/locally/

echo "[OK] Environment ready. Activate with: conda activate mnox"
echo "[OK] Then run: python run_pipeline.py"
