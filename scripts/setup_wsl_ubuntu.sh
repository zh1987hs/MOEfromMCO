#!/usr/bin/env bash
set -euo pipefail

# Bootstrap for WSL Ubuntu/Debian
# Usage:
#   bash scripts/setup_wsl_ubuntu.sh

if ! grep -qi microsoft /proc/version 2>/dev/null; then
  echo "[WARN] This script is intended for WSL. Continuing anyway..."
fi

sudo apt-get update
sudo apt-get install -y hmmer mafft muscle build-essential pkg-config git wget curl

if ! command -v conda >/dev/null 2>&1; then
  echo "[INFO] conda not found. Install Miniconda in WSL first:"
  echo "https://docs.conda.io/en/latest/miniconda.html"
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n mnox python=3.10 -y
conda activate mnox

# Heavy packages from conda to avoid pip source builds.
conda install -c conda-forge -c bioconda -y \
  mmseqs2 hmmer mafft muscle \
  numpy pandas scipy scikit-learn matplotlib pyarrow biopython pyyaml transformers

# CPU PyTorch by default
conda install -c pytorch -y pytorch cpuonly

echo "[OK] WSL Ubuntu environment ready."
echo "[TIP] Put data under Linux FS (e.g. ~/mnox-data) for better IO performance."
echo "[OK] Run: conda activate mnox && python run_pipeline.py"
