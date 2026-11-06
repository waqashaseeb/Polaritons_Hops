#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$1"
CONFIG_PATH="$2"
HOPS_PYTHON="$3"
source "$ROOT_DIR/slurm/environment.sh"
srun --unbuffered "$HOPS_PYTHON" -u "$ROOT_DIR/research_hpc.py" paper-reference-grid --config "$CONFIG_PATH"

