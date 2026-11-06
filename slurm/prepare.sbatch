#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$1"
CONFIG_PATH="$2"
CAMPAIGN_DIR="$3"
HOPS_PYTHON="$4"
source "$ROOT_DIR/slurm/environment.sh"
srun --unbuffered "$HOPS_PYTHON" -u "$ROOT_DIR/research_hpc.py" prepare --config "$CONFIG_PATH" --campaign "$CAMPAIGN_DIR"

