#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$1"
CAMPAIGN_DIR="$2"
HOPS_PYTHON="$3"
source "$ROOT_DIR/slurm/environment.sh"
"$HOPS_PYTHON" "$ROOT_DIR/submit_hpc.py" --accounting "$CAMPAIGN_DIR" || true
srun --unbuffered "$HOPS_PYTHON" -u "$ROOT_DIR/research_hpc.py" analyse --campaign "$CAMPAIGN_DIR" --collect

