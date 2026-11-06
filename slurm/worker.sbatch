#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$1"
CAMPAIGN_DIR="$2"
STAGE="$3"
HOPS_PYTHON="$4"
source "$ROOT_DIR/slurm/environment.sh"
srun --unbuffered "$HOPS_PYTHON" -u "$ROOT_DIR/research_hpc.py" worker --campaign "$CAMPAIGN_DIR" --stage "$STAGE" --position "$SLURM_ARRAY_TASK_ID"

