#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/tariq.qayyum/hpc/polariton_hops
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/project_env.sh"
if ! type module >/dev/null 2>&1; then
    [[ ! -f /etc/profile.d/modules.sh ]] || source /etc/profile.d/modules.sh
fi
module load anaconda3
if ! command -v scontrol >/dev/null 2>&1; then module load slurm; fi
bash "$PROJECT_ROOT/bootstrap.sh"
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/validate_local.py"
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/cluster_preflight.py"
