#!/usr/bin/env bash
# Shared environment for the optional trajectory-array route.
PROJECT_ROOT=/home/tariq.qayyum/hpc/polariton_hops
if [[ "$(cd "$ROOT_DIR" && pwd -P)" != "$(cd "$PROJECT_ROOT" && pwd -P)" ]]; then
    printf 'Wrong project directory.\n' >&2
    exit 2
fi
source "$PROJECT_ROOT/scripts/project_env.sh"
if ! type module >/dev/null 2>&1; then
    [[ ! -f /etc/profile.d/modules.sh ]] || source /etc/profile.d/modules.sh
fi
module load anaconda3
if ! command -v srun >/dev/null 2>&1; then module load slurm; fi
cd "$PROJECT_ROOT"
printf 'Job %s, host %s, project %s\n' "${SLURM_JOB_ID:-setup}" "$(hostname)" "$PROJECT_ROOT"
