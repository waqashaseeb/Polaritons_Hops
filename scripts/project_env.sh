#!/usr/bin/env bash
# Source after setting PROJECT_ROOT. Never changes HOME or the shared Conda root.
case "$PROJECT_ROOT" in
    /home/tariq.qayyum/hpc/polariton_hops) ;;
    *) printf 'Refusing an unexpected project root: %s\n' "$PROJECT_ROOT" >&2; exit 2 ;;
esac
CANONICAL_PROJECT_ROOT="$(readlink -f "$PROJECT_ROOT")"
for directory in logs results outputs figures reports checkpoints runtime .locks .tools .config .venv; do
    candidate="$(readlink -m "$PROJECT_ROOT/$directory")"
    case "$candidate" in
        "$CANONICAL_PROJECT_ROOT/"*) ;;
        *) printf 'Project directory escapes through a symlink: %s\n' "$candidate" >&2; exit 2 ;;
    esac
done
mkdir -p "$PROJECT_ROOT"/{logs,results,outputs,figures,reports,checkpoints,runtime,.locks,.tools,.config}
export TMPDIR="$PROJECT_ROOT/runtime/tmp/${SLURM_JOB_ID:-setup}_${SLURM_PROCID:-0}"
export PIP_CACHE_DIR="$PROJECT_ROOT/runtime/pip-cache"
export PIP_CONFIG_FILE=/dev/null
export XDG_CACHE_HOME="$PROJECT_ROOT/runtime/xdg-cache"
export XDG_CONFIG_HOME="$PROJECT_ROOT/.config"
export XDG_DATA_HOME="$PROJECT_ROOT/runtime/xdg-data"
export MPLCONFIGDIR="$PROJECT_ROOT/runtime/matplotlib_${SLURM_JOB_ID:-setup}_${SLURM_PROCID:-0}"
export NUMBA_CACHE_DIR="$PROJECT_ROOT/runtime/numba_${SLURM_JOB_ID:-setup}_${SLURM_PROCID:-0}"
export PYTHONPYCACHEPREFIX="$PROJECT_ROOT/runtime/pycache"
export PYTHONUSERBASE="$PROJECT_ROOT/runtime/python-userbase"
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PROJECT_ROOT"
unset PYTHONHOME
export CONDA_PKGS_DIRS="$PROJECT_ROOT/runtime/conda-pkgs"
export CONDA_ENVS_PATH="$PROJECT_ROOT/.tools/conda-envs"
export CONDARC="$PROJECT_ROOT/.config/condarc"
export CONDA_REGISTER_ENVS=false
export CONDA_REPORT_ERRORS=false
export CONDA_AUTO_UPDATE_CONDA=false
export CONDA_NO_PLUGINS=true
export CONDA_SOLVER=classic
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMBA_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLBACKEND=Agg
export PYTHONIOENCODING=utf-8
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" \
    "$MPLCONFIGDIR" "$NUMBA_CACHE_DIR" "$PYTHONPYCACHEPREFIX" "$CONDA_PKGS_DIRS" "$CONDA_ENVS_PATH"
