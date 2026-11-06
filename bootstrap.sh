#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=/home/tariq.qayyum/hpc/polariton_hops
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/scripts/project_env.sh"
if ! type module >/dev/null 2>&1; then
    [[ ! -f /etc/profile.d/modules.sh ]] || source /etc/profile.d/modules.sh
fi
module load anaconda3
command -v flock >/dev/null || { printf 'flock from util-linux is required for environment locking.\n' >&2; exit 2; }
exec 9>"$PROJECT_ROOT/.locks/environment.lock"
flock -w 900 9
ENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"
READY="$PROJECT_ROOT/.venv/.dependencies.sha256"
if [[ -L "$PROJECT_ROOT/.venv" ]]; then
    printf 'The project environment must be a local directory, not a symlink to another installation.\n' >&2; exit 2
fi
NEEDED="$(sha256sum "$PROJECT_ROOT/requirements.txt" | cut -d' ' -f1)"
if [[ -x "$ENV_PYTHON" && -f "$READY" && "$(cat "$READY")" == "$NEEDED" ]]; then
    printf 'Using the existing project-local environment.\n'
    exit 0
fi
if [[ ! -x "$ENV_PYTHON" ]]; then
    BOOTSTRAP_PYTHON="${HOPS_BOOTSTRAP_PYTHON:-python}"
    if [[ -z "${HOPS_BOOTSTRAP_PYTHON:-}" ]] && command -v python3.11 >/dev/null 2>&1; then BOOTSTRAP_PYTHON=python3.11; fi
    if ! "$BOOTSTRAP_PYTHON" -c 'import sys; raise SystemExit(sys.version_info[:2] != (3,11))'; then
        command -v conda >/dev/null || { printf 'Provide Python 3.11 with HOPS_BOOTSTRAP_PYTHON.\n' >&2; exit 2; }
        cat > "$CONDARC" <<EOF
auto_update_conda: false
register_envs: false
report_errors: false
pkgs_dirs:
  - $CONDA_PKGS_DIRS
envs_dirs:
  - $CONDA_ENVS_PATH
EOF
        conda config --show register_envs > "$PROJECT_ROOT/logs/conda_registration_check.txt"
        grep -qi 'register_envs: false' "$PROJECT_ROOT/logs/conda_registration_check.txt" || {
            printf 'Cannot disable external environment registration. Supply a Python 3.11 interpreter instead.\n' >&2; exit 2;
        }
        if [[ ! -x "$PROJECT_ROOT/.tools/python311/bin/python" ]]; then
            conda create --yes --copy --no-default-packages --override-channels --channel conda-forge --prefix "$PROJECT_ROOT/.tools/python311" python=3.11 pip
        fi
        BOOTSTRAP_PYTHON="$PROJECT_ROOT/.tools/python311/bin/python"
    fi
    "$BOOTSTRAP_PYTHON" -m venv "$PROJECT_ROOT/.venv"
fi
"$ENV_PYTHON" -c 'from pathlib import Path; import sys; root=Path("/home/tariq.qayyum/hpc/polariton_hops").resolve(); prefix=Path(sys.prefix).resolve(); assert prefix.is_relative_to(root) and prefix==root/".venv", "Wrong install prefix"'
PIP_REQUIRE_VIRTUALENV=true "$ENV_PYTHON" -m pip install -r "$PROJECT_ROOT/requirements.txt"
"$ENV_PYTHON" -m pip check
"$ENV_PYTHON" -m pip freeze > "$PROJECT_ROOT/logs/environment_freeze.txt"
printf '%s\n' "$NEEDED" > "$READY"
printf 'Project-local environment is ready: %s\n' "$ENV_PYTHON"
