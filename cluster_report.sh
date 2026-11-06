#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
WORKSPACE_BASE="$(cd /home/tariq.qayyum/hpc/polariton_hops && pwd -P)"
case "$ROOT_DIR/" in
    "$WORKSPACE_BASE/"*) ;;
    *) printf 'Extract this package below /home/tariq.qayyum/hpc/polariton_hops/ first.\n' >&2; exit 1 ;;
esac
cd "$ROOT_DIR"
mkdir -p logs
{
    date -u
    hostname
    command -v python3 || true
    python3 --version || true
    sinfo -a -o '%P %a %l %D %c %m %G %N'
    scontrol show partition
    scontrol show config | grep -E 'MaxArraySize|AccountingStorageType|SelectType' || true
} 2>&1 | tee logs/cluster_report.log

