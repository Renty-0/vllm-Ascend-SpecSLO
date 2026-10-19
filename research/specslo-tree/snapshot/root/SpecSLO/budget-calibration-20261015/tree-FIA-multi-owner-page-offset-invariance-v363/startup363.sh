#!/usr/bin/env bash
set -euo pipefail
task_path=/root/SpecSLO/budget-calibration-20261015/tree-FIA-multi-owner-page-offset-invariance-v363
if [[ $# -ne 1 || ! $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  exit 2
fi
export TASK_QUEUE_ENABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export ASCEND_CACHE_PATH="$task_path/compiler-cache"
ulimit -c 0
mkdir -p -- "$task_path/actual"
mkdir -- "$task_path/actual/$1"
exec python -B -u "$task_path/probe363.py" --logical-device 1 --output "$task_path/actual/$1/result.json"
