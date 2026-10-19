#!/usr/bin/env bash
set -euo pipefail
export ASCEND_LAUNCH_BLOCKING=1 TASK_QUEUE_ENABLE=0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -c 0
exec python -B -u /root/SpecSLO/budget-calibration-20261015/tree-M165-raw-reference-api-v318/probe_reference318.py "$@"
