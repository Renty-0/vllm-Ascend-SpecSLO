#!/usr/bin/env bash
set -euo pipefail
export TASK_QUEUE_ENABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
ulimit -c 0
exec python -B -u /root/SpecSLO/budget-calibration-20261015/tree-FIA-group-root-isolation-v324/probe324.py
