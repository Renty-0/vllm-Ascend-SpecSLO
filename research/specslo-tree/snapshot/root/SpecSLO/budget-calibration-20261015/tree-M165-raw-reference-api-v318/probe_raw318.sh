#!/usr/bin/env bash
set -euo pipefail
export TASK_QUEUE_ENABLE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TRITON_CACHE_DIR=/root/SpecSLO/budget-calibration-20261015/tree-M165-raw-reference-api-v318/triton-cache
ulimit -c 0
exec python -B -u /root/SpecSLO/budget-calibration-20261015/tree-M165-raw-reference-api-v318/probe_raw318.py
