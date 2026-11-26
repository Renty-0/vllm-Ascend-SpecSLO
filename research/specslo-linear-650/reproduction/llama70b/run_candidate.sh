#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! $1 =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo 'usage: run_candidate.sh UNIQUE_TAG' >&2
  exit 2
fi

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)
export SPECSLO_MC2_MODE=off
export SPECSLO_ABLATION_MODE=auto
export SPECSLO_VISIBLE_DEVICES=1,2,3,4
export SPECSLO_BATCH_SIZE=64
export SPECSLO_DRAFT_MODEL=/data/shared_upload/models/Llama-3.2-1B-Instruct
export SPECSLO_TARGET_MODEL=/data/shared_upload/models/Meta-Llama-3.1-70B-Instruct
export SPECSLO_MAX_ACLGRAPH_ENTRIES=32
export SPECSLO_NUM_KVCACHE_BLOCKS=640
export SPECSLO_MAX_TARGET_BATCH=16
export SPECSLO_ONLINE_TARGET_MAX_BATCH=0
export SPECSLO_TARGET_NZ_MODE=default
export SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=0
export SPECSLO_LOOSE_ADMISSION_NORMAL_GATE=0
export SPECSLO_ADMISSION_KV_GUARD_PAGES=639
export SPECSLO_NORMAL_ACTIVE_CAP=8
export SPECSLO_NORMAL_ACTIVE_HARD_CAP=8
export SPECSLO_TIGHT_ACTIVE_CAP=8
export SPECSLO_EAGER_CLASS_FLOOR_BIAS=0.1
export SPECSLO_GAMMA=4
export SPECSLO_MIN_GAMMA=4
export SPECSLO_VERIFICATION_BUDGET=128
export SPECSLO_TARGET_SLO_FIRST=1
export SPECSLO_HCCL_CONNECT_TIMEOUT=900
export SPECSLO_PROMPT_TOKENIZER_MODEL=target
export SPECSLO_PRODUCTION_GATE_SCRIPT="$repo/research/specslo-linear-650/reproduction/llama70b/gate.py"

exec bash "$repo/research/specslo-linear-650/run.sh" "$1" 0.4
