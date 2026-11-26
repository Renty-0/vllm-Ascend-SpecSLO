#!/usr/bin/env bash
set -euo pipefail

tag=${1:?usage: run_baseline_v021.sh UNIQUE_TAG [BATCH_SIZE]}
batch=${2:-64}
[[ $tag =~ ^[A-Za-z0-9._-]+$ ]] || exit 2
[[ $batch == 8 || $batch == 16 || $batch == 32 || $batch == 64 ]] || exit 2
export SPECSLO_VISIBLE_DEVICES=${SPECSLO_VISIBLE_DEVICES:-1,2,3,4}
export LLAMA_REPRO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SPECSLO_REPO_ROOT=$(cd "$LLAMA_REPRO_DIR/../../../.." && pwd)

exec bash /root/SpecSLO/budget-calibration-20261015/serial-v1/runtime.sh \
  --environment-only /bin/bash -c '
set -euo pipefail
export PYTHONPATH=/root/SpecSLO/official-v021-baseline-20261112/harness:/root/SpecSLO/official-v021-baseline-20261019/python-v2:/root/SpecSLO/official-v021-baseline-20261019/python:/root/SpecSLO/official-v021-baseline-20261019/vllm-ascend:/root/SpecSLO/official-v021-baseline-20261019/vllm:/root/data/specslo-cann-runtime/cann-9.0.0/python/site-packages
export ASCEND_RT_VISIBLE_DEVICES=$(python -B "$SPECSLO_REPO_ROOT/research/specslo-linear-650/scripts/map_current.py" "$SPECSLO_VISIBLE_DEVICES")
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONHASHSEED=0
exec "$@"
' baseline-env bash "$LLAMA_REPRO_DIR/run_baseline_v021_inner.sh" "$tag" "$batch"
