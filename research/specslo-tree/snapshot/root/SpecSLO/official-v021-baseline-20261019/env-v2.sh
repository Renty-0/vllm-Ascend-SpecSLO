#!/usr/bin/env bash
set -euo pipefail
export SPECSLO_VISIBLE_DEVICES=2,5,6,7
exec bash /root/SpecSLO/budget-calibration-20261015/serial-v1/runtime.sh /bin/bash -c '
set -euo pipefail
export PYTHONPATH=/root/SpecSLO/official-v021-baseline-20261019/python-v2:/root/SpecSLO/official-v021-baseline-20261019/python:/root/SpecSLO/official-v021-baseline-20261019/vllm-ascend:/root/SpecSLO/official-v021-baseline-20261019/vllm:/root/data/specslo-cann-runtime/cann-9.0.0/python/site-packages
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONHASHSEED=0
exec "$@"
' old-official-env "$@"
