#!/usr/bin/env bash
set -e

# Discard the base image's CANN 9.1/Python 3.12 paths before loading SpecSLO.
if [[ ${SPECSLO_ENV_CLEANED:-0} != 1 ]]; then
  preserved=()
  while IFS='=' read -r key value; do
    case "$key" in
      SPECSLO_*|VLLM_ASCEND_*|COMPILE_CUSTOM_KERNELS|VLLM_BATCH_INVARIANT|VLLM_TARGET_DEVICE|MAX_JOBS|PYTHONHASHSEED)
        preserved+=("$key=$value") ;;
    esac
  done < <(env)
  exec /usr/bin/env -i \
    HOME=/root USER=root LOGNAME=root SHELL=/bin/bash LANG=C.UTF-8 \
    PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin \
    LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver \
    PYTHONPATH= LD_PRELOAD= SPECSLO_ENV_CLEANED=1 \
    "${preserved[@]}" /bin/bash "$0" "$@"
fi

# Retained runtime is persistent on /root; no installer or transient alias needed.
source /root/data/specslo-cann-runtime/cann-9.0.0/set_env.sh
source /root/data/specslo-cann-runtime/nnal/atb/9.0.0/atb/set_env.sh --cxx_abi=1
export PATH=/root/miniconda3/envs/vllm-hust-dev/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/vllm-hust-dev/lib/python3.11/site-packages/torch/lib:/root/miniconda3/envs/vllm-hust-dev/lib/python3.11/site-packages/torch_npu/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust:/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-hust:$PYTHONPATH
export SPECSLO_DRAFT_MODEL=/root/data/shared-models/Qwen3-0.6B
export SPECSLO_TARGET_MODEL=/root/data/shared-models/Qwen3-32B
export HF_HOME=/data/shared_upload/models/.cache/specslo-huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export MODELSCOPE_CACHE=/data/shared_upload/models/.cache/specslo-modelscope
export SPECSLO_VISIBLE_DEVICES=${SPECSLO_VISIBLE_DEVICES:-0,1,2,7}
export SPECSLO_TARGET_DEVICES=${SPECSLO_TARGET_DEVICES:-1,2,3}
# Bootstrap metadata query skips only the mapper, never model configuration.
if [[ ${1:-} == --environment-only ]]; then
  shift
  unset ASCEND_RT_VISIBLE_DEVICES
  export PYTHONHASHSEED=0
  exec "$@"
fi
container_runtime_devices=$(python -B /root/SpecSLO/budget-calibration-20261015/tree-current-card-rebind-full-v464/map464.py "$SPECSLO_VISIBLE_DEVICES")
export ASCEND_RT_VISIBLE_DEVICES=$container_runtime_devices
export PYTHONHASHSEED=0
exec "$@"
