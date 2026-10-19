#!/usr/bin/env bash
set -euo pipefail
task_root=/root/SpecSLO/budget-calibration-20261015/tree-causal80-small-cohort-role-aware-audit-full-v209
if [[ $# -ne 4 || $1 != --action || $3 != --tag ||
      $2 != alternate-chain-B1to8-M165-N32-node-component ||
      ! $4 =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo 'invalid local diagnostic action/tag' >&2; exit 2
fi
action=$2
tag=$4
export TRITON_CACHE_DIR=/root/SpecSLO/budget-calibration-20261015/tree-M165-N32-independent-target-math-cost-v317/triton-cache/M165-N32-NZ166-root-v320-B1to8-546-math-cost-v1
frozen=/root/data/specslo-baselines/nano-pearl-ascend-v1
repo=$frozen/source/vllm-ascend-hust
vendor=$frozen/runtime/mc2-vendor
python=/root/miniconda3/envs/vllm-hust-dev/bin/python
export PYTHONDONTWRITEBYTECODE=1
"$python" -B "/root/SpecSLO/budget-calibration-20261015/tree-M165-N32-independent-target-math-cost-v317/check317.py"
"$python" -B "$task_root/check209.py"
export PYTHONPATH=/root/SpecSLO/budget-calibration-20261015/tree-M165-common-mask11-target-math-cost-v323:/root/SpecSLO/budget-calibration-20261015/tree-M165-raw-reference-api-v318:/root/SpecSLO/budget-calibration-20261015/tree-M165-N32-independent-target-math-cost-v317:/root/SpecSLO/budget-calibration-20261015/tree-M165-single-prepare-host-v313:/root/SpecSLO/budget-calibration-20261015/tree-uniform-q5-M165-S33-C128-host-v285:/root/SpecSLO/budget-calibration-20261015/tree-single-prepare-target-math-cost-v311:/root/SpecSLO/budget-calibration-20261015/tree-single-validated-prefix-prepare-host-v305:/root/SpecSLO/budget-calibration-20261015/tree-single-validated-prefix-descriptors-host-v304:/root/SpecSLO/budget-calibration-20261015/tree-concrete-device-span-dual-cache-target-cost-v296:/root/SpecSLO/budget-calibration-20261015/tree-contiguous-prefix-span-copy-host-v292:/root/SpecSLO/budget-calibration-20261015/tree-prefix-precopy-dual-cachewrite-host-v288:/root/SpecSLO/budget-calibration-20261015/tree-M165-N32-independent-target-math-cost-v317:/root/SpecSLO/budget-calibration-20261015/tree-prefix-grid-byte-KV-copy-v278:/root/SpecSLO/budget-calibration-20261015/tree-alternate-chain-target-host-v269:/root/SpecSLO/budget-calibration-20261015/tree-fused-byte-branch-KV-copy-v263:/root/SpecSLO/budget-calibration-20261015/tree-fused-branch-KV-row-copy-v262:/root/SpecSLO/budget-calibration-20261015/tree-static-q5-M120-node-math-v255:/root/SpecSLO/budget-calibration-20261015/tree-M165-N32-independent-target-math-cost-v317:/root/SpecSLO/budget-calibration-20261015/tree-packed-KV-preflight-rank-aware-own-AR-full-v218:$task_root:/root/SpecSLO/budget-calibration-20261015/tree-active-window32-16-local-v160:/root/SpecSLO/budget-calibration-20261015/tree-high-normal-cap4-native-families-v148:$repo:$frozen/source/vllm-hust:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES
ASCEND_RT_VISIBLE_DEVICES=$("$python" -B /root/SpecSLO/budget-calibration-20261015/serial-v1/map-current.py 2,5,6,7)
export TASK_QUEUE_ENABLE=1 HCCL_OP_EXPANSION_MODE=AIV HCCL_DETERMINISTIC=true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export ASCEND_CUSTOM_OPP_PATH=$vendor:$repo/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
export LD_LIBRARY_PATH=$vendor/op_api/lib:$repo/vllm_ascend:$repo/vllm_ascend/lib64:$repo/build/temp.linux-aarch64-cpython-311:${LD_LIBRARY_PATH:-}
export VLLM_ASCEND_PEARL_DRAFT_NZ_MODE=-1
export VLLM_ASCEND_PEARL_TARGET_NZ_MODE=14 VLLM_ASCEND_PEARL_TARGET_LARGE_M_NZ_MIN_ROWS=166
export VLLM_ASCEND_PEARL_TARGET_REPLICATE_INPUT_EMBEDDING=0
export VLLM_ASCEND_PEARL_DRAFT_REPLAY_FIRST_TASK_UPDATE=1
export VLLM_ASCEND_PEARL_DRAFT_STEP_MAJOR_PA_TASK_UPDATE=0
export VLLM_ASCEND_PEARL_DRAFT_PA_TASK_UPDATE_WORKERS=2
export VLLM_ASCEND_PEARL_DRAFT_PA_CONTIGUOUS_PARTITION=0
export VLLM_ASCEND_PEARL_DRAFT_GRAPH_UPDATE_STREAM_PRIORITY=0
export VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE=1
export VLLM_ASCEND_PEARL_TARGET_FIA_TASK_EVENT_GROUP_SIZE=1
export VLLM_ASCEND_PEARL_TARGET_FIA_TASK_PREFIX_EVENT_GROUP_SIZE=0
export VLLM_ASCEND_PEARL_TARGET_FIA_TASK_UPDATE_WORKERS=2
export VLLM_ASCEND_PEARL_TARGET_GRAPH_UPDATE_STREAM_PRIORITY=0
export VLLM_ASCEND_PEARL_SHARED_GRAPH_POOL=1 VLLM_ASCEND_PEARL_MC2_DOWN_PROJ=1
export VLLM_ASCEND_SPECRHYTHM_TREE_GRAPH=1
export VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=1 VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST=0
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_PREFILL=1 VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH=0
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_PREFILL_ONLY=0
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_MAX_TOKENS=645
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_BUCKETS=64,128,192,256,320,384,448,512
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_VERIFY_CAPACITIES=8,16,24,32
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_PROMPT_CAPACITY=4
export VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_MAX_CAPACITY=32
export VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_REQUEST_BUCKET=1
export VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION=1 VLLM_ASCEND_SPECRHYTHM_GLOO_ACCOUNTING=1
export VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL=1 VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL_REUSE_HOST_CONTINUATION=1
export VLLM_ASCEND_SPECRHYTHM_DRAIN_FIFO=1 VLLM_ASCEND_SPECRHYTHM_DRAIN_PADDING_REPAIR=1
export VLLM_ASCEND_SPECRHYTHM_LINEAR_ACCEPTANCE_ONLY=1
export VLLM_ASCEND_SPECRHYTHM_EARLY_CURRENT_TARGET_SUBMIT=1
export VLLM_ASCEND_SPECRHYTHM_EARLY_CURRENT_MIXED_TARGET_SUBMIT=0
export VLLM_ASCEND_SPECRHYTHM_MIXED_DRAFT_PREFILL=0 VLLM_ASCEND_SPECRHYTHM_OVERLAP_DRAFT_PREFILL=0
export VLLM_ASCEND_SPECRHYTHM_PREFETCH_AHEAD=0
export VLLM_ASCEND_SPECSLO_DRAFT_DEVICE_PAGED_ATTENTION=0
unset VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE
unset VLLM_ASCEND_PEARL_SYNC_GRAPH_INPUTS VLLM_ASCEND_PEARL_SYNC_GRAPH_TASK_UPDATE
unset VLLM_ASCEND_PEARL_SYNC_GRAPH_REPLAY VLLM_ASCEND_PEARL_VALIDATE_GRAPH_REPLAYS
unset VLLM_ASCEND_PEARL_NPU_PROFILE_DIR VLLM_ASCEND_PEARL_NPU_PROFILE_RANK
unset VLLM_ASCEND_PEARL_NPU_PROFILE_WAIT_STEPS VLLM_ASCEND_PEARL_NPU_PROFILE_MEASUREMENT_ONLY
cd "$task_root"
exec "$python" -B -u "$repo/examples/run_with_vllm_ascend_extension.py" \
  --extension "$repo/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so" \
  --required-op allreduce_add_rmsnorm_chained \
  "/root/SpecSLO/budget-calibration-20261015/tree-M165-common-mask11-target-math-cost-v323/runtime323.py" --action "$action" --tag "$tag"
