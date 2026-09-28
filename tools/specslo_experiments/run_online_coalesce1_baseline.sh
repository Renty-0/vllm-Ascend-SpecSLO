#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <run-tag>" >&2
  exit 2
fi

run_tag=$1
devices=${SPECSLO_VISIBLE_DEVICES:-0,1,2,3}
result_root=${SPECSLO_RESULT_ROOT:-/root/data/nano-pearl-benchmark-results/20260927-specslo-v134-large-m-online-coalesce1}
wrapper=/root/data/nano-pearl-benchmark-results/20260916-qwen3-four-spec-modes-author-workload-slo40-50-150/run_baseline_same_target_cards.sh
IFS=',' read -r -a selected <<< "$devices"
if [[ ${#selected[@]} -ne 4 ]]; then
  echo "exactly four visible NPUs are required" >&2
  exit 2
fi

precheck=${result_root}/baseline-${run_tag}-npu-before.txt
mkdir -p "$result_root"
npu-smi info > "$precheck"
for device in "${selected[@]}"; do
  snapshot=${result_root}/baseline-${run_tag}-npu-${device}-proc-before.txt
  npu-smi info -t proc-mem -i "$device" -c 0 > "$snapshot" 2>&1
  if ! grep -Eiq 'No process in device[.]|No running processes found' "$snapshot"; then
    echo "NPU $device is not process-free; refusing to launch" >&2
    exit 3
  fi
done

unset ASCEND_CUSTOM_OPP_PATH
export SPECSLO_RESULT_ROOT=$result_root
export SPECSLO_VISIBLE_DEVICES=$devices
export SPECSLO_DEVICE_LABEL=cards${devices//,/-}
export SPECSLO_BASELINE_MAX_CUDAGRAPH_CAPTURE_SIZE=640
export SPECSLO_BASELINE_MAX_NUM_BATCHED_TOKENS=640
export SPECSLO_BASELINE_CUDAGRAPH_CAPTURE_SIZES=1,2,4,8,16,24,32,40,48,56,64,128,192,256,320,384,448,512,640
export SPECSLO_BASELINE_REQUIRE_NONUNIFORM_GRAPH=1
export SPECSLO_BASELINE_COPY_GRAPH_INPUTS=1
export SPECSLO_BASELINE_SATURATED_ARRIVALS=0
export SPECSLO_BASELINE_FIXED_OUTPUT_TOKENS=1
# Never reuse a compiled target-only graph from another source snapshot or
# launch contract.  Compile time remains outside the measured online E2E
# window, while the run-local cache prevents stale AOT artifacts from mixing
# meta and NPU tensors during model-runner initialization.
export VLLM_CACHE_ROOT=${SPECSLO_BASELINE_VLLM_CACHE_ROOT:-${result_root}/compile-cache/${run_tag}}
mkdir -p "$VLLM_CACHE_ROOT"

exec "$wrapper" "$run_tag"
