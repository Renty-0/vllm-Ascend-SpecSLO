#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: capture_mc2_real_inputs.sh <attention|down> <run-tag>

Capture real Qwen3-32B TP3 projection inputs for a conservative multi-shape
MC2 qualification envelope.  The harness uses one TP1 draft rank plus three
TP3 target ranks, so exactly four process-free NPUs are required.

Optional environment:
  SPECSLO_VISIBLE_DEVICES  Physical NPUs (default: 0,1,2,3)
  SPECSLO_CAPTURE_ROWS     Exact M values (default: 40..128 step 4)
  SPECSLO_CAPTURE_LAYERS   Decoder layers (default: 0,31,63)
  SPECSLO_CAPTURE_ROOT     Artifact root

This is an enforce-eager diagnostic capture.  It performs no throughput
measurement and never starts a card-reservation process.
EOF
}

if [[ $# -ne 2 || ( $1 != attention && $1 != down ) ]]; then
  usage >&2
  exit 2
fi

capture_kind=$1
run_tag=$2
if [[ ! $run_tag =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid run tag: $run_tag" >&2
  exit 2
fi

repo=/root/data/vllm-ascend-hust
python=/root/miniconda3/envs/vllm-hust-dev/bin/python
cann_env=/usr/local/Ascend/cann-9.0.0/set_env.sh
draft_model=${SPECSLO_DRAFT_MODEL:-/data/shared-models/Qwen3-0.6B}
target_model=${SPECSLO_TARGET_MODEL:-/data/shared-models/Qwen3-32B}
visible_devices=${SPECSLO_VISIBLE_DEVICES:-0,1,2,3}
capture_rows=${SPECSLO_CAPTURE_ROWS:-40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128}
capture_layers=${SPECSLO_CAPTURE_LAYERS:-0,31,63}
capture_root=${SPECSLO_CAPTURE_ROOT:-/root/data/nano-pearl-benchmark-results/20260927-specslo-mc2-multishape-capture}
run_dir=${capture_root}/runs/${capture_kind}-${run_tag}
capture_dir=${run_dir}/captures

for required in \
  "$repo" "$python" "$cann_env" \
  "$draft_model/config.json" "$target_model/config.json" \
  "$target_model/tokenizer_config.json"; do
  if [[ ! -e $required ]]; then
    echo "required path is missing: $required" >&2
    exit 3
  fi
done
if [[ -e $run_dir ]]; then
  echo "refusing to overwrite existing run: $run_dir" >&2
  exit 4
fi

IFS=',' read -r -a devices <<< "$visible_devices"
if [[ ${#devices[@]} -ne 4 ]]; then
  echo "SPECSLO_VISIBLE_DEVICES must contain exactly four NPUs" >&2
  exit 5
fi
declare -A seen_devices=()
for index in "${!devices[@]}"; do
  device=${devices[index]//[[:space:]]/}
  if [[ ! $device =~ ^[0-9]+$ || -n ${seen_devices[$device]:-} ]]; then
    echo "visible NPUs must be unique non-negative integers" >&2
    exit 5
  fi
  devices[index]=$device
  seen_devices[$device]=1
done
visible_devices=$(IFS=,; echo "${devices[*]}")

validate_integer_list() {
  local label=$1 raw=$2 minimum=$3
  local -a values
  local value
  declare -A seen=()
  IFS=',' read -r -a values <<< "$raw"
  if [[ ${#values[@]} -eq 0 ]]; then
    echo "$label must not be empty" >&2
    return 1
  fi
  for value in "${values[@]}"; do
    value=${value//[[:space:]]/}
    if [[ ! $value =~ ^[0-9]+$ || $value -lt $minimum || -n ${seen[$value]:-} ]]; then
      echo "$label must contain unique integers >= $minimum" >&2
      return 1
    fi
    seen[$value]=1
  done
}
validate_integer_list SPECSLO_CAPTURE_ROWS "$capture_rows" 1
validate_integer_list SPECSLO_CAPTURE_LAYERS "$capture_layers" 0
IFS=',' read -r -a row_values <<< "$capture_rows"

mkdir -p "$run_dir"

assert_cards_free() {
  local phase=$1 device snapshot
  npu-smi info > "$run_dir/npu-${phase}.txt"
  for device in "${devices[@]}"; do
    snapshot=$run_dir/npu-${phase}-${device}-proc.txt
    npu-smi info -t proc-mem -i "$device" -c 0 > "$snapshot" 2>&1
    if ! grep -Eiq 'No process in device[.]|No running processes found' "$snapshot"; then
      echo "NPU $device is not process-free; refusing to launch" >&2
      return 1
    fi
  done
}

assert_cards_free initial
cd "$repo"
git rev-parse HEAD > "$run_dir/git-head.txt"
git status --porcelain=v1 --untracked-files=all > "$run_dir/git-status.txt"
git diff --binary > "$run_dir/working-tree.patch"
sha256sum \
  "$draft_model/config.json" "$target_model/config.json" \
  "$target_model/tokenizer_config.json" > "$run_dir/model-config.sha256"

# shellcheck disable=SC1090
source "$cann_env"
export PYTHONPATH=$repo:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=$visible_devices
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_DETERMINISTIC=true
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1
unset OMP_NUM_THREADS OPENBLAS_NUM_THREADS ASCEND_CUSTOM_OPP_PATH
while IFS='=' read -r variable _; do
  case "$variable" in
    VLLM_ASCEND_*) unset "$variable" ;;
  esac
done < <(env)
export VLLM_ASCEND_PEARL_MC2_CAPTURE_DIR=$capture_dir
export VLLM_ASCEND_PEARL_MC2_CAPTURE_ROWS=$capture_rows
export VLLM_ASCEND_PEARL_MC2_CAPTURE_KIND=$capture_kind
export VLLM_ASCEND_PEARL_MC2_CAPTURE_LAYERS=$capture_layers

env | sort > "$run_dir/environment.txt"
{
  printf 'visible_devices=%s\n' "$visible_devices"
  printf 'capture_kind=%s\ncapture_rows=%s\ncapture_layers=%s\n' \
    "$capture_kind" "$capture_rows" "$capture_layers"
  printf 'draft_model=%s\ntarget_model=%s\n' "$draft_model" "$target_model"
} > "$run_dir/contract.env"

assert_cards_free before-launch
set +e
"$python" -u examples/benchmark_nano_pearl_native_target_only.py \
  --draft-model "$draft_model" --target-model "$target_model" \
  --draft-tp-size 1 --target-tp-size 3 --seed 0 \
  --max-model-len 1024 --max-tokens 2 \
  --batch-sizes "${row_values[@]}" --warmup-prompts 1 \
  --gpu-memory-utilization 0.9 --num-kvcache-blocks -1 \
  --worker-timeout-seconds 300 --target-use-paged-attention \
  --enforce-eager --prompt 'Give a short answer.' \
  --output-json "$run_dir/result.json.tmp" > "$run_dir/run.log" 2>&1
status=$?
set -e
npu-smi info > "$run_dir/npu-after.txt" || true
if [[ $status -ne 0 ]]; then
  echo "capture failed with status $status; see $run_dir/run.log" >&2
  exit "$status"
fi
mv "$run_dir/result.json.tmp" "$run_dir/result.json"

IFS=',' read -r -a layer_values <<< "$capture_layers"
for row in "${row_values[@]}"; do
  for layer in "${layer_values[@]}"; do
    for rank in 0 1 2; do
      payload=$capture_dir/$capture_kind/layer-$layer/m$row/rank-$rank.pt
      if [[ ! -s $payload ]]; then
        echo "missing capture payload: $payload" >&2
        exit 7
      fi
    done
  done
done
find "$capture_dir" -type f -name 'rank-*.pt' -print0 \
  | sort -z | xargs -0 sha256sum > "$run_dir/capture-files.sha256"
echo "capture complete: $run_dir"
