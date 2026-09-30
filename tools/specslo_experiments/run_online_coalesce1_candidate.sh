#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 || ( $1 != off && $1 != on ) ]]; then
  echo "usage: $0 <off|on> <run-tag>" >&2
  exit 2
fi

mc2_mode=$1
run_tag=$2
repo=/root/data/vllm-ascend-hust
python=/root/miniconda3/envs/vllm-hust-dev/bin/python
manifest=/root/data/specslo-workloads/eurosys27-rps4-seed0-author-slo40-50-150/rps4.0_mix0.6_0.2_0.2_seed0.jsonl
result_root=${SPECSLO_RESULT_ROOT:-/root/data/nano-pearl-benchmark-results/20260927-specslo-v134-large-m-online-coalesce1}
visible_devices=${SPECSLO_VISIBLE_DEVICES:-0,1,2,3}
profile=${SPECSLO_MC2_PROFILE:-/root/data/nano-pearl-benchmark-results/20260927-specslo-mc2-v134-large-m-qualification/runs/production-current-target345-v134-large-m-s31/qwen3-32b-tp3-v134-large-m-production-profile.json}
vendor=${SPECSLO_MC2_VENDOR:-/root/data/specslo-mc2-v134-large-m/package-extracted/packages/vendors/specslo_mc2_v134_large_m_transformer}
base_vendor=${repo}/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
extension=${repo}/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
torch_lib=/root/miniconda3/envs/vllm-hust-dev/lib/python3.11/site-packages/torch/lib
torch_npu_lib=/root/miniconda3/envs/vllm-hust-dev/lib/python3.11/site-packages/torch_npu/lib

fallback_max_batch=${SPECSLO_TARGET_FALLBACK_MAX_BATCH:-4}
acceptance_only=${SPECSLO_LINEAR_ACCEPTANCE_ONLY:-1}
acceptance_floor=${SPECSLO_ACCEPTANCE_FLOOR:-0.4}
linear_bonus_token=${SPECSLO_LINEAR_BONUS_TOKEN:-0}
prefill_only=${SPECSLO_MIXED_TARGET_GRAPH_PREFILL_ONLY:-0}
residual_tight_only=${SPECSLO_RESIDUAL_EAGER_TIGHT_ONLY:-0}
host_continuation_reuse=${SPECSLO_GLOO_PROPOSAL_REUSE_HOST_CONTINUATION:-1}
drain_fifo=${SPECSLO_DRAIN_FIFO:-1}
drain_lrtf=${SPECSLO_DRAIN_LRTF:-0}
drain_lrtf_max_active=${SPECSLO_DRAIN_LRTF_MAX_ACTIVE:-0}
drain_target_lrtf=${SPECSLO_DRAIN_TARGET_LRTF:-0}
drain_padding_repair=${SPECSLO_DRAIN_PADDING_REPAIR:-0}
target_residual_lrtf=${SPECSLO_TARGET_RESIDUAL_LRTF:-0}
graph_entries=${SPECSLO_MAX_ACLGRAPH_ENTRIES:-80}
kv_blocks=${SPECSLO_NUM_KVCACHE_BLOCKS:-544}
mixed_target_graph_buckets=${SPECSLO_MIXED_TARGET_GRAPH_BUCKETS:-64,128,192,256,320,384,448,512}
profile_host_decode_steps=${SPECSLO_PROFILE_HOST_DECODE_STEPS:-0}
profile_wait_steps=${SPECSLO_NPU_PROFILE_WAIT_STEPS:-16}
enable_npu_profile=${SPECSLO_ENABLE_NPU_PROFILE:-1}
target_fia_task_update_workers=${SPECSLO_TARGET_FIA_TASK_UPDATE_WORKERS:-1}
target_fia_task_event_group_size=${SPECSLO_TARGET_FIA_TASK_EVENT_GROUP_SIZE:-1}
target_fia_task_prefix_event_group_size=${SPECSLO_TARGET_FIA_TASK_PREFIX_EVENT_GROUP_SIZE:-0}
draft_pa_task_update_workers=${SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS:-1}
draft_device_paged_attention=${SPECSLO_DRAFT_DEVICE_PAGED_ATTENTION:-0}
draft_pa_contiguous_partition=${SPECSLO_DRAFT_PA_CONTIGUOUS_PARTITION:-0}
target_replay_ahead=${SPECSLO_TARGET_REPLAY_AHEAD:-0}
early_current_target_submit=${SPECSLO_EARLY_CURRENT_TARGET_SUBMIT:-0}
early_current_mixed_target_submit=${SPECSLO_EARLY_CURRENT_MIXED_TARGET_SUBMIT:-0}
prefetch_ahead=${SPECSLO_PREFETCH_AHEAD:-0}
stable_target_request_bucket=${SPECSLO_STABLE_TARGET_REQUEST_BUCKET:-1}
target_max_batch=${SPECSLO_MAX_TARGET_BATCH:-32}
online_target_max_batch=${SPECSLO_ONLINE_TARGET_MAX_BATCH:-0}
stable_target_dense_max_capacity=${SPECSLO_STABLE_TARGET_DENSE_MAX_CAPACITY:-0}
target_nz_mode=${SPECSLO_TARGET_NZ_MODE:-default}
target_large_m_nz_min_rows=${SPECSLO_TARGET_LARGE_M_NZ_MIN_ROWS:-192}
target_replicate_input_embedding=${SPECSLO_TARGET_REPLICATE_INPUT_EMBEDDING:-0}
npu_alloc_conf=${SPECSLO_NPU_ALLOC_CONF:-default}
validate_only=${SPECSLO_VALIDATE_ONLY:-0}
case "$target_max_batch" in
  32|36|40|48|64)
    # Mixed verification+prefill graphs are much larger than decode-only
    # target graphs.  Keep their proven 32-row family resident and let the
    # scheduler cap only cycles that actually carry staged prompt work.
    mixed_target_verify_capacities=8,16,24,32
    ;;
  *)
    echo "SPECSLO_MAX_TARGET_BATCH must be 32, 36, 40, 48, or 64" >&2
    exit 2
    ;;
esac
if [[ ! $online_target_max_batch =~ ^[0-9]+$ ]] || (( online_target_max_batch > target_max_batch )); then
  echo "SPECSLO_ONLINE_TARGET_MAX_BATCH must be an integer in [0, SPECSLO_MAX_TARGET_BATCH]" >&2
  exit 2
fi
if [[ ! $stable_target_dense_max_capacity =~ ^[0-9]+$ ]] || \
  (( stable_target_dense_max_capacity > target_max_batch )) || \
  (( stable_target_dense_max_capacity > 0 && stable_target_dense_max_capacity < stable_target_request_bucket )); then
  echo "SPECSLO_STABLE_TARGET_DENSE_MAX_CAPACITY must be zero or in [request bucket, target max batch]" >&2
  exit 2
fi
# Widening decode-only target cohorts does not widen the resident mixed
# verification+prefill family.  Its historical envelope therefore remains
# 512 prompt tokens plus 32 gamma4 rows and the fixed bookkeeping margin.
mixed_target_graph_max_tokens=$((645 + 32 * linear_bonus_token))

if [[ ! $run_tag =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid run tag: $run_tag" >&2
  exit 2
fi
if [[ ! $acceptance_floor =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]]; then
  echo "SPECSLO_ACCEPTANCE_FLOOR must be a decimal in [0, 1]" >&2
  exit 2
fi
IFS=',' read -r -a devices <<< "$visible_devices"
if [[ ${#devices[@]} -ne 4 ]]; then
  echo "exactly four visible NPUs are required" >&2
  exit 2
fi
for value in "$acceptance_only" "$linear_bonus_token" "$prefill_only" "$residual_tight_only" \
  "$host_continuation_reuse" "$drain_fifo" "$drain_lrtf" \
  "$drain_target_lrtf" "$drain_padding_repair" \
  "$target_residual_lrtf" \
  "$enable_npu_profile" \
  "$draft_device_paged_attention" \
  "$draft_pa_contiguous_partition" \
  "$target_replay_ahead" \
  "$early_current_target_submit" "$early_current_mixed_target_submit" \
  "$prefetch_ahead" "$validate_only"; do
  if [[ $value != 0 && $value != 1 ]]; then
    echo "boolean experiment switches must be 0 or 1" >&2
    exit 2
  fi
done
if [[ $target_replicate_input_embedding != 0 && $target_replicate_input_embedding != 1 ]]; then
  echo "SPECSLO_TARGET_REPLICATE_INPUT_EMBEDDING must be 0 or 1" >&2
  exit 2
fi
if [[ $drain_lrtf == 1 && $drain_fifo != 1 ]]; then
  echo "SPECSLO_DRAIN_LRTF=1 requires SPECSLO_DRAIN_FIFO=1" >&2
  exit 2
fi
if [[ ! $drain_lrtf_max_active =~ ^[0-9]+$ ]] || (( drain_lrtf_max_active > 64 )); then
  echo "SPECSLO_DRAIN_LRTF_MAX_ACTIVE must be in [0, 64]" >&2
  exit 2
fi
if [[ $drain_lrtf == 0 && $drain_lrtf_max_active != 0 ]]; then
  echo "SPECSLO_DRAIN_LRTF_MAX_ACTIVE requires SPECSLO_DRAIN_LRTF=1" >&2
  exit 2
fi
if [[ $drain_target_lrtf == 1 && $drain_fifo != 1 ]]; then
  echo "SPECSLO_DRAIN_TARGET_LRTF=1 requires SPECSLO_DRAIN_FIFO=1" >&2
  exit 2
fi
if [[ $drain_padding_repair == 1 && $drain_fifo != 1 ]]; then
  echo "SPECSLO_DRAIN_PADDING_REPAIR=1 requires SPECSLO_DRAIN_FIFO=1" >&2
  exit 2
fi
if [[ $drain_target_lrtf == 1 && $drain_lrtf == 1 ]]; then
  echo "unified and target-only drain LRTF are mutually exclusive" >&2
  exit 2
fi
if [[ $target_nz_mode != default && $target_nz_mode != 5 && \
      $target_nz_mode != 9 && \
      $target_nz_mode != 11 && $target_nz_mode != 12 && \
      $target_nz_mode != 13 && $target_nz_mode != 14 && \
      $target_nz_mode != 15 ]]; then
  echo "SPECSLO_TARGET_NZ_MODE must be default, 5, 9, 11, 12, 13, 14, or 15" >&2
  exit 2
fi
if [[ $stable_target_request_bucket != 1 && \
      $stable_target_request_bucket != 2 && \
      $stable_target_request_bucket != 4 && \
      $stable_target_request_bucket != 8 ]]; then
  echo "SPECSLO_STABLE_TARGET_REQUEST_BUCKET must be 1, 2, 4, or 8" >&2
  exit 2
fi
if [[ $mixed_target_graph_buckets != 64,128,192,256,320,384,448,512 && \
      $mixed_target_graph_buckets != 128,256,384,512 ]]; then
  echo "SPECSLO_MIXED_TARGET_GRAPH_BUCKETS must be the production or coarse four-bucket family" >&2
  exit 2
fi
if [[ ! $target_large_m_nz_min_rows =~ ^[0-9]+$ ]] || \
   (( target_large_m_nz_min_rows < 1 || target_large_m_nz_min_rows > 4096 )); then
  echo "SPECSLO_TARGET_LARGE_M_NZ_MIN_ROWS must be in [1, 4096]" >&2
  exit 2
fi
if [[ $npu_alloc_conf != default && \
      ! $npu_alloc_conf =~ ^max_split_size_mb:[0-9]+$ ]]; then
  echo "SPECSLO_NPU_ALLOC_CONF must be default or max_split_size_mb:<MiB>" >&2
  exit 2
fi
for value in "$profile_host_decode_steps" "$profile_wait_steps"; do
  if [[ ! $value =~ ^[0-9]+$ ]]; then
    echo "profiling step counts must be non-negative integers" >&2
    exit 2
  fi
done
if [[ $target_fia_task_update_workers != 1 && \
      $target_fia_task_update_workers != 2 && \
      $target_fia_task_update_workers != 3 && \
      $target_fia_task_update_workers != 4 ]]; then
  echo "SPECSLO_TARGET_FIA_TASK_UPDATE_WORKERS must be 1, 2, 3, or 4" >&2
  exit 2
fi
if [[ $target_fia_task_event_group_size != 1 && \
      $target_fia_task_event_group_size != 2 && \
      $target_fia_task_event_group_size != 4 ]]; then
  echo "SPECSLO_TARGET_FIA_TASK_EVENT_GROUP_SIZE must be 1, 2, or 4" >&2
  exit 2
fi
if [[ $target_fia_task_prefix_event_group_size != 0 && \
      $target_fia_task_prefix_event_group_size != 1 && \
      $target_fia_task_prefix_event_group_size != 2 ]]; then
  echo "SPECSLO_TARGET_FIA_TASK_PREFIX_EVENT_GROUP_SIZE must be 0, 1, or 2" >&2
  exit 2
fi
if (( target_fia_task_prefix_event_group_size >= target_fia_task_event_group_size )) && \
   (( target_fia_task_prefix_event_group_size != 0 )); then
  echo "SPECSLO_TARGET_FIA_TASK_PREFIX_EVENT_GROUP_SIZE must be smaller than the steady event group" >&2
  exit 2
fi
if [[ $draft_pa_task_update_workers != 1 && \
      $draft_pa_task_update_workers != 2 && \
      $draft_pa_task_update_workers != 4 ]]; then
  echo "SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS must be 1, 2, or 4" >&2
  exit 2
fi
for required in "$repo" "$python" "$manifest" "$profile" "$vendor" "$extension"; do
  if [[ ! -e $required ]]; then
    echo "required path is missing: $required" >&2
    exit 2
  fi
done
if [[ ! -d $vendor || ! -f $vendor/op_api/lib/libcust_opapi.so ]]; then
  echo "MC2 vendor is not an installed custom-op tree: $vendor" >&2
  exit 2
fi
if [[ $mc2_mode == on ]] && ! nm -D "$vendor/op_api/lib/libcust_opapi.so" 2>/dev/null \
  | grep -q 'aclnnAllreduceAddRmsnormGetWorkspaceSize'; then
  echo "MC2 vendor does not provide aclnnAllreduceAddRmsnorm: $vendor" >&2
  exit 2
fi

run_dir=${result_root}/runs/candidate-${mc2_mode}-${run_tag}
if [[ $validate_only == 1 ]]; then
  if [[ ! -f $run_dir/result.json ]]; then
    echo "validation result is missing: $run_dir/result.json" >&2
    exit 2
  fi
else
  if [[ -e $run_dir ]]; then
    echo "refusing to overwrite existing run: $run_dir" >&2
    exit 3
  fi
  mkdir -p "$run_dir"
fi

assert_cards_free() {
  npu-smi info > "$run_dir/npu-before.txt"
  local device snapshot
  for device in "${devices[@]}"; do
    snapshot=$run_dir/npu-${device}-proc-before.txt
    npu-smi info -t proc-mem -i "$device" -c 0 > "$snapshot" 2>&1
    if ! grep -Eiq 'No process in device[.]|No running processes found' "$snapshot"; then
      echo "NPU $device is not process-free; refusing to launch" >&2
      return 1
    fi
  done
}
if [[ $validate_only == 0 ]]; then
  assert_cards_free
fi
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH=$repo:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=$visible_devices
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_DETERMINISTIC=true
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
if [[ $npu_alloc_conf == default ]]; then
  unset PYTORCH_NPU_ALLOC_CONF
else
  export PYTORCH_NPU_ALLOC_CONF=$npu_alloc_conf
fi
export ASCEND_CUSTOM_OPP_PATH=$vendor:$base_vendor
export LD_LIBRARY_PATH=$vendor/op_api/lib:$repo/vllm_ascend:$repo/vllm_ascend/lib64:$repo/build/temp.linux-aarch64-cpython-311:$torch_lib:$torch_npu_lib:/usr/local/Ascend/cann-9.0.0/lib64:${LD_LIBRARY_PATH:-}

# The formal performance rounds leave SPECSLO_PROFILE_HOST_DECODE_STEPS at
# zero, so inherited shell state cannot accidentally turn on CANN profiling.
# A separate diagnostic run opts in explicitly.  Measurement-only mode arms
# the profiler after graph qualification when the benchmark configures its
# non-synchronizing host timeline for the final generate call.
profile_args=()
npu_profile_dir=
if (( profile_host_decode_steps > 0 && enable_npu_profile == 1 )); then
  npu_profile_dir=$run_dir/npu-profile
  mkdir -p "$npu_profile_dir"
  export VLLM_ASCEND_PEARL_NPU_PROFILE_DIR=$npu_profile_dir
  export VLLM_ASCEND_PEARL_NPU_PROFILE_RANK=-1
  export VLLM_ASCEND_PEARL_NPU_PROFILE_WAIT_STEPS=$profile_wait_steps
  export VLLM_ASCEND_PEARL_NPU_PROFILE_MEASUREMENT_ONLY=1
  profile_args=(--profile-host-decode-steps "$profile_host_decode_steps")
else
  unset VLLM_ASCEND_PEARL_NPU_PROFILE_DIR
  unset VLLM_ASCEND_PEARL_NPU_PROFILE_RANK
  unset VLLM_ASCEND_PEARL_NPU_PROFILE_WAIT_STEPS
  unset VLLM_ASCEND_PEARL_NPU_PROFILE_MEASUREMENT_ONLY
fi
if (( profile_host_decode_steps > 0 )); then
  profile_args=(--profile-host-decode-steps "$profile_host_decode_steps")
fi

export VLLM_ASCEND_PEARL_DRAFT_NZ_MODE=-1
export VLLM_ASCEND_PEARL_TARGET_LARGE_M_NZ_MIN_ROWS=$target_large_m_nz_min_rows
export VLLM_ASCEND_PEARL_TARGET_REPLICATE_INPUT_EMBEDDING=$target_replicate_input_embedding
if [[ $target_nz_mode == default ]]; then
  unset VLLM_ASCEND_PEARL_TARGET_NZ_MODE
else
  export VLLM_ASCEND_PEARL_TARGET_NZ_MODE=$target_nz_mode
fi
unset VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE
export VLLM_ASCEND_PEARL_DRAFT_REPLAY_FIRST_TASK_UPDATE=1
export VLLM_ASCEND_PEARL_DRAFT_STEP_MAJOR_PA_TASK_UPDATE=0
# Overwrite the runtime variable even at the production default so a parent
# shell cannot silently contaminate the candidate contract.
export VLLM_ASCEND_PEARL_DRAFT_PA_TASK_UPDATE_WORKERS=$draft_pa_task_update_workers
export VLLM_ASCEND_PEARL_DRAFT_PA_CONTIGUOUS_PARTITION=$draft_pa_contiguous_partition
export VLLM_ASCEND_PEARL_DRAFT_GRAPH_UPDATE_STREAM_PRIORITY=0
export VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE=1
export VLLM_ASCEND_PEARL_TARGET_FIA_TASK_EVENT_GROUP_SIZE=$target_fia_task_event_group_size
export VLLM_ASCEND_PEARL_TARGET_FIA_TASK_PREFIX_EVENT_GROUP_SIZE=$target_fia_task_prefix_event_group_size
# Always overwrite inherited runtime state.  The production default remains
# one; workers=2/4 are explicit experiments whose measured parallel path must
# pass the post-warmup evidence gate below.
export VLLM_ASCEND_PEARL_TARGET_FIA_TASK_UPDATE_WORKERS=$target_fia_task_update_workers
export VLLM_ASCEND_PEARL_TARGET_GRAPH_UPDATE_STREAM_PRIORITY=0
export VLLM_ASCEND_PEARL_SHARED_GRAPH_POOL=1
# The production profile qualifies both the attention output projection and
# the FFN down projection.  Freeze the latter on in the launcher instead of
# inheriting its fail-closed process default; otherwise an apparently valid
# "MC2-on" campaign would silently benchmark only half of the qualified
# target path.
export VLLM_ASCEND_PEARL_MC2_DOWN_PROJ=1
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_PREFILL=1
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH=1
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_PREFILL_ONLY=$prefill_only
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_MAX_TOKENS=$mixed_target_graph_max_tokens
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_BUCKETS=$mixed_target_graph_buckets
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_VERIFY_CAPACITIES=$mixed_target_verify_capacities
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_PROMPT_CAPACITY=4
export VLLM_ASCEND_SPECRHYTHM_PREFETCH_AHEAD=$prefetch_ahead
export VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_MAX_CAPACITY=$target_max_batch
export VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_DENSE_MAX_CAPACITY=$stable_target_dense_max_capacity
export VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_REQUEST_BUCKET=$stable_target_request_bucket
export VLLM_ASCEND_SPECRHYTHM_MIXED_DRAFT_PREFILL=0
export VLLM_ASCEND_SPECRHYTHM_OVERLAP_DRAFT_PREFILL=0
export VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION=1
export VLLM_ASCEND_SPECRHYTHM_GLOO_ACCOUNTING=1
export VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL=1
export VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL_REUSE_HOST_CONTINUATION=$host_continuation_reuse
export VLLM_ASCEND_SPECRHYTHM_GOODPUT_EDF=0
export VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST=0
export VLLM_ASCEND_SPECRHYTHM_DRAIN_FIFO=$drain_fifo
export VLLM_ASCEND_SPECRHYTHM_ONLINE_TARGET_MAX_BATCH=$online_target_max_batch
export VLLM_ASCEND_SPECRHYTHM_DRAIN_LRTF=$drain_lrtf
export VLLM_ASCEND_SPECRHYTHM_DRAIN_LRTF_MAX_ACTIVE=$drain_lrtf_max_active
export VLLM_ASCEND_SPECRHYTHM_DRAIN_TARGET_LRTF=$drain_target_lrtf
export VLLM_ASCEND_SPECRHYTHM_DRAIN_PADDING_REPAIR=$drain_padding_repair
export VLLM_ASCEND_SPECRHYTHM_TARGET_RESIDUAL_LRTF=$target_residual_lrtf
export VLLM_ASCEND_SPECRHYTHM_TARGET_REPLAY_AHEAD=$target_replay_ahead
export VLLM_ASCEND_SPECRHYTHM_EARLY_CURRENT_TARGET_SUBMIT=$early_current_target_submit
export VLLM_ASCEND_SPECRHYTHM_EARLY_CURRENT_MIXED_TARGET_SUBMIT=$early_current_mixed_target_submit
export VLLM_ASCEND_SPECRHYTHM_RESIDUAL_EAGER_TIGHT_ONLY=$residual_tight_only
export VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=0
export VLLM_ASCEND_SPECRHYTHM_ARRIVAL_DEBT_CAP_TOKENS=0
export VLLM_ASCEND_SPECRHYTHM_NORMAL_ACTIVE_CAP=0
export VLLM_ASCEND_SPECRHYTHM_LOOSE_ACTIVE_CAP=0
export VLLM_ASCEND_SPECRHYTHM_PREEMPT_LOOSE_FOR_TIGHT=0
export VLLM_ASCEND_SPECRHYTHM_TIGHT_TARGET_BURST=0
export VLLM_ASCEND_SPECRHYTHM_TIGHT_HOME_FRACTION=0.5
export VLLM_ASCEND_SPECRHYTHM_LINEAR_ACCEPTANCE_ONLY=$acceptance_only
unset VLLM_ASCEND_PEARL_INLINE_GRAPH_TASK_UPDATE
unset VLLM_ASCEND_PEARL_SYNC_GRAPH_INPUTS
unset VLLM_ASCEND_PEARL_SYNC_GRAPH_TASK_UPDATE
unset VLLM_ASCEND_PEARL_SYNC_GRAPH_REPLAY
unset VLLM_ASCEND_PEARL_VALIDATE_GRAPH_REPLAYS
export VLLM_ASCEND_SPECSLO_DRAFT_DEVICE_PAGED_ATTENTION=$draft_device_paged_attention
unset VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET
unset VLLM_ASCEND_SPECRHYTHM_FORCE_STEPWISE_TARGET
unset VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH
unset VLLM_ASCEND_SPECRHYTHM_STEPWISE_TARGET_FIA

mc2_args=()
if [[ $mc2_mode == on ]]; then
  mc2_args=(--enable-mc2 --mc2-profile "$profile")
fi
bonus_args=()
if [[ $linear_bonus_token == 1 ]]; then
  bonus_args=(--spec-rhythm-linear-bonus-token)
fi

if [[ $validate_only == 0 ]]; then
  cd "$repo"
  git rev-parse HEAD > "$run_dir/git-head.txt"
  git diff --binary > "$run_dir/working-tree.patch"
  git status --porcelain=v1 --untracked-files=all > "$run_dir/git-status.txt"
  sha256sum "$manifest" "$profile" "$extension" > "$run_dir/input.sha256"
  {
  printf 'mode=%s\nrun_tag=%s\nvisible_devices=%s\n' "$mc2_mode" "$run_tag" "$visible_devices"
  printf 'result_root=%s\nrun_dir=%s\n' "$result_root" "$run_dir"
  printf 'fallback_max_batch=%s\nacceptance_only=%s\nacceptance_floor=%s\nlinear_bonus_token=%s\nprefill_only=%s\n' \
    "$fallback_max_batch" "$acceptance_only" "$acceptance_floor" "$linear_bonus_token" "$prefill_only"
  printf 'mixed_target_graph_max_tokens=%s\n' "$mixed_target_graph_max_tokens"
  printf 'mixed_target_graph_buckets=%s\n' "$mixed_target_graph_buckets"
  printf 'residual_tight_only=%s\ngraph_entries=%s\nkv_blocks=%s\n' "$residual_tight_only" "$graph_entries" "$kv_blocks"
  printf 'host_continuation_reuse=%s\ndrain_fifo=%s\ndrain_lrtf=%s\n' \
    "$host_continuation_reuse" "$drain_fifo" "$drain_lrtf"
  printf 'drain_lrtf_max_active=%s\n' "$drain_lrtf_max_active"
  printf 'drain_target_lrtf=%s\n' "$drain_target_lrtf"
  printf 'drain_padding_repair=%s\n' "$drain_padding_repair"
  printf 'target_residual_lrtf=%s\n' "$target_residual_lrtf"
  printf 'profile_host_decode_steps=%s\nnpu_profile_wait_steps=%s\nenable_npu_profile=%s\n' \
    "$profile_host_decode_steps" "$profile_wait_steps" "$enable_npu_profile"
  printf 'target_fia_task_update_workers=%s\n' "$target_fia_task_update_workers"
  printf 'target_fia_task_event_group_size=%s\n' "$target_fia_task_event_group_size"
  printf 'target_fia_task_prefix_event_group_size=%s\n' "$target_fia_task_prefix_event_group_size"
  printf 'draft_pa_task_update_workers=%s\n' "$draft_pa_task_update_workers"
  printf 'draft_device_paged_attention=%s\n' "$draft_device_paged_attention"
  printf 'draft_pa_contiguous_partition=%s\n' "$draft_pa_contiguous_partition"
  printf 'target_replay_ahead=%s\n' "$target_replay_ahead"
  printf 'early_current_target_submit=%s\n' "$early_current_target_submit"
  printf 'early_current_mixed_target_submit=%s\n' "$early_current_mixed_target_submit"
  printf 'prefetch_ahead=%s\n' "$prefetch_ahead"
  printf 'stable_target_request_bucket=%s\n' "$stable_target_request_bucket"
  printf 'target_max_batch=%s\nmixed_target_verify_capacities=%s\n' \
    "$target_max_batch" "$mixed_target_verify_capacities"
  printf 'online_target_max_batch=%s\n' "$online_target_max_batch"
  printf 'stable_target_dense_max_capacity=%s\n' "$stable_target_dense_max_capacity"
  printf 'target_nz_mode=%s\ntarget_large_m_nz_min_rows=%s\n' \
    "$target_nz_mode" "$target_large_m_nz_min_rows"
  printf 'target_replicate_input_embedding=%s\n' "$target_replicate_input_embedding"
  printf 'npu_alloc_conf=%s\n' "$npu_alloc_conf"
  printf 'npu_profile_measurement_only=%s\nnpu_profile_dir=%s\n' \
    "$(( profile_host_decode_steps > 0 && enable_npu_profile == 1 ))" "$npu_profile_dir"
  printf 'mc2_down_proj=1\n'
  printf 'profile=%s\nvendor=%s\n' "$profile" "$vendor"
  } > "$run_dir/contract.env"

  timeout 2400 "$python" -u examples/run_with_vllm_ascend_extension.py \
  --extension "$extension" --required-op allreduce_add_rmsnorm_chained \
  examples/benchmark_nano_pearl_speculative.py \
  --draft-model /data/shared-models/Qwen3-0.6B \
  --target-model /data/shared-models/Qwen3-32B \
  --draft-tp-size 1 --target-tp-size 3 --draft-mode serial_linear \
  --seed 0 --gamma 4 --spec-rhythm-min-gamma 4 \
  --spec-rhythm-request-max-gamma 4 --spec-rhythm-linear-full-window \
  "${bonus_args[@]}" \
  --spec-rhythm-merge-ready-homes --spec-rhythm-linear-eager-cross-graph-bucket \
  --spec-rhythm-linear-idle-residual-eager \
  --spec-rhythm-target-fallback-max-batch "$fallback_max_batch" \
  --spec-rhythm-tree-width 1 --spec-rhythm-tree-depth 1 \
  --spec-rhythm-max-eager-tokens 4 --max-model-len 4096 --max-tokens 256 \
  --batch-sizes 64 --num-prompts 463 --request-manifest "$manifest" \
  --gpu-memory-utilization 0.8 --num-kvcache-blocks "$kv_blocks" \
  --max-aclgraph-entries "$graph_entries" --target-verification-graph-buckets 8 \
  --enable-continuous-batching --enable-preemptive-scheduling --enable-spec-rhythm \
  --spec-rhythm-priority-burst 2 --spec-rhythm-acceptance-floor "$acceptance_floor" \
  --spec-rhythm-online-prefill --spec-rhythm-prefill-coalesce-min-requests 1 \
  --spec-rhythm-prefill-coalesce-max-wait-ms 0 --spec-rhythm-prefill-token-chunk-size 512 \
  --precompile-serial-draft-graphs --spec-rhythm-stable-graphs \
  --require-no-graph-fallback --draft-use-paged-attention \
  --spec-rhythm-max-target-batch "$target_max_batch" "${mc2_args[@]}" \
  --warmup-runs 1 --graph-qualification-max-runs 8 \
  "${profile_args[@]}" \
  --worker-timeout-seconds 1800 --output-json "$run_dir/result.json.tmp" \
    > "$run_dir/run.log" 2>&1
  mv "$run_dir/result.json.tmp" "$run_dir/result.json"
fi

"$python" - "$run_dir/result.json" "$run_dir/gate.json" "$mc2_mode" \
  "$fallback_max_batch" "$prefill_only" "$residual_tight_only" \
  "$acceptance_only" "$host_continuation_reuse" "$drain_fifo" "$drain_lrtf" \
  "$target_fia_task_update_workers" "$target_fia_task_event_group_size" \
  "$target_fia_task_prefix_event_group_size" "$draft_pa_task_update_workers" \
  "$draft_device_paged_attention" \
  "$draft_pa_contiguous_partition" \
  "$target_replay_ahead" "$early_current_target_submit" \
  "$early_current_mixed_target_submit" "$prefetch_ahead" \
  "$stable_target_request_bucket" "$target_nz_mode" \
  "$target_large_m_nz_min_rows" "$npu_alloc_conf" "$profile" \
  "$drain_target_lrtf" "$target_residual_lrtf" "$drain_lrtf_max_active" \
  "$linear_bonus_token" "$acceptance_floor" "$target_max_batch" \
  "$mixed_target_verify_capacities" "$target_replicate_input_embedding" \
  "$mixed_target_graph_buckets" "$online_target_max_batch" \
  "$stable_target_dense_max_capacity" "$drain_padding_repair" <<'PY'
import json
import sys
from pathlib import Path

from examples.benchmark_nano_pearl_speculative import (
    _host_continuation_measurement_issues,
)
from examples.specslo_benchmark_report import (
    draft_pa_parallel_update_evidence,
    measure_qualified_mc2_replay_coverage,
    qualified_mc2_profile_replay_contract,
    target_fia_parallel_update_evidence,
    target_replay_ahead_evidence,
)

result_path, gate_path = map(Path, sys.argv[1:3])
mode = sys.argv[3]
expected_fallback = int(sys.argv[4])
expected_prefill_only = bool(int(sys.argv[5]))
expected_residual_tight_only = bool(int(sys.argv[6]))
expected_acceptance_only = bool(int(sys.argv[7]))
expected_host_continuation_reuse = bool(int(sys.argv[8]))
expected_drain_fifo = bool(int(sys.argv[9]))
expected_drain_lrtf = bool(int(sys.argv[10]))
expected_target_fia_task_update_workers = int(sys.argv[11])
expected_target_fia_task_event_group_size = int(sys.argv[12])
expected_target_fia_task_prefix_event_group_size = int(sys.argv[13])
expected_draft_pa_task_update_workers = int(sys.argv[14])
expected_draft_device_paged_attention = bool(int(sys.argv[15]))
expected_draft_pa_contiguous_partition = bool(int(sys.argv[16]))
expected_target_replay_ahead = bool(int(sys.argv[17]))
expected_early_current_target_submit = bool(int(sys.argv[18]))
expected_early_current_mixed_target_submit = bool(int(sys.argv[19]))
expected_prefetch_ahead = bool(int(sys.argv[20]))
expected_stable_target_request_bucket = int(sys.argv[21])
expected_target_nz_mode = sys.argv[22]
expected_target_large_m_nz_min_rows = int(sys.argv[23])
expected_npu_alloc_conf = sys.argv[24]
profile_path = Path(sys.argv[25]).resolve()
expected_drain_target_lrtf = bool(int(sys.argv[26]))
expected_target_residual_lrtf = bool(int(sys.argv[27]))
expected_drain_lrtf_max_active = int(sys.argv[28])
expected_linear_bonus_token = bool(int(sys.argv[29]))
expected_acceptance_floor = float(sys.argv[30])
expected_target_max_batch = int(sys.argv[31])
expected_mixed_target_verify_capacities = sys.argv[32]
expected_target_replicate_input_embedding = bool(int(sys.argv[33]))
expected_mixed_target_graph_buckets = sys.argv[34]
expected_online_target_max_batch = int(sys.argv[35])
expected_stable_target_dense_max_capacity = int(sys.argv[36])
expected_drain_padding_repair = bool(int(sys.argv[37]))
payload = json.loads(result_path.read_text(encoding="utf-8"))
row = payload["results"][0]
environment = payload.get("runtime_environment", {})
errors = []
if payload.get("backend") != "specslo-native-specrhythm":
    errors.append("unexpected backend")
if payload.get("spec_rhythm_prefill_coalesce_min_requests") != 1:
    errors.append("not coalesce1")
if bool(payload.get("spec_rhythm_linear_bonus_token", False)) != expected_linear_bonus_token:
    errors.append("linear bonus-token contract mismatch")
if float(payload.get("spec_rhythm_acceptance_floor", -1.0)) != expected_acceptance_floor:
    errors.append("acceptance-floor contract mismatch")
expected_mixed_target_graph_max_tokens = 645 + 32 * int(expected_linear_bonus_token)
if environment.get("VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_MAX_TOKENS") != str(
    expected_mixed_target_graph_max_tokens
):
    errors.append("mixed-target graph token-capacity contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_BUCKETS") != (
    expected_mixed_target_graph_buckets
):
    errors.append("mixed-target graph bucket contract mismatch")
if payload.get("spec_rhythm_target_fallback_max_batch") != expected_fallback:
    errors.append("target fallback contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_PREFILL_ONLY") != str(int(expected_prefill_only)):
    errors.append("mixed-target prefill-only contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_RESIDUAL_EAGER_TIGHT_ONLY") != str(int(expected_residual_tight_only)):
    errors.append("residual-eager tight-only contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_LINEAR_ACCEPTANCE_ONLY") != str(int(expected_acceptance_only)):
    errors.append("linear acceptance-only contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_DRAIN_FIFO") != str(int(expected_drain_fifo)):
    errors.append("drain-FIFO contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_DRAIN_PADDING_REPAIR") != str(
    int(expected_drain_padding_repair)
):
    errors.append("drain-padding-repair contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_DRAIN_LRTF") != str(int(expected_drain_lrtf)):
    errors.append("drain-LRTF contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_DRAIN_LRTF_MAX_ACTIVE") != str(
    expected_drain_lrtf_max_active
):
    errors.append("drain-LRTF active guard contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_DRAIN_TARGET_LRTF") != str(
    int(expected_drain_target_lrtf)
):
    errors.append("drain target-LRTF contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_TARGET_RESIDUAL_LRTF") != str(
    int(expected_target_residual_lrtf)
):
    errors.append("target residual-LRTF contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL_REUSE_HOST_CONTINUATION") != str(int(expected_host_continuation_reuse)):
    errors.append("host-continuation reuse contract mismatch")
if environment.get("VLLM_ASCEND_PEARL_TARGET_FIA_TASK_UPDATE_WORKERS") != str(expected_target_fia_task_update_workers):
    errors.append("target-FIA task-update worker contract mismatch")
if environment.get("VLLM_ASCEND_PEARL_TARGET_FIA_TASK_EVENT_GROUP_SIZE") != str(
    expected_target_fia_task_event_group_size
):
    errors.append("target-FIA task event-group contract mismatch")
if environment.get("VLLM_ASCEND_PEARL_TARGET_FIA_TASK_PREFIX_EVENT_GROUP_SIZE") != str(
    expected_target_fia_task_prefix_event_group_size
):
    errors.append("target-FIA task prefix event-group contract mismatch")
if environment.get("VLLM_ASCEND_PEARL_DRAFT_PA_TASK_UPDATE_WORKERS") != str(expected_draft_pa_task_update_workers):
    errors.append("draft-PA task-update worker contract mismatch")
if environment.get("VLLM_ASCEND_SPECSLO_DRAFT_DEVICE_PAGED_ATTENTION") != str(
    int(expected_draft_device_paged_attention)
):
    errors.append("draft device-position PagedAttention contract mismatch")
if environment.get("VLLM_ASCEND_PEARL_DRAFT_PA_CONTIGUOUS_PARTITION") != str(
    int(expected_draft_pa_contiguous_partition)
):
    errors.append("draft-PA contiguous partition contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_TARGET_REPLAY_AHEAD") != str(int(expected_target_replay_ahead)):
    errors.append("target replay-ahead contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_EARLY_CURRENT_TARGET_SUBMIT") != str(
    int(expected_early_current_target_submit)
):
    errors.append("early current-target submit contract mismatch")
early_mixed_environment = environment.get(
    "VLLM_ASCEND_SPECRHYTHM_EARLY_CURRENT_MIXED_TARGET_SUBMIT"
)
# Results produced immediately before this variable was added to
# CRITICAL_ENVIRONMENT do not carry the parent-process copy.  Their four
# worker counters below are still an execution-time proof of the exact value.
# New results must record the environment as well.
if early_mixed_environment is not None and early_mixed_environment != str(
    int(expected_early_current_mixed_target_submit)
):
    errors.append("early current mixed-target submit contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_PREFETCH_AHEAD") != str(
    int(expected_prefetch_ahead)
):
    errors.append("prefetch-ahead contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_REQUEST_BUCKET") != str(
    expected_stable_target_request_bucket
):
    errors.append("stable target request-bucket contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_ONLINE_TARGET_MAX_BATCH") != str(
    expected_online_target_max_batch
):
    errors.append("online target max-batch contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_DENSE_MAX_CAPACITY") != str(
    expected_stable_target_dense_max_capacity
):
    errors.append("stable target dense max-capacity contract mismatch")
if payload.get("spec_rhythm_max_target_batch") != expected_target_max_batch:
    errors.append("target max-batch contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_MAX_CAPACITY") != str(
    expected_target_max_batch
):
    errors.append("stable target max-capacity contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_VERIFY_CAPACITIES") != (
    expected_mixed_target_verify_capacities
):
    errors.append("mixed-target verify-capacity contract mismatch")
if environment.get("VLLM_ASCEND_PEARL_TARGET_NZ_MODE") != (
    None if expected_target_nz_mode == "default" else expected_target_nz_mode
):
    errors.append("target NZ mode contract mismatch")
if environment.get("VLLM_ASCEND_PEARL_TARGET_LARGE_M_NZ_MIN_ROWS") != str(
    expected_target_large_m_nz_min_rows
):
    errors.append("target large-M NZ threshold contract mismatch")
if environment.get("VLLM_ASCEND_PEARL_TARGET_REPLICATE_INPUT_EMBEDDING") != str(
    int(expected_target_replicate_input_embedding)
):
    errors.append("target replicated input-embedding contract mismatch")
if environment.get("PYTORCH_NPU_ALLOC_CONF") != (
    None if expected_npu_alloc_conf == "default" else expected_npu_alloc_conf
):
    errors.append("NPU allocator contract mismatch")
if row.get("output_tokens") != 463 * 256:
    errors.append("not fixed 463x256 output")
if row.get("graph_qualification_fixed_point") is not True:
    errors.append("graph qualification did not converge")
for name in ("aclgraph_failed_captures", "aclgraph_capacity_fallbacks", "aclgraph_shape_fallbacks"):
    if int(row.get(name, -1)) != 0:
        errors.append(f"{name} is nonzero")
workers = row.get("worker_aclgraph_metrics", [])
targets = [worker for worker in workers if not int(worker.get("is_draft_rank", 0))]
if len(targets) != 3:
    errors.append("expected three target workers")
for worker in workers:
    rank = int(worker.get("rank", -1))
    is_draft = bool(int(worker.get("is_draft_rank", 0)))
    replicated = bool(int(worker.get("target_input_embedding_replicated", 0)))
    replica_bytes = int(worker.get("target_input_embedding_replica_bytes", 0))
    expected_replicated = expected_target_replicate_input_embedding and not is_draft
    if replicated != expected_replicated:
        errors.append(f"worker rank {rank} target input-embedding replication mismatch")
    if expected_replicated and replica_bytes <= 0:
        errors.append(f"worker rank {rank} reports no replicated input-embedding storage")
    if not expected_replicated and replica_bytes != 0:
        errors.append(f"worker rank {rank} unexpectedly reports replicated input-embedding storage")
    lm_head_nz_mode15 = bool(int(worker.get("target_lm_head_nz_mode15", 0)))
    lm_head_nz_bytes = int(worker.get("target_lm_head_nz_bytes", 0))
    expected_lm_head_nz_mode15 = expected_target_nz_mode == "15" and not is_draft
    if lm_head_nz_mode15 != expected_lm_head_nz_mode15:
        errors.append(f"worker rank {rank} target LM-head NZ15 mismatch")
    if expected_lm_head_nz_mode15 and lm_head_nz_bytes <= 0:
        errors.append(f"worker rank {rank} reports no target LM-head NZ15 storage")
    if not expected_lm_head_nz_mode15 and lm_head_nz_bytes != 0:
        errors.append(f"worker rank {rank} unexpectedly reports target LM-head NZ15 storage")
measured_workers = row.get("measured_worker_aclgraph_deltas", [])
measured_targets = [
    worker for worker in measured_workers
    if not int(worker.get("is_draft_rank", 0))
]
if len(measured_targets) != 3:
    errors.append("expected three measured target-worker deltas")
bonus_evidence = {
    "enabled": int(expected_linear_bonus_token),
    "eligible_rows_by_rank": {
        str(int(worker.get("rank", -1))): int(
            worker.get("worker_spec_rhythm_linear_bonus_eligible_rows", 0)
        )
        for worker in measured_workers
    },
    "committed_tokens_by_rank": {
        str(int(worker.get("rank", -1))): int(
            worker.get("worker_spec_rhythm_linear_bonus_committed_tokens", 0)
        )
        for worker in measured_workers
    },
}
for metric_name in ("eligible_rows_by_rank", "committed_tokens_by_rank"):
    values = list(bonus_evidence[metric_name].values())
    if len(values) != 4 or len(set(values)) != 1:
        errors.append(f"linear bonus-token {metric_name} is inconsistent across ranks")
    elif expected_linear_bonus_token and values[0] <= 0:
        errors.append(f"linear bonus-token measured no {metric_name}")
    elif not expected_linear_bonus_token and values[0] != 0:
        errors.append(f"linear bonus-token disabled but measured {metric_name}")
target_fia_evidence = target_fia_parallel_update_evidence(
    workers,
    measured_workers,
    expected_workers=expected_target_fia_task_update_workers,
)
errors.extend(
    f"target-FIA evidence: {message}"
    for message in target_fia_evidence.get("errors", ())
)
if expected_draft_device_paged_attention:
    draft_ranks = [
        worker for worker in measured_workers
        if int(worker.get("is_draft_rank", 0))
    ]
    draft_pa_evidence = {
        "schema": "specslo-draft-device-pa-evidence-v1",
        "enabled": 1,
        "errors": [],
    }
    if len(draft_ranks) != 1:
        draft_pa_evidence["errors"].append("expected one measured draft rank")
    else:
        draft_rank = draft_ranks[0]
        replay_count = int(draft_rank.get("aclgraph_draft_replay_calls_delta", 0))
        taskless_count = int(draft_rank.get("aclgraph_draft_taskless_replays_delta", 0))
        update_replays = int(draft_rank.get("aclgraph_draft_task_update_replays_delta", 0))
        update_tasks = int(draft_rank.get("aclgraph_draft_task_update_tasks_delta", 0))
        draft_pa_evidence.update({
            "replay_count": replay_count,
            "taskless_count": taskless_count,
            "task_update_replays": update_replays,
            "task_update_tasks": update_tasks,
        })
        if replay_count <= 0 or taskless_count != replay_count:
            draft_pa_evidence["errors"].append("taskless replay count does not equal draft replay count")
        if update_replays != 0 or update_tasks != 0:
            draft_pa_evidence["errors"].append("device PA unexpectedly refreshed host attention tasks")
    errors.extend(
        f"draft device-PA evidence: {message}"
        for message in draft_pa_evidence["errors"]
    )
else:
    draft_pa_evidence = draft_pa_parallel_update_evidence(
        workers,
        measured_workers,
        expected_workers=expected_draft_pa_task_update_workers,
    )
    errors.extend(
        f"draft-PA evidence: {message}"
        for message in draft_pa_evidence.get("errors", ())
    )
draft_contiguous_replays = {
    int(worker.get("rank", -1)): int(
        worker.get("aclgraph_draft_pa_contiguous_partition_replays_delta", 0)
    )
    for worker in measured_workers
    if int(worker.get("is_draft_rank", 0))
}
if len(draft_contiguous_replays) != 1:
    errors.append("expected one measured draft rank for contiguous-partition evidence")
elif expected_draft_pa_contiguous_partition:
    if next(iter(draft_contiguous_replays.values())) <= 0:
        errors.append("draft-PA contiguous partition did not execute")
elif next(iter(draft_contiguous_replays.values())) != 0:
    errors.append("draft-PA contiguous partition executed while disabled")
target_replay_evidence = target_replay_ahead_evidence(
    workers,
    enabled=expected_target_replay_ahead,
)
errors.extend(
    f"target replay-ahead evidence: {message}"
    for message in target_replay_evidence.get("errors", ())
)
early_target_cycles = {
    int(worker.get("rank", -1)): int(
        worker.get(
            "worker_spec_rhythm_early_current_target_submit_cycles",
            0,
        )
    )
    for worker in workers
}
early_mixed_target_cycles = {
    int(worker.get("rank", -1)): int(
        worker.get(
            "worker_spec_rhythm_early_current_mixed_target_submit_cycles",
            0,
        )
    )
    for worker in workers
}
for worker in workers:
    rank = int(worker.get("rank", -1))
    enabled_value = int(
        worker.get(
            "worker_spec_rhythm_early_current_target_submit_enabled",
            0,
        )
    )
    if enabled_value != int(expected_early_current_target_submit):
        errors.append(
            f"worker rank {rank} early current-target enabled mismatch"
        )
    mixed_enabled_value = int(
        worker.get(
            "worker_spec_rhythm_early_current_mixed_target_submit_enabled",
            0,
        )
    )
    if mixed_enabled_value != int(expected_early_current_mixed_target_submit):
        errors.append(
            f"worker rank {rank} early current mixed-target enabled mismatch"
        )
    is_draft = bool(int(worker.get("is_draft_rank", 0)))
    if (
        expected_early_current_target_submit
        and not is_draft
        and early_target_cycles[rank] <= 0
    ):
        errors.append(
            f"worker rank {rank} submitted no early current-target graph"
        )
    if (
        (not expected_early_current_target_submit or is_draft)
        and early_target_cycles[rank] != 0
    ):
        errors.append(
            f"worker rank {rank} unexpectedly submitted an early target graph"
        )
    if (
        expected_early_current_mixed_target_submit
        and not is_draft
        and early_mixed_target_cycles[rank] <= 0
    ):
        errors.append(
            f"worker rank {rank} submitted no early mixed target graph"
        )
    if (
        (not expected_early_current_mixed_target_submit or is_draft)
        and early_mixed_target_cycles[rank] != 0
    ):
        errors.append(
            f"worker rank {rank} unexpectedly submitted an early mixed target graph"
        )
prefetch_evidence = {
    "enabled": int(expected_prefetch_ahead),
    "reserved_batches_by_rank": {},
    "reserved_requests_by_rank": {},
    "deferred_publications_by_rank": {},
}
for worker in workers:
    rank = int(worker.get("rank", -1))
    enabled_value = int(
        worker.get("worker_spec_rhythm_prefetch_ahead_enabled", 0)
    )
    reserved_batches = int(
        worker.get("worker_spec_rhythm_prefetch_ahead_reserved_batches", 0)
    )
    reserved_requests = int(
        worker.get("worker_spec_rhythm_prefetch_ahead_reserved_requests", 0)
    )
    deferred_publications = int(
        worker.get(
            "worker_spec_rhythm_prefetch_ahead_deferred_publications",
            0,
        )
    )
    prefetch_evidence["reserved_batches_by_rank"][str(rank)] = reserved_batches
    prefetch_evidence["reserved_requests_by_rank"][str(rank)] = reserved_requests
    prefetch_evidence["deferred_publications_by_rank"][str(rank)] = deferred_publications
    if enabled_value != int(expected_prefetch_ahead):
        errors.append(f"worker rank {rank} prefetch-ahead enablement mismatch")
    if expected_prefetch_ahead and (
        reserved_batches <= 0
        or reserved_requests <= 0
        or deferred_publications <= 0
    ):
        errors.append(
            f"worker rank {rank} has no measured full-batch prefetch-ahead evidence"
        )
    if not expected_prefetch_ahead and (
        reserved_batches or reserved_requests or deferred_publications
    ):
        errors.append(
            f"worker rank {rank} unexpectedly executed prefetch-ahead"
        )
profile_contract = (
    qualified_mc2_profile_replay_contract(profile_path)
    if mode == "on"
    else None
)
mc2_replay_coverage = (
    measure_qualified_mc2_replay_coverage(
        measured_workers,
        profile_contract,
    )
    if profile_contract is not None
    else None
)
if mode == "on":
    if environment.get("VLLM_ASCEND_PEARL_MC2_DOWN_PROJ") != "1":
        errors.append("MC2 down-projection route is not enabled")
    if payload.get("mc2_replay_profile_contract") != profile_contract:
        errors.append("result MC2 replay profile contract differs from the input profile")
    if row.get("mc2_measured_replay_coverage") != mc2_replay_coverage:
        errors.append("result MC2 measured replay coverage is not reproducible")
    if not mc2_replay_coverage.get("passed", False):
        errors.extend(
            f"MC2 measured replay coverage: {message}"
            for message in mc2_replay_coverage.get("errors", ())
        )
errors.extend(
    _host_continuation_measurement_issues(
        measured_workers,
        expected_reuse=expected_host_continuation_reuse,
    )
)
measured_drafts = [
    worker for worker in measured_workers
    if int(worker.get("is_draft_rank", 0))
]
continuation_evidence = {
    "expected_host_reuse": int(expected_host_continuation_reuse),
    "draft_host_reuses": sum(
        int(worker.get(
            "worker_spec_rhythm_gloo_proposal_host_continuation_reuses", 0
        ))
        for worker in measured_drafts
    ),
    "draft_device_reads": sum(
        int(worker.get(
            "worker_spec_rhythm_gloo_proposal_device_continuation_reads", 0
        ))
        for worker in measured_drafts
    ),
    "host_confidence_reuses_by_rank": {
        str(int(worker.get("rank", -1))): int(
            worker.get(
                "worker_spec_rhythm_gloo_proposal_host_confidence_reuses",
                0,
            )
        )
        for worker in measured_workers
    },
    "confidence_elisions_by_rank": {
        str(int(worker.get("rank", -1))): int(
            worker.get(
                "worker_spec_rhythm_linear_acceptance_confidence_elisions",
                0,
            )
        )
        for worker in measured_workers
    },
    "proposal_matrix_view_reuses_by_rank": {
        str(int(worker.get("rank", -1))): int(
            worker.get(
                "worker_spec_rhythm_proposal_matrix_view_reuses",
                0,
            )
        )
        for worker in measured_workers
    },
    "proposal_matrix_stack_fallbacks_by_rank": {
        str(int(worker.get("rank", -1))): int(
            worker.get(
                "worker_spec_rhythm_proposal_matrix_stack_fallbacks",
                0,
            )
        )
        for worker in measured_workers
    },
    "target_staged_submits": sum(
        int(worker.get(
            "spec_rhythm_staged_gloo_correction_submit_calls_delta", 0
        ))
        for worker in measured_targets
    ),
    "target_host_reuses": sum(
        int(worker.get(
            "spec_rhythm_staged_gloo_correction_host_continuation_reuses_delta", 0
        ))
        for worker in measured_targets
    ),
    "target_device_copy_submits": sum(
        int(worker.get(
            "spec_rhythm_staged_gloo_correction_device_continuation_copies_delta", 0
        ))
        for worker in measured_targets
    ),
}
for rank, count in continuation_evidence["host_confidence_reuses_by_rank"].items():
    elisions = continuation_evidence["confidence_elisions_by_rank"][rank]
    if expected_acceptance_only:
        if count != 0:
            errors.append(
                f"worker rank {rank} transported confidence in acceptance-only mode"
            )
        if elisions <= 0:
            errors.append(
                f"worker rank {rank} elided no acceptance-only confidence values"
            )
    elif count <= 0:
        errors.append(
            f"worker rank {rank} reused no host draft confidence"
        )
for rank, count in continuation_evidence["proposal_matrix_view_reuses_by_rank"].items():
    if count <= 0:
        errors.append(
            f"worker rank {rank} reused no contiguous proposal matrix view"
        )
fused = sum(int(worker.get("mc2_dispatch_fused_success", 0)) for worker in targets)
fallback = sum(int(worker.get("mc2_dispatch_fallback", 0)) for worker in targets)
exceptions = sum(int(worker.get("mc2_dispatch_exception", 0)) for worker in targets)
measured_fallback = sum(
    int(worker.get("mc2_dispatch_fallback_delta", 0))
    for worker in measured_targets
)
measured_exceptions = sum(
    int(worker.get("mc2_dispatch_exception_delta", 0))
    for worker in measured_targets
)
if mode == "on" and fused <= 0:
    errors.append("MC2-on never dispatched")
if mode == "off" and fused != 0:
    errors.append("MC2-off dispatched fused work")
if fallback or exceptions or measured_fallback or measured_exceptions:
    errors.append("MC2 fallback/exception observed")
if mode == "on":
    for worker in targets:
        rank = int(worker.get("rank", -1))
        # Qwen3-32B has 64 decoder layers.  A complete attention+down-proj
        # route manifest therefore contains exactly two routes per layer.
        if int(worker.get("mc2_static_route_count", 0)) != 128:
            errors.append(
                f"target rank {rank} does not expose all 128 attention/down-proj MC2 routes"
            )
for worker in measured_workers:
    rank = int(worker.get("rank", -1))
    if int(worker.get("worker_spec_rhythm_dual_batch_overlap_protocol", 0)) != 1:
        errors.append(f"worker rank {rank} did not run the dual-batch overlap protocol")
    if int(worker.get("worker_spec_rhythm_dual_batch_overlap_submission_cycles", 0)) <= 0:
        errors.append(f"worker rank {rank} has no measured dual-batch overlap submissions")
for worker in workers:
    rank = int(worker.get("rank", -1))
    is_draft = bool(int(worker.get("is_draft_rank", 0)))
    if int(worker.get("worker_spec_rhythm_drain_fifo_enabled", -1)) != int(expected_drain_fifo):
        errors.append(f"worker rank {rank} drain-FIFO enablement mismatch")
    if expected_drain_fifo and int(worker.get("worker_spec_rhythm_drain_fifo_cycles", 0)) <= 0:
        errors.append(f"worker rank {rank} has no drain-FIFO cycles")
    if int(worker.get("worker_spec_rhythm_drain_padding_repair_enabled", -1)) != int(
        expected_drain_padding_repair
    ):
        errors.append(f"worker rank {rank} drain-padding-repair enablement mismatch")
    if expected_drain_padding_repair and int(
        worker.get("worker_spec_rhythm_drain_padding_repair_admitted_rows", 0)
    ) <= 0:
        errors.append(f"worker rank {rank} has no admitted drain padding repairs")
    if int(worker.get("worker_spec_rhythm_online_target_max_batch", -1)) != expected_online_target_max_batch:
        errors.append(f"worker rank {rank} online target max-batch mismatch")
    if not is_draft and int(worker.get("spec_rhythm_stable_target_verify_dense_max_capacity", -1)) != (
        expected_stable_target_dense_max_capacity or expected_target_max_batch
    ):
        errors.append(f"target rank {rank} stable target dense max-capacity mismatch")
    if expected_online_target_max_batch > 0:
        if int(worker.get("worker_spec_rhythm_online_target_cap_cycles", 0)) <= 0:
            errors.append(f"worker rank {rank} never exercised the online target cap")
        if expected_target_max_batch > expected_online_target_max_batch:
            if int(worker.get("worker_spec_rhythm_drain_target_expand_cycles", 0)) <= 0:
                errors.append(f"worker rank {rank} never expanded the immutable-drain target cap")
            if int(
                worker.get("worker_spec_rhythm_drain_target_selected_over_online_cap_cycles", 0)
            ) <= 0:
                errors.append(f"worker rank {rank} never selected a drain cohort above the online cap")
    if int(worker.get("worker_spec_rhythm_drain_lrtf_enabled", -1)) != int(expected_drain_lrtf):
        errors.append(f"worker rank {rank} drain-LRTF enablement mismatch")
    if int(worker.get("worker_spec_rhythm_drain_lrtf_max_active", -1)) != expected_drain_lrtf_max_active:
        errors.append(f"worker rank {rank} drain-LRTF active guard mismatch")
    if expected_drain_lrtf and int(worker.get("worker_spec_rhythm_drain_lrtf_cycles", 0)) <= 0:
        errors.append(f"worker rank {rank} has no drain-LRTF cycles")
    if expected_drain_lrtf:
        if expected_drain_lrtf_max_active > 0 and int(
            worker.get("worker_spec_rhythm_drain_lrtf_threshold_skipped_cycles", 0)
        ) <= 0:
            errors.append(f"worker rank {rank} never exercised the drain-LRTF active guard")
        if int(worker.get("worker_spec_rhythm_drain_lrtf_target_ranked_cycles", 0)) <= 0:
            errors.append(f"worker rank {rank} never ranked ready target work by drain-LRTF")
        if int(worker.get("worker_spec_rhythm_drain_lrtf_normal_ranked_cycles", 0)) <= 0:
            errors.append(f"worker rank {rank} never ranked normal/repair draft work by drain-LRTF")
        if int(worker.get("worker_spec_rhythm_drain_lrtf_unified_cycles", 0)) <= 0:
            errors.append(f"worker rank {rank} has no unified target+draft drain-LRTF cycle")
        if int(worker.get("worker_spec_rhythm_drain_lrtf_target_candidates", 0)) <= 0:
            errors.append(f"worker rank {rank} has no drain-LRTF target candidates")
        if int(worker.get("worker_spec_rhythm_drain_lrtf_normal_candidates", 0)) <= 0:
            errors.append(f"worker rank {rank} has no drain-LRTF normal/repair candidates")
        if int(worker.get("worker_spec_rhythm_drain_lrtf_repair_penalty_rows", 0)) <= 0:
            errors.append(f"worker rank {rank} never charged the drain-LRTF repair-stage penalty")
        if int(worker.get("worker_spec_rhythm_drain_lrtf_normal_selected", 0)) <= 0:
            errors.append(f"worker rank {rank} selected no normal/repair drain-LRTF work")
        if int(worker.get("worker_spec_rhythm_drain_lrtf_normal_cross_home_selected", 0)) <= 0:
            errors.append(f"worker rank {rank} has no cross-home drain-LRTF replacement")
    if int(worker.get("worker_spec_rhythm_drain_target_lrtf_enabled", -1)) != int(
        expected_drain_target_lrtf
    ):
        errors.append(f"worker rank {rank} drain target-LRTF enablement mismatch")
    if expected_drain_target_lrtf:
        if int(worker.get("worker_spec_rhythm_drain_target_lrtf_cycles", 0)) <= 0:
            errors.append(f"worker rank {rank} has no drain target-LRTF cycles")
        if int(worker.get("worker_spec_rhythm_drain_target_lrtf_ranked_cycles", 0)) <= 0:
            errors.append(f"worker rank {rank} never ranked target work by target-only drain-LRTF")
        if int(worker.get("worker_spec_rhythm_drain_target_lrtf_candidates", 0)) <= 0:
            errors.append(f"worker rank {rank} has no target-only drain-LRTF candidates")
        if int(worker.get("worker_spec_rhythm_drain_lrtf_normal_ranked_cycles", 0)) != 0:
            errors.append(f"worker rank {rank} changed draft ranking in target-only drain-LRTF")
        if int(worker.get("worker_spec_rhythm_drain_lrtf_normal_cross_home_selected", 0)) != 0:
            errors.append(f"worker rank {rank} selected cross-home draft work in target-only drain-LRTF")
    if int(worker.get("worker_spec_rhythm_target_residual_lrtf_enabled", -1)) != int(
        expected_target_residual_lrtf
    ):
        errors.append(f"worker rank {rank} target residual-LRTF enablement mismatch")
    if expected_target_residual_lrtf:
        if int(worker.get("worker_spec_rhythm_target_residual_lrtf_ranked_cycles", 0)) <= 0:
            errors.append(f"worker rank {rank} never ranked residual target work")
        if int(worker.get("worker_spec_rhythm_target_residual_lrtf_candidates", 0)) <= 0:
            errors.append(f"worker rank {rank} has no residual target candidates")
        if int(worker.get("worker_spec_rhythm_target_residual_lrtf_selected", 0)) <= 0:
            errors.append(f"worker rank {rank} selected no residual target work")
        if int(worker.get("worker_spec_rhythm_target_residual_lrtf_deferred", 0)) <= 0:
            errors.append(f"worker rank {rank} deferred no residual target work")
gate = {
    "status": "pass" if not errors else "fail",
    "mode": mode,
    "target_fia_task_event_group_size": expected_target_fia_task_event_group_size,
    "target_fia_task_prefix_event_group_size": expected_target_fia_task_prefix_event_group_size,
    "target_nz_mode": expected_target_nz_mode,
    "target_large_m_nz_min_rows": expected_target_large_m_nz_min_rows,
    "target_replicate_input_embedding": int(expected_target_replicate_input_embedding),
    "stable_target_request_bucket": expected_stable_target_request_bucket,
    "target_max_batch": expected_target_max_batch,
    "online_target_max_batch": expected_online_target_max_batch,
    "stable_target_dense_max_capacity": expected_stable_target_dense_max_capacity,
    "mixed_target_verify_capacities": expected_mixed_target_verify_capacities,
    "mixed_target_graph_buckets": expected_mixed_target_graph_buckets,
    "npu_alloc_conf": expected_npu_alloc_conf,
    "raw_tokens_per_second": row.get("e2e_throughput_tokens_per_second"),
    "paper_goodput_tokens_per_second": row.get("slo", {}).get("paper", {}).get("goodput_tokens_per_e2e_second"),
    "paper_attainment": row.get("slo", {}).get("paper", {}).get("attainment"),
    "mc2_fused_success": fused,
    "mc2_fallback": fallback,
    "mc2_exception": exceptions,
    "measured_mc2_fallback": measured_fallback,
    "measured_mc2_exception": measured_exceptions,
    "mc2_measured_replay_coverage": mc2_replay_coverage,
    "target_fia_task_update": target_fia_evidence,
    "draft_pa_task_update": draft_pa_evidence,
    "target_replay_ahead": target_replay_evidence,
    "early_current_target_submit": {
        "enabled": int(expected_early_current_target_submit),
        "cycles_by_rank": {
            str(rank): count for rank, count in early_target_cycles.items()
        },
        "mixed_cycles_by_rank": {
            str(rank): count for rank, count in early_mixed_target_cycles.items()
        },
        "mixed_enabled": int(expected_early_current_mixed_target_submit),
    },
    "prefetch_ahead": prefetch_evidence,
    "dual_batch_overlap_submission_cycles_by_rank": {
        str(int(worker.get("rank", -1))): int(
            worker.get("worker_spec_rhythm_dual_batch_overlap_submission_cycles", 0)
        )
        for worker in measured_workers
    },
    "drain_fifo": {
        "enabled": int(expected_drain_fifo),
        "cycles_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_fifo_cycles", 0)
            )
            for worker in workers
        },
    },
    "dynamic_target_cohort": {
        "online_max_batch": expected_online_target_max_batch,
        "configured_max_batch": expected_target_max_batch,
        "online_cap_cycles_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_online_target_cap_cycles", 0)
            )
            for worker in workers
        },
        "drain_expand_cycles_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_target_expand_cycles", 0)
            )
            for worker in workers
        },
        "drain_over_cap_cycles_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_target_selected_over_online_cap_cycles", 0)
            )
            for worker in workers
        },
        "drain_over_cap_rows_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_target_selected_over_online_cap_rows", 0)
            )
            for worker in workers
        },
        "max_selected_requests_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_target_selected_max_requests", 0)
            )
            for worker in workers
        },
    },
    "drain_lrtf": {
        "enabled": int(expected_drain_lrtf),
        "max_active": expected_drain_lrtf_max_active,
        "threshold_skipped_cycles_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_lrtf_threshold_skipped_cycles", 0)
            )
            for worker in workers
        },
        "cycles_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_lrtf_cycles", 0)
            )
            for worker in workers
        },
        "target_ranked_cycles_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_lrtf_target_ranked_cycles", 0)
            )
            for worker in workers
        },
        "normal_ranked_cycles_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_lrtf_normal_ranked_cycles", 0)
            )
            for worker in workers
        },
        "unified_cycles_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_lrtf_unified_cycles", 0)
            )
            for worker in workers
        },
        "target_candidates_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_lrtf_target_candidates", 0)
            )
            for worker in workers
        },
        "normal_candidates_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_lrtf_normal_candidates", 0)
            )
            for worker in workers
        },
        "repair_penalty_rows_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_lrtf_repair_penalty_rows", 0)
            )
            for worker in workers
        },
        "normal_selected_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_lrtf_normal_selected", 0)
            )
            for worker in workers
        },
        "normal_cross_home_selected_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_lrtf_normal_cross_home_selected", 0)
            )
            for worker in workers
        },
    },
    "drain_target_lrtf": {
        "enabled": int(expected_drain_target_lrtf),
        "cycles_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_target_lrtf_cycles", 0)
            )
            for worker in workers
        },
        "target_ranked_cycles_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_target_lrtf_ranked_cycles", 0)
            )
            for worker in workers
        },
        "target_candidates_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_drain_target_lrtf_candidates", 0)
            )
            for worker in workers
        },
    },
    "target_residual_lrtf": {
        "enabled": int(expected_target_residual_lrtf),
        "ranked_cycles_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_target_residual_lrtf_ranked_cycles", 0)
            )
            for worker in workers
        },
        "candidates_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_target_residual_lrtf_candidates", 0)
            )
            for worker in workers
        },
        "selected_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_target_residual_lrtf_selected", 0)
            )
            for worker in workers
        },
        "deferred_by_rank": {
            str(int(worker.get("rank", -1))): int(
                worker.get("worker_spec_rhythm_target_residual_lrtf_deferred", 0)
            )
            for worker in workers
        },
    },
    "host_continuation_evidence": continuation_evidence,
    "linear_bonus_token": bonus_evidence,
    "errors": errors,
}
gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if errors:
    raise SystemExit("; ".join(errors))
PY

echo "candidate result: $run_dir/result.json"
echo "candidate gate:   $run_dir/gate.json"
