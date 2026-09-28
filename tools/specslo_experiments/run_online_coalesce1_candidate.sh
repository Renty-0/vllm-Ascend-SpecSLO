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
prefill_only=${SPECSLO_MIXED_TARGET_GRAPH_PREFILL_ONLY:-0}
residual_tight_only=${SPECSLO_RESIDUAL_EAGER_TIGHT_ONLY:-0}
host_continuation_reuse=${SPECSLO_GLOO_PROPOSAL_REUSE_HOST_CONTINUATION:-1}
drain_fifo=${SPECSLO_DRAIN_FIFO:-1}
drain_lrtf=${SPECSLO_DRAIN_LRTF:-0}
graph_entries=${SPECSLO_MAX_ACLGRAPH_ENTRIES:-80}
kv_blocks=${SPECSLO_NUM_KVCACHE_BLOCKS:-544}
profile_host_decode_steps=${SPECSLO_PROFILE_HOST_DECODE_STEPS:-0}
profile_wait_steps=${SPECSLO_NPU_PROFILE_WAIT_STEPS:-16}
target_fia_task_update_workers=${SPECSLO_TARGET_FIA_TASK_UPDATE_WORKERS:-1}
draft_pa_task_update_workers=${SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS:-1}

if [[ ! $run_tag =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid run tag: $run_tag" >&2
  exit 2
fi
IFS=',' read -r -a devices <<< "$visible_devices"
if [[ ${#devices[@]} -ne 4 ]]; then
  echo "exactly four visible NPUs are required" >&2
  exit 2
fi
for value in "$acceptance_only" "$prefill_only" "$residual_tight_only" \
  "$host_continuation_reuse" "$drain_fifo" "$drain_lrtf"; do
  if [[ $value != 0 && $value != 1 ]]; then
    echo "boolean experiment switches must be 0 or 1" >&2
    exit 2
  fi
done
if [[ $drain_lrtf == 1 && $drain_fifo != 1 ]]; then
  echo "SPECSLO_DRAIN_LRTF=1 requires SPECSLO_DRAIN_FIFO=1" >&2
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
      $target_fia_task_update_workers != 4 ]]; then
  echo "SPECSLO_TARGET_FIA_TASK_UPDATE_WORKERS must be 1, 2, or 4" >&2
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

run_dir=${result_root}/runs/candidate-${mc2_mode}-${run_tag}
if [[ -e $run_dir ]]; then
  echo "refusing to overwrite existing run: $run_dir" >&2
  exit 3
fi
mkdir -p "$run_dir"

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
assert_cards_free

source /usr/local/Ascend/cann-9.0.0/set_env.sh
export PYTHONPATH=$repo:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=$visible_devices
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_DETERMINISTIC=true
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export ASCEND_CUSTOM_OPP_PATH=$base_vendor:$vendor
export LD_LIBRARY_PATH=$vendor/op_api/lib:$repo/vllm_ascend:$repo/vllm_ascend/lib64:$repo/build/temp.linux-aarch64-cpython-311:$torch_lib:$torch_npu_lib:/usr/local/Ascend/cann-9.0.0/lib64:${LD_LIBRARY_PATH:-}

# The formal performance rounds leave SPECSLO_PROFILE_HOST_DECODE_STEPS at
# zero, so inherited shell state cannot accidentally turn on CANN profiling.
# A separate diagnostic run opts in explicitly.  Measurement-only mode arms
# the profiler after graph qualification when the benchmark configures its
# non-synchronizing host timeline for the final generate call.
profile_args=()
npu_profile_dir=
if (( profile_host_decode_steps > 0 )); then
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

export VLLM_ASCEND_PEARL_DRAFT_NZ_MODE=-1
unset VLLM_ASCEND_PEARL_TARGET_NZ_MODE
unset VLLM_ASCEND_PEARL_ENABLE_TP3_MM_ALL_REDUCE
export VLLM_ASCEND_PEARL_DRAFT_REPLAY_FIRST_TASK_UPDATE=1
export VLLM_ASCEND_PEARL_DRAFT_STEP_MAJOR_PA_TASK_UPDATE=0
# Overwrite the runtime variable even at the production default so a parent
# shell cannot silently contaminate the candidate contract.
export VLLM_ASCEND_PEARL_DRAFT_PA_TASK_UPDATE_WORKERS=$draft_pa_task_update_workers
export VLLM_ASCEND_PEARL_DRAFT_GRAPH_UPDATE_STREAM_PRIORITY=0
export VLLM_ASCEND_PEARL_TARGET_REPLAY_FIRST_TASK_UPDATE=1
export VLLM_ASCEND_PEARL_TARGET_FIA_TASK_EVENT_GROUP_SIZE=1
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
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_MAX_TOKENS=645
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_BUCKETS=64,128,192,256,320,384,448,512
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_VERIFY_CAPACITIES=8,16,24,32
export VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_PROMPT_CAPACITY=4
export VLLM_ASCEND_SPECRHYTHM_PREFETCH_AHEAD=0
export VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_MAX_CAPACITY=32
export VLLM_ASCEND_SPECRHYTHM_STABLE_TARGET_VERIFY_REQUEST_BUCKET=1
export VLLM_ASCEND_SPECRHYTHM_MIXED_DRAFT_PREFILL=0
export VLLM_ASCEND_SPECRHYTHM_OVERLAP_DRAFT_PREFILL=0
export VLLM_ASCEND_SPECRHYTHM_GLOO_CORRECTION=1
export VLLM_ASCEND_SPECRHYTHM_GLOO_ACCOUNTING=1
export VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL=1
export VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL_REUSE_HOST_CONTINUATION=$host_continuation_reuse
export VLLM_ASCEND_SPECRHYTHM_GOODPUT_EDF=0
export VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST=0
export VLLM_ASCEND_SPECRHYTHM_DRAIN_FIFO=$drain_fifo
export VLLM_ASCEND_SPECRHYTHM_DRAIN_LRTF=$drain_lrtf
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
unset VLLM_ASCEND_SPECSLO_DRAFT_DEVICE_PAGED_ATTENTION
unset VLLM_ASCEND_SPECRHYTHM_LINEAR_DRAFT_FIA_BUCKET
unset VLLM_ASCEND_SPECRHYTHM_FORCE_STEPWISE_TARGET
unset VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH
unset VLLM_ASCEND_SPECRHYTHM_STEPWISE_TARGET_FIA

mc2_args=()
if [[ $mc2_mode == on ]]; then
  mc2_args=(--enable-mc2 --mc2-profile "$profile")
fi

cd "$repo"
git rev-parse HEAD > "$run_dir/git-head.txt"
git diff --binary > "$run_dir/working-tree.patch"
git status --porcelain=v1 --untracked-files=all > "$run_dir/git-status.txt"
sha256sum "$manifest" "$profile" "$extension" > "$run_dir/input.sha256"
{
  printf 'mode=%s\nrun_tag=%s\nvisible_devices=%s\n' "$mc2_mode" "$run_tag" "$visible_devices"
  printf 'result_root=%s\nrun_dir=%s\n' "$result_root" "$run_dir"
  printf 'fallback_max_batch=%s\nacceptance_only=%s\nprefill_only=%s\n' "$fallback_max_batch" "$acceptance_only" "$prefill_only"
  printf 'residual_tight_only=%s\ngraph_entries=%s\nkv_blocks=%s\n' "$residual_tight_only" "$graph_entries" "$kv_blocks"
  printf 'host_continuation_reuse=%s\ndrain_fifo=%s\ndrain_lrtf=%s\n' \
    "$host_continuation_reuse" "$drain_fifo" "$drain_lrtf"
  printf 'profile_host_decode_steps=%s\nnpu_profile_wait_steps=%s\n' "$profile_host_decode_steps" "$profile_wait_steps"
  printf 'target_fia_task_update_workers=%s\n' "$target_fia_task_update_workers"
  printf 'draft_pa_task_update_workers=%s\n' "$draft_pa_task_update_workers"
  printf 'npu_profile_measurement_only=%s\nnpu_profile_dir=%s\n' \
    "$(( profile_host_decode_steps > 0 ))" "$npu_profile_dir"
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
  --spec-rhythm-merge-ready-homes --spec-rhythm-linear-eager-cross-graph-bucket \
  --spec-rhythm-linear-idle-residual-eager \
  --spec-rhythm-target-fallback-max-batch "$fallback_max_batch" \
  --spec-rhythm-tree-width 1 --spec-rhythm-tree-depth 1 \
  --spec-rhythm-max-eager-tokens 4 --max-model-len 4096 --max-tokens 256 \
  --batch-sizes 64 --num-prompts 463 --request-manifest "$manifest" \
  --gpu-memory-utilization 0.8 --num-kvcache-blocks "$kv_blocks" \
  --max-aclgraph-entries "$graph_entries" --target-verification-graph-buckets 8 \
  --enable-continuous-batching --enable-preemptive-scheduling --enable-spec-rhythm \
  --spec-rhythm-priority-burst 2 --spec-rhythm-acceptance-floor 0.4 \
  --spec-rhythm-online-prefill --spec-rhythm-prefill-coalesce-min-requests 1 \
  --spec-rhythm-prefill-coalesce-max-wait-ms 0 --spec-rhythm-prefill-token-chunk-size 512 \
  --precompile-serial-draft-graphs --spec-rhythm-stable-graphs \
  --require-no-graph-fallback --draft-use-paged-attention \
  --spec-rhythm-max-target-batch 32 "${mc2_args[@]}" \
  --warmup-runs 1 --graph-qualification-max-runs 8 \
  "${profile_args[@]}" \
  --worker-timeout-seconds 1800 --output-json "$run_dir/result.json.tmp" \
  > "$run_dir/run.log" 2>&1
mv "$run_dir/result.json.tmp" "$run_dir/result.json"

"$python" - "$run_dir/result.json" "$run_dir/gate.json" "$mc2_mode" \
  "$fallback_max_batch" "$prefill_only" "$residual_tight_only" \
  "$acceptance_only" "$host_continuation_reuse" "$drain_fifo" "$drain_lrtf" \
  "$target_fia_task_update_workers" "$draft_pa_task_update_workers" "$profile" <<'PY'
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
expected_draft_pa_task_update_workers = int(sys.argv[12])
profile_path = Path(sys.argv[13]).resolve()
payload = json.loads(result_path.read_text(encoding="utf-8"))
row = payload["results"][0]
errors = []
if payload.get("backend") != "specslo-native-specrhythm":
    errors.append("unexpected backend")
if payload.get("spec_rhythm_prefill_coalesce_min_requests") != 1:
    errors.append("not coalesce1")
if payload.get("spec_rhythm_target_fallback_max_batch") != expected_fallback:
    errors.append("target fallback contract mismatch")
environment = payload.get("runtime_environment", {})
if environment.get("VLLM_ASCEND_SPECRHYTHM_MIXED_TARGET_GRAPH_PREFILL_ONLY") != str(int(expected_prefill_only)):
    errors.append("mixed-target prefill-only contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_RESIDUAL_EAGER_TIGHT_ONLY") != str(int(expected_residual_tight_only)):
    errors.append("residual-eager tight-only contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_LINEAR_ACCEPTANCE_ONLY") != str(int(expected_acceptance_only)):
    errors.append("linear acceptance-only contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_DRAIN_FIFO") != str(int(expected_drain_fifo)):
    errors.append("drain-FIFO contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_DRAIN_LRTF") != str(int(expected_drain_lrtf)):
    errors.append("drain-LRTF contract mismatch")
if environment.get("VLLM_ASCEND_SPECRHYTHM_GLOO_PROPOSAL_REUSE_HOST_CONTINUATION") != str(int(expected_host_continuation_reuse)):
    errors.append("host-continuation reuse contract mismatch")
if environment.get("VLLM_ASCEND_PEARL_TARGET_FIA_TASK_UPDATE_WORKERS") != str(expected_target_fia_task_update_workers):
    errors.append("target-FIA task-update worker contract mismatch")
if environment.get("VLLM_ASCEND_PEARL_DRAFT_PA_TASK_UPDATE_WORKERS") != str(expected_draft_pa_task_update_workers):
    errors.append("draft-PA task-update worker contract mismatch")
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
measured_workers = row.get("measured_worker_aclgraph_deltas", [])
measured_targets = [
    worker for worker in measured_workers
    if not int(worker.get("is_draft_rank", 0))
]
if len(measured_targets) != 3:
    errors.append("expected three measured target-worker deltas")
target_fia_evidence = target_fia_parallel_update_evidence(
    workers,
    measured_workers,
    expected_workers=expected_target_fia_task_update_workers,
)
errors.extend(
    f"target-FIA evidence: {message}"
    for message in target_fia_evidence.get("errors", ())
)
draft_pa_evidence = draft_pa_parallel_update_evidence(
    workers,
    measured_workers,
    expected_workers=expected_draft_pa_task_update_workers,
)
errors.extend(
    f"draft-PA evidence: {message}"
    for message in draft_pa_evidence.get("errors", ())
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
    if int(worker.get("worker_spec_rhythm_drain_fifo_enabled", -1)) != int(expected_drain_fifo):
        errors.append(f"worker rank {rank} drain-FIFO enablement mismatch")
    if expected_drain_fifo and int(worker.get("worker_spec_rhythm_drain_fifo_cycles", 0)) <= 0:
        errors.append(f"worker rank {rank} has no drain-FIFO cycles")
    if int(worker.get("worker_spec_rhythm_drain_lrtf_enabled", -1)) != int(expected_drain_lrtf):
        errors.append(f"worker rank {rank} drain-LRTF enablement mismatch")
    if expected_drain_lrtf and int(worker.get("worker_spec_rhythm_drain_lrtf_cycles", 0)) <= 0:
        errors.append(f"worker rank {rank} has no drain-LRTF cycles")
    if expected_drain_lrtf:
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
gate = {
    "status": "pass" if not errors else "fail",
    "mode": mode,
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
    "drain_lrtf": {
        "enabled": int(expected_drain_lrtf),
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
    "host_continuation_evidence": continuation_evidence,
    "errors": errors,
}
gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
if errors:
    raise SystemExit("; ".join(errors))
PY

echo "candidate result: $run_dir/result.json"
echo "candidate gate:   $run_dir/gate.json"
