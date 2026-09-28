#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: qualify_mc2_multishape.sh <attention-capture-root> <down-capture-root> <run-tag>

Qualify Qwen3-32B TP3 native-MatMul + rank-3 chained epilogue on real inputs.
Each capture root must contain KIND/layer-{0,31,63}/mM/rank-{0,1,2}.pt.

The script screens every projection/M on layer 31 first, completes layers 0
and 63 independently for each passing projection/M, then emits one strict
conservative cross-layer, multi-shape production profile.

Optional environment:
  SPECSLO_TARGET_DEVICES     Three target NPUs (default: 1,2,3)
  SPECSLO_QUALIFY_ROWS       Exact M values (default: 40..128 step 4)
  SPECSLO_QUALIFY_ROOT       Result root
  SPECSLO_MC2_VENDOR         Installed MC2 vendor root
  SPECSLO_MC2_PROFILE_FILENAME  Output profile basename
  SPECSLO_QUALIFY_SCREEN_ONLY  Stop after layer-31 screening (default: 0)
  SPECSLO_QUALIFY_SAMPLES      Repeated latency samples per exact shape (default: 11)
  SPECSLO_QUALIFY_TIMEOUT_SECONDS  Whole persistent run timeout (default: 21600)
EOF
}

if [[ $# -ne 3 ]]; then
  usage >&2
  exit 2
fi

attention_root=$1
down_root=$2
run_tag=$3
if [[ ! $run_tag =~ ^[A-Za-z0-9._-]+$ || $run_tag == . || $run_tag == .. ]]; then
  echo "invalid run tag: $run_tag" >&2
  exit 2
fi

repo=/root/data/vllm-ascend-hust
python=/root/miniconda3/envs/vllm-hust-dev/bin/python
cann_env=/usr/local/Ascend/cann-9.0.0/set_env.sh
extension=$repo/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
base_vendor=$repo/vllm_ascend/_cann_ops_custom/vendors/custom_transformer
vendor=${SPECSLO_MC2_VENDOR:-/root/data/specslo-mc2-v134-large-m/package-extracted/packages/vendors/specslo_mc2_v134_large_m_transformer}
torch_lib=/root/miniconda3/envs/vllm-hust-dev/lib/python3.11/site-packages/torch/lib
torch_npu_lib=/root/miniconda3/envs/vllm-hust-dev/lib/python3.11/site-packages/torch_npu/lib
target_devices=${SPECSLO_TARGET_DEVICES:-1,2,3}
rows=${SPECSLO_QUALIFY_ROWS:-40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,104,108,112,116,120,124,128}
screen_only=${SPECSLO_QUALIFY_SCREEN_ONLY:-0}
samples=${SPECSLO_QUALIFY_SAMPLES:-11}
timeout_seconds=${SPECSLO_QUALIFY_TIMEOUT_SECONDS:-21600}
result_root=${SPECSLO_QUALIFY_ROOT:-/root/data/nano-pearl-benchmark-results/20260927-specslo-mc2-v134-large-m-qualification}
run_dir=$result_root/runs/$run_tag
status_file=$run_dir/qualification-status.tsv
profile_filename=${SPECSLO_MC2_PROFILE_FILENAME:-qwen3-32b-tp3-v134-large-m-production-profile.json}
if [[ $profile_filename == */* || ! $profile_filename =~ ^[A-Za-z0-9._-]+[.]json$ ]]; then
  echo "SPECSLO_MC2_PROFILE_FILENAME must be a JSON basename" >&2
  exit 2
fi
profile_output=$run_dir/$profile_filename

for required in "$repo" "$python" "$cann_env" "$extension" "$vendor" \
  "$attention_root/attention" "$down_root/down"; do
  if [[ ! -e $required ]]; then
    echo "required path is missing: $required" >&2
    exit 3
  fi
done
if [[ ! -d $vendor || ! -f $vendor/op_api/lib/libcust_opapi.so ]]; then
  echo "MC2 vendor is not an installed custom-op tree: $vendor" >&2
  exit 3
fi
if [[ -e $run_dir ]]; then
  echo "refusing to overwrite existing run: $run_dir" >&2
  exit 4
fi

device_list_pattern='^[[:space:]]*(0|[1-9][0-9]*)[[:space:]]*(,[[:space:]]*(0|[1-9][0-9]*)[[:space:]]*)*$'
if [[ ! $target_devices =~ $device_list_pattern ]]; then
  echo "SPECSLO_TARGET_DEVICES must be a canonical comma-separated NPU list" >&2
  exit 5
fi
IFS=',' read -r -a devices <<< "$target_devices"
if [[ ${#devices[@]} -ne 3 ]]; then
  echo "SPECSLO_TARGET_DEVICES must contain exactly three NPUs" >&2
  exit 5
fi
declare -A seen_devices=()
for index in "${!devices[@]}"; do
  device=${devices[index]}
  device=${device#"${device%%[![:space:]]*}"}
  device=${device%"${device##*[![:space:]]}"}
  if [[ -n ${seen_devices[$device]:-} ]]; then
    echo "target NPUs must be unique non-negative integers" >&2
    exit 5
  fi
  devices[index]=$device
  seen_devices[$device]=1
done
target_devices=$(IFS=,; echo "${devices[*]}")

row_list_pattern='^[[:space:]]*[1-9][0-9]*[[:space:]]*(,[[:space:]]*[1-9][0-9]*[[:space:]]*)*$'
if [[ ! $rows =~ $row_list_pattern ]]; then
  echo "SPECSLO_QUALIFY_ROWS must be a canonical comma-separated positive integer list" >&2
  exit 6
fi
IFS=',' read -r -a row_values <<< "$rows"
declare -A seen_rows=()
for index in "${!row_values[@]}"; do
  row=${row_values[index]}
  row=${row#"${row%%[![:space:]]*}"}
  row=${row%"${row##*[![:space:]]}"}
  if [[ -n ${seen_rows[$row]:-} ]]; then
    echo "SPECSLO_QUALIFY_ROWS must contain unique positive integers" >&2
    exit 6
  fi
  row_values[index]=$row
  seen_rows[$row]=1
done
if [[ $screen_only != 0 && $screen_only != 1 ]]; then
  echo "SPECSLO_QUALIFY_SCREEN_ONLY must be 0 or 1" >&2
  exit 6
fi
if [[ ! $samples =~ ^[1-9][0-9]*$ || $samples -lt 3 ]]; then
  echo "SPECSLO_QUALIFY_SAMPLES must be an integer >= 3" >&2
  exit 6
fi
if [[ ! $timeout_seconds =~ ^[1-9][0-9]*$ ]]; then
  echo "SPECSLO_QUALIFY_TIMEOUT_SECONDS must be a positive integer" >&2
  exit 6
fi

mkdir -p "$run_dir/measurements"
assert_cards_free() {
  local device snapshot
  npu-smi info > "$run_dir/npu-before.txt"
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

cd "$repo"
git rev-parse HEAD > "$run_dir/git-head.txt"
git status --porcelain=v1 --untracked-files=all > "$run_dir/git-status.txt"
git diff --binary > "$run_dir/working-tree.patch"
sha256sum "$repo/tools/specslo_experiments/qualify_mc2_multishape.sh" \
  "$repo/examples/measure_specslo_rank3_reduce_add_rmsnorm.py" \
  "$repo/examples/qualify_specslo_mc2_multishape.py" \
  "$repo/examples/aggregate_specslo_mc2_profiles.py" \
  > "$run_dir/qualification-tools.sha256"
find "$attention_root/attention" "$down_root/down" -type f -name 'rank-*.pt' \
  -print0 | sort -z | xargs -0 sha256sum > "$run_dir/capture-files.sha256"
{
  printf 'target_devices=%s\nrows=%s\nscreen_only=%s\nsamples=%s\ntimeout_seconds=%s\n' \
    "$target_devices" "$(IFS=,; echo "${row_values[*]}")" \
    "$screen_only" "$samples" "$timeout_seconds"
  printf 'result_root=%s\nrun_dir=%s\nprofile_filename=%s\n' \
    "$result_root" "$run_dir" "$profile_filename"
  printf 'attention_root=%s\ndown_root=%s\nvendor=%s\n' \
    "$attention_root" "$down_root" "$vendor"
  printf 'profile_output=%s\n' "$profile_output"
} > "$run_dir/contract.env"
env | LC_ALL=C sort > "$run_dir/environment.txt"

# shellcheck disable=SC1090
source "$cann_env"
export PYTHONPATH=$repo:${PYTHONPATH:-}
export ASCEND_RT_VISIBLE_DEVICES=$target_devices
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export HCCL_DETERMINISTIC=true
export ASCEND_CUSTOM_OPP_PATH=$base_vendor:$vendor
export LD_LIBRARY_PATH=$vendor/op_api/lib:$repo/vllm_ascend:$repo/vllm_ascend/lib64:$repo/build/temp.linux-aarch64-cpython-311:$torch_lib:$torch_npu_lib:/usr/local/Ascend/cann-9.0.0/lib64:${LD_LIBRARY_PATH:-}
export PYTHONHASHSEED=0
export PYTHONUNBUFFERED=1

profile_path() {
  printf '%s/measurements/%s/layer-%s/m%s/profile.json\n' \
    "$run_dir" "$1" "$2" "$3"
}

driver_args=(
  --attention-root "$attention_root"
  --down-root "$down_root"
  --run-dir "$run_dir"
  --rows "$(IFS=,; echo "${row_values[*]}")"
  --warmup-replays 40
  --replays-per-sample 50
  --samples "$samples"
  --changed-input-replays 1024
  --changed-input-delta 0.015625
  --latency-percentile 95
  --minimum-speedup 1.02
)
if [[ $screen_only == 1 ]]; then
  driver_args+=(--screen-only)
fi

set +e
timeout -k 30 "$timeout_seconds" "$python" -m torch.distributed.run \
  --standalone --nproc_per_node=3 -- \
  "$repo/examples/run_with_vllm_ascend_extension.py" \
  --extension "$extension" --required-op allreduce_add_rmsnorm_chained \
  "$repo/examples/qualify_specslo_mc2_multishape.py" \
  "${driver_args[@]}" > "$run_dir/persistent-run.log" 2>&1
driver_status=$?
set -e
npu-smi info > "$run_dir/npu-after.txt" || true
if [[ $driver_status -ne 0 ]]; then
  printf '%s\n' "$driver_status" > "$run_dir/driver-exit-status.txt"
  if [[ ! -e $status_file ]]; then
    printf 'kind\tlayer\tm\tstatus\n' > "$status_file"
  fi
  failure_kind=fatal
  if grep -Eiq \
    'Execution_Error_SDMA|vector core exception|DDR address of the MTE|error code is 507035|aicpu execute failed|HcclAllocComResourceByTiling|runtime result = 507018|HCCL' \
    "$run_dir/persistent-run.log"; then
    failure_kind=device-fail
  fi
  if [[ -s "$run_dir/current-job.tsv" ]]; then
    IFS=$'\t' read -r failed_projection failed_layer failed_row \
      < "$run_dir/current-job.tsv"
    if ! awk -F '\t' -v kind="$failed_projection" -v layer="$failed_layer" \
      -v row="$failed_row" \
      'NR > 1 && $1 == kind && $2 == layer && $3 == row { found = 1 } END { exit !found }' \
      "$status_file"; then
      printf '%s\t%s\t%s\t%s:torchrun-exit-%s\n' \
        "$failed_projection" "$failed_layer" "$failed_row" \
        "$failure_kind" "$driver_status" >> "$status_file"
    fi
  fi
  printf 'campaign\t-\t-\t%s:torchrun-exit-%s\n' \
    "$failure_kind" "$driver_status" >> "$status_file"
  printf 'persistent qualification exited with status %s\n' \
    "$driver_status" >&2
  exit 9
fi

if [[ $screen_only == 1 ]]; then
  "$python" - "$run_dir/screened-projection-rows.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
document = json.loads(path.read_text(encoding="utf-8"))
projection_rows = document.get("projection_rows")
if not isinstance(projection_rows, dict):
    raise SystemExit("layer-31 projection manifest is invalid")
if not any(projection_rows.get(kind) for kind in ("attention", "down")):
    raise SystemExit("no projection/M passed layer-31 screening")
print(
    "screened projection rows: "
    + ", ".join(
        f"{kind}={projection_rows.get(kind, [])}"
        for kind in ("attention", "down")
    )
)
PY
  exit 0
fi

# A projection/M can pass the latency gate on each layer independently yet
# fail the production envelope, which deliberately compares the slowest fused
# layer against the fastest baseline layer.  Screen every projection/M group
# with the same public aggregator used for the final profile before building
# the union.  Attention and down projections are deliberately independent.
"$python" - "$run_dir" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from examples.aggregate_specslo_mc2_profiles import aggregate_mc2_profiles

run_dir = Path(sys.argv[1])
manifest_path = run_dir / "qualified-projection-rows.json"
manifest_bytes = manifest_path.read_bytes()
manifest = json.loads(manifest_bytes)
expected_k = {"attention": 3072, "down": 8576}
schema_version = manifest.get("schema_version")
if isinstance(schema_version, bool) or schema_version != 1:
    raise SystemExit("projection qualification manifest requires schema_version=1")
if manifest.get("kind") != "specslo_mc2_projection_rows":
    raise SystemExit("projection qualification manifest has an invalid kind")
if manifest.get("stage") != "per-layer-qualification":
    raise SystemExit("projection qualification manifest has an invalid stage")
operations_per_graph = manifest.get("operations_per_graph")
if (
    manifest.get("execution_mode") != "graph"
    or isinstance(operations_per_graph, bool)
    or operations_per_graph != 1
):
    raise SystemExit("projection qualification manifest is not ACLGraph/one-operation evidence")
if manifest.get("qualification_layers") != [0, 31, 63]:
    raise SystemExit("projection qualification manifest does not cover layers 0/31/63")
if manifest.get("projection_k") != expected_k:
    raise SystemExit("projection qualification manifest has an invalid projection mapping")
candidate_projection_rows = manifest.get("projection_rows")
if not isinstance(candidate_projection_rows, dict) or set(candidate_projection_rows) != set(expected_k):
    raise SystemExit("projection qualification manifest has invalid projection rows")
for kind, values in candidate_projection_rows.items():
    if (
        not isinstance(values, list)
        or any(isinstance(row, bool) or not isinstance(row, int) or row <= 0 for row in values)
        or len(set(values)) != len(values)
    ):
        raise SystemExit(f"projection qualification manifest has invalid {kind} rows")

accepted = {kind: [] for kind in expected_k}
groups = []
for kind in expected_k:
    for row in candidate_projection_rows[kind]:
        paths = [
            run_dir / "measurements" / kind / f"layer-{layer}" / f"m{row}" / "profile.json"
            for layer in (0, 31, 63)
        ]
        error = None
        try:
            envelope = aggregate_mc2_profiles(paths)
            measurements = envelope.get("measurements")
            if not isinstance(measurements, list) or len(measurements) != 1:
                raise ValueError("cross-layer envelope does not contain one exact row")
            shape = tuple(measurements[0].get(field) for field in ("m", "k", "n"))
            expected_shape = (row, expected_k[kind], 5120)
            if shape != expected_shape:
                raise ValueError(
                    f"cross-layer envelope shape {shape} does not match {expected_shape}"
                )
        except (OSError, TypeError, ValueError) as exc:
            error = str(exc)
        else:
            accepted[kind].append(row)
        groups.append(
            {
                "m": row,
                "kind": kind,
                "status": "pass" if error is None else "fail",
                "error": error,
                "inputs": [str(path.resolve()) for path in paths],
            }
        )

accepted = {kind: sorted(values) for kind, values in accepted.items()}
intersection = sorted(set(accepted["attention"]).intersection(accepted["down"]))
candidate_intersection = sorted(
    set(candidate_projection_rows["attention"]).intersection(
        candidate_projection_rows["down"]
    )
)

report = {
    "schema_version": 1,
    "kind": "specslo_mc2_conservative_cross_layer_screen",
    "execution_mode": "graph",
    "operations_per_graph": 1,
    "qualification_layers": [0, 31, 63],
    "projection_k": expected_k,
    "candidate_projection_rows": candidate_projection_rows,
    "qualified_projection_rows": accepted,
    # Compatibility fields remain the two-projection intersection.
    "candidate_rows": candidate_intersection,
    "qualified_rows": intersection,
    "groups": groups,
}
(run_dir / "cross-layer-screen.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
(run_dir / "production-projection-rows.json").write_text(
    json.dumps(
        {
            "schema_version": 1,
            "kind": "specslo_mc2_projection_rows",
            "stage": "conservative-cross-layer-production",
            "execution_mode": "graph",
            "operations_per_graph": 1,
            "qualification_layers": [0, 31, 63],
            "projection_k": expected_k,
            "projection_rows": accepted,
            "intersection_rows": intersection,
            "source_manifest": str(manifest_path.resolve()),
            "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
(run_dir / "qualified-rows.txt").write_text(
    " ".join(map(str, intersection)) + "\n",
    encoding="utf-8",
)
if not any(accepted.values()):
    raise SystemExit("no projection/M passed the conservative cross-layer production envelope")
PY

qualified_shapes=()
while IFS=$'\t' read -r kind row; do
  qualified_shapes+=("$kind:$row")
done < <("$python" - "$run_dir/production-projection-rows.json" <<'PY'
import json
import sys
from pathlib import Path

document = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for kind in ("attention", "down"):
    for row in document["projection_rows"][kind]:
        print(f"{kind}\t{row}")
PY
)

aggregate_args=()
for shape in "${qualified_shapes[@]}"; do
  kind=${shape%%:*}
  row=${shape#*:}
  aggregate_args+=(--input)
  for layer in 0 31 63; do
    aggregate_args+=("$(profile_path "$kind" "$layer" "$row")")
  done
done
"$python" examples/aggregate_specslo_mc2_profiles.py \
  "${aggregate_args[@]}" --output "$profile_output" \
  > "$run_dir/aggregate.log" 2>&1

sha256sum "$profile_output" > "$run_dir/profile.sha256"
npu-smi info > "$run_dir/npu-after.txt" || true
echo "qualified projection rows: ${qualified_shapes[*]}"
echo "qualified row intersection: $(<"$run_dir/qualified-rows.txt")"
echo "production profile: $profile_output"
