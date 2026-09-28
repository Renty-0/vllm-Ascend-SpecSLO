#!/usr/bin/env bash
set -euo pipefail

# Run the formal three-round SpecSLO MC2 system A/B on one fixed four-NPU
# topology.  Compile/qualification warmup remains outside each benchmark's
# measured E2E interval.  A campaign passes only when every paired candidate
# has both Raw(candidate) > Raw(TP4 baseline) and paper Goodput(candidate) >=
# 1.3 * paper Goodput(TP4 baseline), with the complete workload, source,
# profile, physical-device, and graph-no-fallback contracts intact.

usage() {
  cat >&2 <<'EOF'
usage: run_online_coalesce1_formal_ab.sh [--dry-run|--validate-only] <campaign-tag>

Environment:
  SPECSLO_VISIBLE_DEVICES   Four physical NPUs, default: 0,1,2,3
  SPECSLO_RESULT_ROOT       Result root shared by the child launchers
  SPECSLO_MC2_PROFILE       Current-source TP3 MC2 production profile
  SPECSLO_MC2_VENDOR        Installed current-source MC2 vendor root
  SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS
                            Draft PA update workers (1, 2, or 4; default: 1)
  SPECSLO_TARGET_REPLAY_AHEAD
                            Early target graph replay (0 or 1; default: 0)

The launch order is deliberately interleaved:
  round 1: TP4 baseline -> TP1+TP3 candidate
  round 2: TP1+TP3 candidate -> TP4 baseline
  round 3: TP4 baseline -> TP1+TP3 candidate

--dry-run prints the immutable plan and never queries or launches an NPU.
--validate-only re-runs all artifact and performance gates without an NPU.
EOF
  exit 2
}

mode=run
if [[ ${1:-} == --dry-run ]]; then
  mode=dry-run
  shift
elif [[ ${1:-} == --validate-only ]]; then
  mode=validate-only
  shift
fi
[[ $# -eq 1 ]] || usage

campaign_tag=$1
if [[ ! $campaign_tag =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid campaign tag: $campaign_tag" >&2
  exit 2
fi

repo=/root/data/vllm-ascend-hust
python=/root/miniconda3/envs/vllm-hust-dev/bin/python
candidate_script=$repo/tools/specslo_experiments/run_online_coalesce1_candidate.sh
baseline_script=$repo/tools/specslo_experiments/run_online_coalesce1_baseline.sh
baseline_wrapper=/root/data/nano-pearl-benchmark-results/20260916-qwen3-four-spec-modes-author-workload-slo40-50-150/run_baseline_same_target_cards.sh
manifest=/root/data/specslo-workloads/eurosys27-rps4-seed0-author-slo40-50-150/rps4.0_mix0.6_0.2_0.2_seed0.jsonl
provenance=${manifest%/*}/provenance.json
result_root=${SPECSLO_RESULT_ROOT:-/root/data/nano-pearl-benchmark-results/20260927-specslo-v134-large-m-online-coalesce1}
profile=${SPECSLO_MC2_PROFILE:-/root/data/nano-pearl-benchmark-results/20260927-specslo-mc2-v134-large-m-qualification/runs/production-current-target345-v134-large-m-s31/qwen3-32b-tp3-v134-large-m-production-profile.json}
vendor=${SPECSLO_MC2_VENDOR:-/root/data/specslo-mc2-v134-large-m/package-extracted/packages/vendors/specslo_mc2_v134_large_m_transformer}
extension=$repo/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so
visible_devices=${SPECSLO_VISIBLE_DEVICES:-0,1,2,3}
campaign_dir=$result_root/formal-ab-$campaign_tag
draft_pa_task_update_workers=${SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS:-1}
target_replay_ahead=${SPECSLO_TARGET_REPLAY_AHEAD:-0}

IFS=',' read -r -a devices <<< "$visible_devices"
if [[ ${#devices[@]} -ne 4 ]]; then
  echo "exactly four physical NPUs are required" >&2
  exit 2
fi
declare -A unique_devices=()
for device in "${devices[@]}"; do
  if [[ ! $device =~ ^[0-9]+$ ]]; then
    echo "invalid physical NPU id: $device" >&2
    exit 2
  fi
  if [[ -n ${unique_devices[$device]:-} ]]; then
    echo "physical NPU ids must be unique: $visible_devices" >&2
    exit 2
  fi
  unique_devices[$device]=1
done
device_label=cards${visible_devices//,/-}

if [[ $draft_pa_task_update_workers != 1 && \
      $draft_pa_task_update_workers != 2 && \
      $draft_pa_task_update_workers != 4 ]]; then
  echo "SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS must be 1, 2, or 4" >&2
  exit 2
fi
if [[ $target_replay_ahead != 0 && $target_replay_ahead != 1 ]]; then
  echo "SPECSLO_TARGET_REPLAY_AHEAD must be 0 or 1" >&2
  exit 2
fi

round_tag() {
  local round=$1
  printf '%s-r%s' "$campaign_tag" "$round"
}

baseline_run_dir() {
  local round=$1
  printf '%s/runs/baseline-target-only-tp4-%s-%s' \
    "$result_root" "$device_label" "$(round_tag "$round")"
}

candidate_run_dir() {
  local round=$1
  printf '%s/runs/candidate-on-%s' "$result_root" "$(round_tag "$round")"
}

print_plan() {
  cat <<EOF
campaign_tag=$campaign_tag
campaign_dir=$campaign_dir
result_root=$result_root
physical_npus=$visible_devices
manifest=$manifest
profile=$profile
vendor=$vendor
baseline_wrapper=$baseline_wrapper
draft_pa_task_update_workers=$draft_pa_task_update_workers
target_replay_ahead=$target_replay_ahead
contract=online Poisson RPS4; coalesce1/wait0; gamma4; P463/B64/T256; SLO 40/50/150 ms
round_1=baseline,candidate
round_2=candidate,baseline
round_3=baseline,candidate
round_gate=Raw(candidate)>Raw(baseline) AND paper_Goodput(candidate)>=1.3*paper_Goodput(baseline)
EOF
}

if [[ $mode == dry-run ]]; then
  print_plan
  for round in 1 2 3; do
    printf 'round_%s_baseline_dir=%s\n' "$round" "$(baseline_run_dir "$round")"
    printf 'round_%s_candidate_dir=%s\n' "$round" "$(candidate_run_dir "$round")"
  done
  exit 0
fi

for required in "$repo" "$python" "$candidate_script" "$baseline_script" \
  "$baseline_wrapper" "$manifest" "$provenance" "$profile" "$vendor" \
  "$extension"; do
  if [[ ! -e $required ]]; then
    echo "required path is missing: $required" >&2
    exit 2
  fi
done
if [[ ! -d $vendor || ! -f $vendor/op_api/lib/libcust_opapi.so ]]; then
  echo "MC2 vendor is not an installed custom-op tree: $vendor" >&2
  exit 2
fi

snapshot_source() {
  local output=$1
  "$python" - "$repo" "$output" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

repo = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2])
source_suffixes = {
    ".c", ".cc", ".cpp", ".h", ".hpp", ".json", ".py", ".sh",
    ".toml", ".yaml", ".yml",
}


def git(*arguments: str) -> bytes:
    return subprocess.check_output(("git", "-C", str(repo), *arguments))


head = git("rev-parse", "HEAD").decode().strip()
diff = git("diff", "--binary", "--no-ext-diff")
status = git("status", "--porcelain=v1", "--untracked-files=all")
paths = sorted(
    {
        Path(raw.decode())
        for raw in git(
            "ls-files", "--cached", "--others", "--exclude-standard", "-z"
        ).split(b"\0")
        if raw and Path(raw.decode()).suffix in source_suffixes
    },
    key=lambda path: path.as_posix(),
)
sources = []
for relative in paths:
    data = (repo / relative).read_bytes()
    sources.append(
        {
            "path": relative.as_posix(),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    )
source_manifest = json.dumps(sources, sort_keys=True, separators=(",", ":")).encode()
components = {
    "git_head": head,
    "git_diff_sha256": hashlib.sha256(diff).hexdigest(),
    "git_status_sha256": hashlib.sha256(status).hexdigest(),
    "source_manifest_sha256": hashlib.sha256(source_manifest).hexdigest(),
}
identity_payload = json.dumps(
    components, sort_keys=True, separators=(",", ":")
).encode()
document = {
    **components,
    "identity_sha256": hashlib.sha256(identity_payload).hexdigest(),
    "source_files": sources,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
PY
}

assert_source_unchanged() {
  local label=$1
  local snapshot=$campaign_dir/source-$label.json
  snapshot_source "$snapshot"
  "$python" - "$campaign_dir/source-initial.json" "$snapshot" <<'PY'
import json
import sys
from pathlib import Path

expected_path, actual_path = map(Path, sys.argv[1:3])
expected = json.loads(expected_path.read_text())
actual = json.loads(actual_path.read_text())
if expected["identity_sha256"] != actual["identity_sha256"]:
    raise SystemExit(
        "repository source identity changed during the formal A/B: "
        f"expected={expected['identity_sha256']} actual={actual['identity_sha256']}"
    )
PY
}

write_input_identity() {
  "$python" - "$repo" "$manifest" "$provenance" "$profile" \
    "$baseline_wrapper" "$extension" "$vendor" "$visible_devices" \
    "$draft_pa_task_update_workers" "$target_replay_ahead" \
    "$campaign_dir/input-identity.json" <<'PY'
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

repo, manifest_path, provenance_path, profile_path, baseline_wrapper, extension, vendor = map(
    Path, sys.argv[1:8]
)
devices = sys.argv[8].split(",")
draft_pa_task_update_workers = int(sys.argv[9])
target_replay_ahead = int(sys.argv[10])
output_path = Path(sys.argv[11])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def directory_identity(path: Path) -> tuple[str, int]:
    files = []
    for child in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        if not child.is_file():
            continue
        files.append(
            {
                "path": child.relative_to(path).as_posix(),
                "sha256": sha256(child),
                "size": child.stat().st_size,
            }
        )
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), len(files)


manifest_bytes = manifest_path.read_bytes()
manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
rows = [json.loads(line) for line in manifest_bytes.decode().splitlines() if line.strip()]
errors = []
if len(rows) != 463:
    errors.append(f"manifest rows={len(rows)}, expected 463")
expected_classes = {
    "tight": ("coding", 271, 40.0),
    "normal": ("chat", 96, 50.0),
    "loose": ("summarization", 96, 150.0),
}
class_counts = Counter(row.get("slo_class") for row in rows)
category_counts = Counter(row.get("category") for row in rows)
for slo_class, (category, count, threshold) in expected_classes.items():
    selected = [row for row in rows if row.get("slo_class") == slo_class]
    if len(selected) != count:
        errors.append(f"{slo_class} rows={len(selected)}, expected {count}")
    if any(row.get("category") != category for row in selected):
        errors.append(f"{slo_class} rows are not uniformly {category}")
    if any(float(row.get("slo_tpot_ms", math.nan)) != threshold for row in selected):
        errors.append(f"{slo_class} SLO is not uniformly {threshold} ms")
if any(int(row.get("per_request_gamma", -1)) != 4 for row in rows):
    errors.append("manifest gamma is not uniformly 4")
if any(int(row.get("max_tokens", -1)) != 256 for row in rows):
    errors.append("manifest max_tokens is not uniformly 256")
arrivals = [float(row.get("arrival_offset_sec", math.nan)) for row in rows]
if any(not math.isfinite(value) for value in arrivals):
    errors.append("manifest has a non-finite arrival offset")
if any(right <= left for left, right in zip(arrivals, arrivals[1:])):
    errors.append("manifest arrivals are not strictly increasing")

provenance = json.loads(provenance_path.read_text())
workload = provenance.get("workload", {})
derived = provenance.get("derived_workload", {})
if provenance.get("source_commit") != "22c196bdf571e84887e5573607bd8e734faa0625":
    errors.append("unexpected workload source commit")
if derived.get("sha256") != manifest_sha:
    errors.append("provenance derived-workload SHA256 mismatch")
if workload.get("arrival_process") != "homogeneous_poisson_exponential_interarrivals":
    errors.append("workload is not identified as homogeneous Poisson arrivals")
if float(workload.get("rps", math.nan)) != 4.0:
    errors.append("workload RPS is not 4.0")
if int(workload.get("arrival_seed", -1)) != 0:
    errors.append("workload arrival seed is not 0")

profile = json.loads(profile_path.read_text())
metadata = profile.get("metadata", {})
runtime_binding = metadata.get("runtime_binding", {})
expected_mapping = devices[1:]
if int(metadata.get("tensor_parallel_size", -1)) != 3:
    errors.append("MC2 profile is not TP3")
if runtime_binding.get("tp_rank_device_mapping") != expected_mapping:
    errors.append(
        "MC2 profile target mapping does not match candidate target cards: "
        f"profile={runtime_binding.get('tp_rank_device_mapping')} "
        f"expected={expected_mapping}"
    )

# Import only the pure source-identity helper.  This does not initialize an NPU.
sys.path.insert(0, str(repo))
from vllm_ascend.spec_decode.pearl.mc2 import mc2_source_sha256  # noqa: E402
from examples.specslo_benchmark_report import (  # noqa: E402
    qualified_mc2_profile_replay_contract,
)

current_mc2_source_sha = mc2_source_sha256()
profile_mc2_source_sha = metadata.get("source_sha256")
if profile_mc2_source_sha != current_mc2_source_sha:
    errors.append(
        "MC2 profile source SHA256 is stale: "
        f"profile={profile_mc2_source_sha} current={current_mc2_source_sha}"
    )
mc2_replay_profile_contract = qualified_mc2_profile_replay_contract(
    profile_path,
)
vendor_sha, vendor_file_count = directory_identity(vendor)
if vendor_file_count == 0:
    errors.append("MC2 vendor tree is empty")
document = {
    "status": "pass" if not errors else "fail",
    "physical_npus": [int(value) for value in devices],
    "draft_pa_task_update_workers": draft_pa_task_update_workers,
    "target_replay_ahead": target_replay_ahead,
    "manifest": str(manifest_path.resolve()),
    "manifest_sha256": manifest_sha,
    "manifest_class_counts": dict(sorted(class_counts.items())),
    "manifest_category_counts": dict(sorted(category_counts.items())),
    "provenance": str(provenance_path.resolve()),
    "arrival_process": workload.get("arrival_process"),
    "arrival_rps": workload.get("rps"),
    "arrival_seed": workload.get("arrival_seed"),
    "profile": str(profile_path.resolve()),
    "profile_sha256": sha256(profile_path),
    "profile_source_sha256": profile_mc2_source_sha,
    "current_mc2_source_sha256": current_mc2_source_sha,
    "mc2_replay_profile_contract": mc2_replay_profile_contract,
    "profile_tp_rank_device_mapping": runtime_binding.get("tp_rank_device_mapping"),
    "baseline_wrapper": str(baseline_wrapper.resolve()),
    "baseline_wrapper_sha256": sha256(baseline_wrapper),
    "extension": str(extension.resolve()),
    "extension_sha256": sha256(extension),
    "vendor": str(vendor.resolve()),
    "vendor_sha256": vendor_sha,
    "vendor_file_count": vendor_file_count,
    "errors": errors,
}
output_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
if errors:
    raise SystemExit("; ".join(errors))
PY
}

assert_inputs_unchanged() {
  "$python" - "$campaign_dir/input-identity.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

identity = json.loads(Path(sys.argv[1]).read_text())
errors = []


def directory_identity(path: Path) -> tuple[str, int]:
    files = []
    for child in sorted(path.rglob("*"), key=lambda item: item.as_posix()):
        if not child.is_file():
            continue
        files.append(
            {
                "path": child.relative_to(path).as_posix(),
                "sha256": hashlib.sha256(child.read_bytes()).hexdigest(),
                "size": child.stat().st_size,
            }
        )
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest(), len(files)


for name in ("manifest", "profile", "baseline_wrapper", "extension"):
    path = Path(identity[name])
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        errors.append(f"cannot re-read {name} {path}: {error}")
        continue
    expected = identity[f"{name}_sha256"]
    if actual != expected:
        errors.append(
            f"{name} identity changed during formal A/B: "
            f"expected={expected} actual={actual}"
        )
vendor_path = Path(identity["vendor"])
try:
    vendor_sha, vendor_file_count = directory_identity(vendor_path)
except OSError as error:
    errors.append(f"cannot re-read vendor {vendor_path}: {error}")
else:
    if vendor_sha != identity["vendor_sha256"]:
        errors.append(
            "vendor identity changed during formal A/B: "
            f"expected={identity['vendor_sha256']} actual={vendor_sha}"
        )
    if vendor_file_count != identity["vendor_file_count"]:
        errors.append(
            "vendor file count changed during formal A/B: "
            f"expected={identity['vendor_file_count']} actual={vendor_file_count}"
        )
if errors:
    raise SystemExit("; ".join(errors))
PY
}

validate_round() {
  local round=$1
  local baseline_dir candidate_dir output
  baseline_dir=$(baseline_run_dir "$round")
  candidate_dir=$(candidate_run_dir "$round")
  output=$campaign_dir/round-$round-gate.json
  "$python" - "$round" "$visible_devices" \
    "$campaign_dir/source-initial.json" "$campaign_dir/input-identity.json" \
    "$baseline_dir" "$candidate_dir" "$output" <<'PY'
import hashlib
import json
import math
import sys
from pathlib import Path

round_number = int(sys.argv[1])
devices = [int(value) for value in sys.argv[2].split(",")]
source_identity_path, input_identity_path, baseline_dir, candidate_dir, output_path = map(
    Path, sys.argv[3:8]
)
source_identity = json.loads(source_identity_path.read_text())
input_identity = json.loads(input_identity_path.read_text())
errors = []


def load(path: Path, label: str):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"cannot read {label} {path}: {error}")
        return {}


def require(condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def finite_number(value, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        errors.append(f"{label} is not numeric")
        return math.nan
    if not math.isfinite(result):
        errors.append(f"{label} is not finite")
    return result


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_sha256sum(path: Path) -> dict[str, str]:
    result = {}
    try:
        lines = path.read_text().splitlines()
    except OSError as error:
        errors.append(f"cannot read SHA256 manifest {path}: {error}")
        return result
    for line in lines:
        digest, separator, name = line.partition("  ")
        if not separator or len(digest) != 64:
            errors.append(f"malformed sha256sum row in {path}: {line!r}")
            continue
        result[str(Path(name).resolve())] = digest
    return result


baseline = load(baseline_dir / "result.json", "baseline result")
baseline_graph = load(baseline_dir / "graph-gate.json", "baseline graph gate")
candidate = load(candidate_dir / "result.json", "candidate result")
candidate_gate = load(candidate_dir / "gate.json", "candidate gate")
if errors:
    output_path.write_text(json.dumps({"status": "fail", "errors": errors}, indent=2) + "\n")
    raise SystemExit("; ".join(errors))

baseline_rows = baseline.get("results", [])
candidate_rows = candidate.get("results", [])
require(len(baseline_rows) == 1, "baseline must contain exactly one result row")
require(len(candidate_rows) == 1, "candidate must contain exactly one result row")
baseline_row = baseline_rows[0] if len(baseline_rows) == 1 else {}
candidate_row = candidate_rows[0] if len(candidate_rows) == 1 else {}

manifest_path = str(Path(input_identity["manifest"]).resolve())
manifest_sha = input_identity["manifest_sha256"]
profile_path = str(Path(input_identity["profile"]).resolve())
profile_sha = input_identity["profile_sha256"]
vendor_path = str(Path(input_identity["vendor"]).resolve())
vendor_sha = input_identity["vendor_sha256"]
extension_path = str(Path(input_identity["extension"]).resolve())
extension_sha = input_identity["extension_sha256"]
expected_draft_pa_task_update_workers = int(
    input_identity.get("draft_pa_task_update_workers", -1)
)
expected_target_replay_ahead = int(input_identity.get("target_replay_ahead", -1))

# Exact workload and execution envelopes.
require(baseline.get("backend") == "vllm-ascend-target-only", "baseline backend mismatch")
require(baseline.get("model") == "/data/shared-models/Qwen3-32B", "baseline model mismatch")
require(baseline.get("tensor_parallel_size") == 4, "baseline is not TP4")
require(baseline.get("max_num_seqs") == 64, "baseline max_num_seqs is not 64")
require(baseline.get("max_tokens") == 256, "baseline max_tokens is not 256")
require(baseline.get("online_arrivals") is True, "baseline is not online arrivals")
require(baseline.get("prefill_coalesce_min_requests") == 1, "baseline is not coalesce1")
require(float(baseline.get("prefill_coalesce_max_wait_ms", math.nan)) == 0.0,
        "baseline coalesce wait is not 0 ms")
require(baseline.get("respect_eos") is False, "baseline is not fixed T256")

require(candidate.get("backend") == "specslo-native-specrhythm", "candidate backend mismatch")
require(candidate.get("draft_model") == "/data/shared-models/Qwen3-0.6B", "candidate draft model mismatch")
require(candidate.get("target_model") == "/data/shared-models/Qwen3-32B", "candidate target model mismatch")
require(candidate.get("draft_tensor_parallel_size") == 1, "candidate draft is not TP1")
require(candidate.get("target_tensor_parallel_size") == 3, "candidate target is not TP3")
require(candidate.get("draft_mode") == "serial_linear", "candidate is not serial chain draft")
require(candidate.get("gamma") == 4, "candidate gamma is not 4")
require(candidate.get("spec_rhythm_min_gamma") == 4, "candidate minimum gamma is not 4")
require(candidate.get("spec_rhythm_request_max_gamma") == 4, "candidate request gamma is not 4")
require(candidate.get("spec_rhythm_tree_width") == 1, "candidate tree width is not 1")
require(candidate.get("spec_rhythm_tree_depth") == 1, "candidate tree depth is not 1")
require(candidate.get("max_tokens") == 256, "candidate max_tokens is not 256")
require(candidate.get("online_arrivals") is True, "candidate is not online arrivals")
require(candidate.get("arrival_mode") == "manifest_trace", "candidate arrival mode mismatch")
require(candidate.get("spec_rhythm_online_prefill") is True, "candidate online prefill is disabled")
require(candidate.get("spec_rhythm_prefill_coalesce_min_requests") == 1,
        "candidate is not coalesce1")
require(float(candidate.get("spec_rhythm_prefill_coalesce_max_wait_ms", math.nan)) == 0.0,
        "candidate coalesce wait is not 0 ms")
require(candidate.get("respect_eos") is False, "candidate is not fixed T256")
require(candidate.get("enable_mc2") is True, "candidate MC2 is not enabled")
require(str(Path(candidate.get("mc2_profile", "")).resolve()) == profile_path,
        "candidate MC2 profile path mismatch")
candidate_environment = candidate.get("runtime_environment", {})
require(candidate_environment.get("VLLM_ASCEND_SPECRHYTHM_DRAIN_FIFO") == "1",
        "candidate drain-FIFO policy is not enabled")
require(
    candidate_environment.get("VLLM_ASCEND_PEARL_DRAFT_PA_TASK_UPDATE_WORKERS")
    == str(expected_draft_pa_task_update_workers),
    "candidate draft-PA task-update worker runtime contract mismatch",
)
require(
    candidate_environment.get("VLLM_ASCEND_SPECRHYTHM_TARGET_REPLAY_AHEAD")
    == str(expected_target_replay_ahead),
    "candidate target replay-ahead runtime contract mismatch",
)

for label, payload, row in (
    ("baseline", baseline, baseline_row),
    ("candidate", candidate, candidate_row),
):
    require(str(Path(payload.get("request_manifest", "")).resolve()) == manifest_path,
            f"{label} manifest path mismatch")
    require(payload.get("request_manifest_sha256") == manifest_sha,
            f"{label} manifest SHA256 mismatch")
    require(payload.get("requested_output_tokens") == 463 * 256,
            f"{label} requested output tokens mismatch")
    require(row.get("batch_size") == 64, f"{label} batch size is not 64")
    require(row.get("num_prompts") == 463, f"{label} prompt count is not 463")
    require(row.get("output_tokens") == 463 * 256,
            f"{label} output tokens are not fixed 463x256")
    limits = row.get("request_output_token_limits", [])
    require(len(limits) == 463 and all(value == 256 for value in limits),
            f"{label} request limits are not fixed 463x256")
    token_rows = row.get("output_token_ids")
    valid_token_rows = (
        isinstance(token_rows, list)
        and len(token_rows) == 463
        and all(
            isinstance(tokens, list)
            and len(tokens) == 256
            and all(isinstance(token, int) and not isinstance(token, bool) for token in tokens)
            for tokens in token_rows
        )
    )
    require(valid_token_rows, f"{label} output token rows are not fixed 463x256 integers")
    if valid_token_rows:
        recomputed_output_sha = hashlib.sha256(
            json.dumps(token_rows, separators=(",", ":")).encode()
        ).hexdigest()
        require(
            row.get("output_token_ids_sha256") == recomputed_output_sha,
            f"{label} stored output-token SHA256 differs from the serialized rows",
        )
    paper = row.get("slo", {}).get("paper", {})
    require(paper.get("constrained_requests") == 463,
            f"{label} paper SLO does not cover 463 requests")
    e2e = finite_number(row.get("e2e_elapsed_seconds"), f"{label} E2E seconds")
    output_tokens = finite_number(row.get("output_tokens"), f"{label} output tokens")
    raw_field = (
        "output_throughput_tokens_per_second"
        if label == "baseline"
        else "e2e_throughput_tokens_per_second"
    )
    raw = finite_number(row.get(raw_field), f"{label} Raw")
    goodput_tokens = finite_number(paper.get("goodput_tokens"), f"{label} paper goodput tokens")
    goodput = finite_number(
        paper.get("goodput_tokens_per_e2e_second"), f"{label} paper Goodput"
    )
    if all(math.isfinite(value) for value in (e2e, output_tokens, raw)):
        require(math.isclose(raw, output_tokens / e2e, rel_tol=1e-9, abs_tol=1e-9),
                f"{label} Raw is not output_tokens/E2E")
    if all(math.isfinite(value) for value in (e2e, goodput_tokens, goodput)):
        require(math.isclose(goodput, goodput_tokens / e2e, rel_tol=1e-9, abs_tol=1e-9),
                f"{label} paper Goodput is not goodput_tokens/E2E")

# Baseline measured graph interval must contain replay and no capture/fallback.
require(baseline_graph.get("status") == "pass", "baseline graph gate failed")
require(baseline_graph.get("errors") == [], "baseline graph gate contains errors")
require(baseline_graph.get("physical_npus") == devices, "baseline physical NPU mapping mismatch")
require(baseline_graph.get("manifest_sha256") == manifest_sha,
        "baseline graph gate manifest SHA256 mismatch")
require(baseline_graph.get("saturated_arrivals") is False,
        "baseline graph gate is not online Poisson mode")
require(baseline_graph.get("fixed_output_tokens") is True,
        "baseline graph gate is not fixed T256")
require(int(baseline_graph.get("measured_aclgraph_replays", 0)) > 0,
        "baseline measured ACLGraph replays are zero")
for field in (
    "measured_aclgraph_captures",
    "measured_decode_eager_fallbacks",
    "measured_prefill_or_unclassified_eager_calls",
    "measured_nonuniform_eager_fallbacks",
):
    require(int(baseline_graph.get(field, -1)) == 0, f"baseline {field} is nonzero or missing")

# Candidate graph qualification and all measured fallback counters are strict.
require(candidate_gate.get("status") == "pass", "candidate launcher gate failed")
require(candidate_gate.get("mode") == "on", "candidate launcher gate is not MC2-on")
require(candidate_gate.get("errors") == [], "candidate launcher gate contains errors")
draft_pa_evidence = candidate_gate.get("draft_pa_task_update", {})
require(isinstance(draft_pa_evidence, dict),
        "candidate launcher gate lacks draft-PA update evidence")
if isinstance(draft_pa_evidence, dict):
    require(draft_pa_evidence.get("passed") is True,
            "candidate draft-PA task-update evidence failed")
    require(draft_pa_evidence.get("errors") == [],
            "candidate draft-PA task-update evidence contains errors")
    require(
        int(draft_pa_evidence.get("expected_draft_workers", -1))
        == expected_draft_pa_task_update_workers,
        "candidate draft-PA evidence worker count differs from campaign input",
    )
    require(int(draft_pa_evidence.get("expected_target_workers", -1)) == 1,
            "candidate draft-PA evidence does not keep target workers at one")
    require(set(draft_pa_evidence.get("ranks", {})) == {"0", "1", "2", "3"},
            "candidate draft-PA evidence does not cover ranks 0,1,2,3")
target_replay_evidence = candidate_gate.get("target_replay_ahead", {})
require(isinstance(target_replay_evidence, dict),
        "candidate launcher gate lacks target replay-ahead evidence")
if isinstance(target_replay_evidence, dict):
    require(target_replay_evidence.get("passed") is True,
            "candidate target replay-ahead evidence failed")
    require(target_replay_evidence.get("errors") == [],
            "candidate target replay-ahead evidence contains errors")
    require(target_replay_evidence.get("enabled") is bool(expected_target_replay_ahead),
            "candidate target replay-ahead enablement mismatch")
    require(set(target_replay_evidence.get("ranks", {})) == {"0", "1", "2", "3"},
            "candidate target replay-ahead evidence does not cover ranks 0,1,2,3")
require(candidate_row.get("graph_qualification_fixed_point") is True,
        "candidate graph qualification did not reach a fixed point")
require(int(candidate_row.get("aclgraph_replays", 0)) > 0,
        "candidate ACLGraph replays are zero")
for field in (
    "aclgraph_failed_captures",
    "aclgraph_capacity_fallbacks",
    "aclgraph_shape_fallbacks",
):
    require(int(candidate_row.get(field, -1)) == 0,
            f"candidate {field} is nonzero or missing")
workers = candidate_row.get("worker_aclgraph_metrics", [])
targets = [worker for worker in workers if int(worker.get("is_draft_rank", 0)) == 0]
require(len(targets) == 3, "candidate does not report three target workers")
for worker in workers:
    rank = worker.get("rank")
    require(int(worker.get("worker_spec_rhythm_drain_fifo_enabled", -1)) == 1,
            f"candidate worker rank {rank} did not enable drain-FIFO")
    require(int(worker.get("worker_spec_rhythm_drain_fifo_cycles", 0)) > 0,
            f"candidate worker rank {rank} has no drain-FIFO cycles")
drain_lrtf_gate = candidate_gate.get("drain_lrtf", {})
if int(drain_lrtf_gate.get("enabled", 0)):
    for field in (
        "target_ranked_cycles_by_rank",
        "normal_ranked_cycles_by_rank",
        "unified_cycles_by_rank",
        "target_candidates_by_rank",
        "normal_candidates_by_rank",
        "repair_penalty_rows_by_rank",
        "normal_selected_by_rank",
        "normal_cross_home_selected_by_rank",
    ):
        values = drain_lrtf_gate.get(field, {})
        require(len(values) == len(workers),
                f"candidate drain-LRTF {field} does not cover every worker")
        require(all(int(value) > 0 for value in values.values()),
                f"candidate drain-LRTF {field} contains an unhit worker")
graph_fallback_fields = (
    "worker_spec_rhythm_mixed_target_graph_bounded_shape_fallback_batches",
    "worker_spec_rhythm_mixed_target_graph_token_capacity_fallback_batches",
    "worker_spec_rhythm_mixed_target_graph_execution_fallback_batches",
    "worker_spec_rhythm_stable_target_verify_graph_bounded_shape_fallback_batches",
    "worker_spec_rhythm_stable_target_verify_graph_token_capacity_fallback_batches",
    "worker_spec_rhythm_stable_target_verify_graph_execution_fallback_batches",
)
for worker in targets:
    rank = worker.get("rank")
    require(worker.get("spec_rhythm_mixed_target_graph_enabled") == 1,
            f"candidate target rank {rank} mixed-target graph is disabled")
    require(worker.get("spec_rhythm_stable_target_verify_graph_enabled") == 1,
            f"candidate target rank {rank} stable verify graph is disabled")
    require(worker.get("spec_rhythm_stable_target_verify_graph_qualified") == 1,
            f"candidate target rank {rank} stable verify graph is unqualified")
    require(int(worker.get("worker_spec_rhythm_mixed_target_graph_batches", 0)) > 0,
            f"candidate target rank {rank} has no mixed-target graph batches")
    require(int(worker.get("worker_spec_rhythm_stable_target_verify_graph_batches", 0)) > 0,
            f"candidate target rank {rank} has no stable verify graph batches")
    for field in (
        "aclgraph_failed_captures",
        "aclgraph_capacity_fallbacks",
        "aclgraph_shape_fallbacks",
    ):
        require(int(worker.get(field, -1)) == 0,
                f"candidate target rank {rank} {field} is nonzero or missing")
    for field in graph_fallback_fields:
        require(int(worker.get(field, -1)) == 0,
                f"candidate target rank {rank} {field} is nonzero or missing")
    eager_fallback_fields = [
        name
        for name in worker
        if name.startswith("aclgraph_")
        and (
            name.endswith("eager_fallback_calls")
            or name.endswith("eager_fallback_calls_delta")
        )
    ]
    require(bool(eager_fallback_fields),
            f"candidate target rank {rank} lacks measured ACLGraph fallback counters")
    for field in eager_fallback_fields:
        require(int(worker[field]) == 0,
                f"candidate target rank {rank} {field} is nonzero")

require(int(candidate_gate.get("mc2_fused_success", 0)) > 0,
        "candidate MC2 never dispatched successfully")
require(int(candidate_gate.get("mc2_fallback", -1)) == 0,
        "candidate MC2 fallback is nonzero or missing")
require(int(candidate_gate.get("mc2_exception", -1)) == 0,
        "candidate MC2 exception is nonzero or missing")
require(int(candidate_gate.get("measured_mc2_fallback", -1)) == 0,
        "candidate measured MC2 fallback is nonzero or missing")
require(int(candidate_gate.get("measured_mc2_exception", -1)) == 0,
        "candidate measured MC2 exception is nonzero or missing")

# A fused-success counter proves only that capture/warmup entered MC2.  Bind
# the timed graph replays to the exact profile-qualified M rows and frozen
# static-route manifest before accepting any system-level speedup.
profile_replay_contract = input_identity.get("mc2_replay_profile_contract")
candidate_replay_contract = candidate.get("mc2_replay_profile_contract")
row_replay_coverage = candidate_row.get("mc2_measured_replay_coverage")
gate_replay_coverage = candidate_gate.get("mc2_measured_replay_coverage")
require(isinstance(profile_replay_contract, dict),
        "campaign lacks an MC2 replay profile contract")
require(candidate_replay_contract == profile_replay_contract,
        "candidate MC2 replay profile contract differs from campaign input")
require(row_replay_coverage == gate_replay_coverage,
        "candidate result/gate MC2 replay evidence differs")
if isinstance(gate_replay_coverage, dict):
    require(gate_replay_coverage.get("passed") is True,
            "candidate measured qualified-M MC2 replay gate failed")
    require(gate_replay_coverage.get("errors") == [],
            "candidate measured qualified-M MC2 replay gate contains errors")
    require(gate_replay_coverage.get("static_routes_consistent") is True,
            "candidate target ranks do not share one MC2 static route")
    require(gate_replay_coverage.get("measured_replay_histograms_consistent") is True,
            "candidate target ranks have divergent replay histograms")
    replay_targets = gate_replay_coverage.get("target_ranks", [])
    require(len(replay_targets) == 3,
            "candidate replay coverage does not report three target ranks")
    require(sorted(int(item.get("rank", -1)) for item in replay_targets) == [1, 2, 3],
            "candidate replay coverage target-rank set is not [1,2,3]")
    for item in replay_targets:
        rank = item.get("rank")
        require(item.get("passed") is True,
                f"candidate target rank {rank} qualified-M replay gate failed")
        require(int(item.get("qualified_mc2_replay_count", 0)) > 0,
                f"candidate target rank {rank} measured no qualified MC2 replay")
        require(item.get("static_projection_routes_match_profile") is True,
                f"candidate target rank {rank} projection-shape MC2 routes differ from profile")
        require(int(item.get("measured_mc2_fallback", -1)) == 0,
                f"candidate target rank {rank} measured MC2 fallback")
        require(int(item.get("measured_mc2_exception", -1)) == 0,
                f"candidate target rank {rank} measured MC2 exception")
else:
    errors.append("candidate gate lacks measured qualified-M MC2 replay evidence")

# Bind child artifacts to the source snapshot and qualified MC2 profile.
expected_head = source_identity["git_head"]
expected_diff_sha = source_identity["git_diff_sha256"]
expected_status_sha = source_identity["git_status_sha256"]
for label, run_dir, head_name, patch_name, status_name in (
    ("baseline", baseline_dir, "git-head-before.txt", "working-tree-before.patch", "git-status-before.txt"),
    ("candidate", candidate_dir, "git-head.txt", "working-tree.patch", "git-status.txt"),
):
    try:
        child_head = (run_dir / head_name).read_text().strip()
        child_diff_sha = sha256(run_dir / patch_name)
        child_status_sha = sha256(run_dir / status_name)
    except OSError as error:
        errors.append(f"cannot read {label} source identity: {error}")
        continue
    require(child_head == expected_head, f"{label} git HEAD differs from campaign source")
    require(child_diff_sha == expected_diff_sha,
            f"{label} working-tree patch differs from campaign source")
    require(child_status_sha == expected_status_sha,
            f"{label} git status differs from campaign source")

candidate_contract = {}
try:
    for line in (candidate_dir / "contract.env").read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator:
            candidate_contract[key] = value
except OSError as error:
    errors.append(f"cannot read candidate contract.env: {error}")
require(candidate_contract.get("visible_devices") == ",".join(map(str, devices)),
        "candidate physical NPU mapping mismatch")
require(str(Path(candidate_contract.get("profile", "")).resolve()) == profile_path,
        "candidate contract profile path mismatch")
require(str(Path(candidate_contract.get("vendor", "")).resolve()) == vendor_path,
        "candidate contract vendor path mismatch")
require(
    candidate_contract.get("draft_pa_task_update_workers")
    == str(expected_draft_pa_task_update_workers),
    "candidate contract draft-PA task-update worker mismatch",
)
require(candidate_contract.get("target_replay_ahead") == str(expected_target_replay_ahead),
        "candidate contract target replay-ahead mismatch")
candidate_inputs = parse_sha256sum(candidate_dir / "input.sha256")
require(candidate_inputs.get(manifest_path) == manifest_sha,
        "candidate input manifest identity mismatch")
require(candidate_inputs.get(profile_path) == profile_sha,
        "candidate input MC2 profile identity mismatch")
require(candidate_inputs.get(extension_path) == extension_sha,
        "candidate input extension identity mismatch")

baseline_raw = finite_number(
    baseline_row.get("output_throughput_tokens_per_second"), "baseline Raw"
)
candidate_raw = finite_number(
    candidate_row.get("e2e_throughput_tokens_per_second"), "candidate Raw"
)
baseline_goodput = finite_number(
    baseline_row.get("slo", {}).get("paper", {}).get("goodput_tokens_per_e2e_second"),
    "baseline paper Goodput",
)
candidate_goodput = finite_number(
    candidate_row.get("slo", {}).get("paper", {}).get("goodput_tokens_per_e2e_second"),
    "candidate paper Goodput",
)
require(
    math.isclose(
        finite_number(candidate_gate.get("raw_tokens_per_second"), "candidate gate Raw"),
        candidate_raw,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ),
    "candidate launcher gate Raw differs from result",
)
require(
    math.isclose(
        finite_number(
            candidate_gate.get("paper_goodput_tokens_per_second"),
            "candidate gate paper Goodput",
        ),
        candidate_goodput,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ),
    "candidate launcher gate paper Goodput differs from result",
)
require(baseline_raw > 0.0, "baseline Raw must be positive")
require(baseline_goodput > 0.0, "baseline paper Goodput must be positive")
raw_speedup = (
    candidate_raw / baseline_raw
    if math.isfinite(candidate_raw) and math.isfinite(baseline_raw) and baseline_raw > 0.0
    else None
)
goodput_speedup = (
    candidate_goodput / baseline_goodput
    if math.isfinite(candidate_goodput)
    and math.isfinite(baseline_goodput)
    and baseline_goodput > 0.0
    else None
)
require(candidate_raw > baseline_raw,
        f"round {round_number} Raw gate failed: {candidate_raw} <= {baseline_raw}")
require(candidate_goodput >= 1.3 * baseline_goodput,
        f"round {round_number} Goodput gate failed: {candidate_goodput} < 1.3*{baseline_goodput}")

document = {
    "status": "pass" if not errors else "fail",
    "round": round_number,
    "physical_npus": devices,
    "baseline_result": str((baseline_dir / "result.json").resolve()),
    "candidate_result": str((candidate_dir / "result.json").resolve()),
    "baseline_raw_tokens_per_second": baseline_raw if math.isfinite(baseline_raw) else None,
    "candidate_raw_tokens_per_second": candidate_raw if math.isfinite(candidate_raw) else None,
    "raw_speedup": raw_speedup,
    "baseline_paper_goodput_tokens_per_second": (
        baseline_goodput if math.isfinite(baseline_goodput) else None
    ),
    "candidate_paper_goodput_tokens_per_second": (
        candidate_goodput if math.isfinite(candidate_goodput) else None
    ),
    "paper_goodput_speedup": goodput_speedup,
    "required_raw_relation": "candidate > paired baseline",
    "required_paper_goodput_speedup": 1.3,
    "manifest_sha256": manifest_sha,
    "profile_sha256": profile_sha,
    "vendor_sha256": vendor_sha,
    "source_identity_sha256": source_identity["identity_sha256"],
    "mc2_measured_replay_coverage": gate_replay_coverage,
    "draft_pa_task_update_workers": expected_draft_pa_task_update_workers,
    "draft_pa_task_update": draft_pa_evidence,
    "target_replay_ahead": target_replay_evidence,
    "target_replay_ahead_enabled": expected_target_replay_ahead,
    "errors": errors,
}
output_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
if errors:
    raise SystemExit("; ".join(errors))
PY
}

write_final_gate() {
  "$python" - "$campaign_dir" <<'PY'
import json
import statistics
import sys
from pathlib import Path

campaign_dir = Path(sys.argv[1])
rounds = [
    json.loads((campaign_dir / f"round-{round_number}-gate.json").read_text())
    for round_number in (1, 2, 3)
]
errors = []
for row in rounds:
    if row.get("status") != "pass":
        errors.append(f"round {row.get('round')} failed")
source_identities = {row.get("source_identity_sha256") for row in rounds}
manifest_hashes = {row.get("manifest_sha256") for row in rounds}
profile_hashes = {row.get("profile_sha256") for row in rounds}
vendor_hashes = {row.get("vendor_sha256") for row in rounds}
physical_mappings = {tuple(row.get("physical_npus", [])) for row in rounds}
draft_pa_worker_counts = {row.get("draft_pa_task_update_workers") for row in rounds}
target_replay_modes = {row.get("target_replay_ahead_enabled") for row in rounds}
if len(source_identities) != 1:
    errors.append("source identity differs across rounds")
if len(manifest_hashes) != 1:
    errors.append("manifest identity differs across rounds")
if len(profile_hashes) != 1:
    errors.append("MC2 profile identity differs across rounds")
if len(vendor_hashes) != 1:
    errors.append("MC2 vendor identity differs across rounds")
if len(physical_mappings) != 1:
    errors.append("physical NPU mapping differs across rounds")
if len(draft_pa_worker_counts) != 1 or not draft_pa_worker_counts.issubset({1, 2, 4}):
    errors.append("draft-PA task-update worker contract differs across rounds")
if len(target_replay_modes) != 1 or not target_replay_modes.issubset({0, 1}):
    errors.append("target replay-ahead contract differs across rounds")
raw_speedups = [float(row["raw_speedup"]) for row in rounds]
goodput_speedups = [float(row["paper_goodput_speedup"]) for row in rounds]
all_rounds_raw_above_baseline = all(value > 1.0 for value in raw_speedups)
all_rounds_paper_goodput_at_least_1_3x = all(
    value >= 1.3 for value in goodput_speedups
)
if not all_rounds_raw_above_baseline:
    errors.append(
        "not every round has Raw(candidate) > Raw(paired baseline)"
    )
if not all_rounds_paper_goodput_at_least_1_3x:
    errors.append(
        "not every round has paper Goodput(candidate) >= 1.3x paired baseline"
    )
document = {
    "status": "pass" if not errors else "fail",
    "round_order": ["baseline,candidate", "candidate,baseline", "baseline,candidate"],
    "rounds": rounds,
    "draft_pa_task_update_workers": (
        next(iter(draft_pa_worker_counts)) if len(draft_pa_worker_counts) == 1 else None
    ),
    "target_replay_ahead_enabled": (
        next(iter(target_replay_modes)) if len(target_replay_modes) == 1 else None
    ),
    "raw_speedup_min": min(raw_speedups),
    "raw_speedup_median": statistics.median(raw_speedups),
    "paper_goodput_speedup_min": min(goodput_speedups),
    "paper_goodput_speedup_median": statistics.median(goodput_speedups),
    "all_rounds_raw_above_baseline": all_rounds_raw_above_baseline,
    "all_rounds_paper_goodput_at_least_1_3x": (
        all_rounds_paper_goodput_at_least_1_3x
    ),
    "errors": errors,
}
output = campaign_dir / "formal-ab-gate.json"
output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
if errors:
    raise SystemExit("; ".join(errors))
print(f"formal A/B gate: {output}")
PY
}

if [[ $mode == validate-only ]]; then
  [[ -d $campaign_dir ]] || {
    echo "campaign directory is missing: $campaign_dir" >&2
    exit 2
  }
  for round in 1 2 3; do
    validate_round "$round"
  done
  assert_source_unchanged validate-only
  assert_inputs_unchanged
  write_final_gate
  exit 0
fi

if [[ -e $campaign_dir ]]; then
  echo "refusing to overwrite existing campaign: $campaign_dir" >&2
  exit 3
fi
for round in 1 2 3; do
  for run_dir in "$(baseline_run_dir "$round")" "$(candidate_run_dir "$round")"; do
    if [[ -e $run_dir ]]; then
      echo "refusing to reuse an existing child run: $run_dir" >&2
      exit 3
    fi
  done
done
mkdir -p "$campaign_dir"
print_plan > "$campaign_dir/plan.txt"
snapshot_source "$campaign_dir/source-initial.json"
write_input_identity

export SPECSLO_RESULT_ROOT=$result_root
export SPECSLO_VISIBLE_DEVICES=$visible_devices
export SPECSLO_MC2_PROFILE=$profile
export SPECSLO_MC2_VENDOR=$vendor
# Freeze the experiment knob for every candidate child.  The child launcher
# separately overwrites the VLLM runtime variable, so an inherited runtime
# setting cannot contaminate the formal campaign.
export SPECSLO_DRAFT_PA_TASK_UPDATE_WORKERS=$draft_pa_task_update_workers
export SPECSLO_TARGET_REPLAY_AHEAD=$target_replay_ahead

run_baseline() {
  local round=$1
  assert_source_unchanged "r${round}-before-baseline"
  assert_inputs_unchanged
  "$baseline_script" "$(round_tag "$round")"
  assert_source_unchanged "r${round}-after-baseline"
  assert_inputs_unchanged
}

run_candidate() {
  local round=$1
  assert_source_unchanged "r${round}-before-candidate"
  assert_inputs_unchanged
  # Formal throughput/Goodput rounds are intentionally profiler-free even if
  # the caller's shell previously ran an overlap diagnostic.
  SPECSLO_PROFILE_HOST_DECODE_STEPS=0 \
    "$candidate_script" on "$(round_tag "$round")"
  assert_source_unchanged "r${round}-after-candidate"
  assert_inputs_unchanged
}

run_baseline 1
run_candidate 1
validate_round 1

run_candidate 2
run_baseline 2
validate_round 2

run_baseline 3
run_candidate 3
validate_round 3

assert_source_unchanged final
assert_inputs_unchanged
write_final_gate
