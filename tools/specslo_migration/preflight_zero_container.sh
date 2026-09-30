#!/usr/bin/env bash
set -uo pipefail

# Read-only preflight for rebuilding the frozen SpecSLO experiment in a new
# container.  Run it after source bootstrap and portable-evidence restore.
# It never reserves an NPU and never starts an inference process.

usage() {
  cat <<'EOF'
usage: preflight_zero_container.sh [--full-model-hash]

Environment overrides:
  SPECSLO_REPO              SpecSLO source root
  SPECSLO_VLLM_ROOT         patched vLLM-HUST source root
  SPECSLO_PYTHON            vllm-hust-dev Python executable
  SPECSLO_CANN_ROOT         CANN toolkit root
  SPECSLO_WORKLOAD          restored fixed workload
  SPECSLO_CANDIDATE_RESULT  restored candidate result.json

--full-model-hash reads both Qwen3 model trees (about 67 GB) and checks every
file against CORE_MODEL_SHA256SUMS.  Without it, model presence is checked and
the omitted bit-exact verification is reported as a warning.
EOF
}

full_model_hash=0
if [[ ${1:-} == --full-model-hash ]]; then
  full_model_hash=1
  shift
fi
if [[ $# -ne 0 ]]; then
  usage >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=${SPECSLO_REPO:-$(cd -- "$script_dir/../.." && pwd)}
vllm_root=${SPECSLO_VLLM_ROOT:-/root/data/vllm-hust}
python=${SPECSLO_PYTHON:-/root/miniconda3/envs/vllm-hust-dev/bin/python}
cann_root=${SPECSLO_CANN_ROOT:-/usr/local/Ascend/cann-9.0.0}
docs=$repo/docs/migration/20260930
artifacts=$repo/migration_artifacts/20260930
workload=${SPECSLO_WORKLOAD:-/root/data/specslo-workloads/eurosys27-rps4-seed0-author-slo40-50-150/rps4.0_mix0.6_0.2_0.2_seed0.jsonl}
candidate_result=${SPECSLO_CANDIDATE_RESULT:-/root/data/nano-pearl-benchmark-results/20260929-specslo-v149-drain-padding-repair-precheck/runs/candidate-on-v149-drain-padding-repair-target457-r1/result.json}

runtime_checkpoint=31b3cde3296c00867e8d9e23558598debed276a6
vllm_base=933842ab0fdd53ee47dee8148bca1671d7d6da61
catlass_commit=41bf90da655bba3c66d0acd7e00abe33960ecfd6
patch_sha=133d96d1645225a6c9c40db445bf660a1f576787bd59265e5d0cc239f0c6db63
workload_sha=dc58dc5af1c5b28c6944f8f3faadeaae2dae34298c07b3ac84745d367c67d662
candidate_sha=943cba8031561a23aef795541d1f07aebb9769886817871596317a6b4289095b
baseline_sha=2b957c1e59b1443e9d9cddd0bc2481a248d94db24808cbe06d590eb1bee198cd

passes=0
warnings=0
failures=0

pass() {
  printf 'PASS  %s\n' "$*"
  passes=$((passes + 1))
}

warn() {
  printf 'WARN  %s\n' "$*"
  warnings=$((warnings + 1))
}

fail() {
  printf 'FAIL  %s\n' "$*"
  failures=$((failures + 1))
}

check_sha() {
  local expected=$1
  local path=$2
  local label=$3
  local actual
  if [[ ! -f $path ]]; then
    fail "$label missing: $path"
    return
  fi
  actual=$(sha256sum "$path" | awk '{print $1}')
  if [[ $actual == "$expected" ]]; then
    pass "$label SHA256=$actual"
  else
    fail "$label SHA256 expected=$expected actual=$actual"
  fi
}

printf 'SpecSLO zero-container preflight\n'
printf 'repo=%s\n' "$repo"
printf 'vllm_root=%s\n' "$vllm_root"
printf 'python=%s\n\n' "$python"

if [[ $(uname -m) == aarch64 ]]; then
  pass 'architecture is aarch64'
else
  fail "architecture is $(uname -m), expected aarch64"
fi

for command in git sha256sum zstd; do
  if command -v "$command" >/dev/null 2>&1; then
    pass "command available: $command"
  else
    fail "command missing: $command"
  fi
done

if git -C "$repo" cat-file -e "$runtime_checkpoint^{commit}" 2>/dev/null &&
   git -C "$repo" merge-base --is-ancestor "$runtime_checkpoint" HEAD 2>/dev/null; then
  pass "SpecSLO runtime checkpoint is present in HEAD: $runtime_checkpoint"
else
  fail "SpecSLO HEAD does not contain runtime checkpoint $runtime_checkpoint"
fi

patch=$repo/patches/vllm-hust/specslo-core.patch
check_sha "$patch_sha" "$patch" 'vLLM-HUST integration patch'

if [[ -d $vllm_root/.git ]] &&
   [[ $(git -C "$vllm_root" rev-parse HEAD 2>/dev/null) == "$vllm_base" ]]; then
  pass "vLLM-HUST base is $vllm_base"
else
  fail "vLLM-HUST base is missing or not $vllm_base"
fi

if [[ -d $vllm_root/.git ]] &&
   git -C "$vllm_root" apply --reverse --check "$patch" >/dev/null 2>&1; then
  pass 'complete SpecSLO vLLM-HUST patch is applied'
else
  fail 'complete SpecSLO vLLM-HUST patch is not reverse-applicable'
fi

catlass=$repo/csrc/third_party/catlass
if [[ -e $catlass/.git ]] &&
   [[ $(git -C "$catlass" rev-parse HEAD 2>/dev/null) == "$catlass_commit" ]]; then
  pass "Catlass commit is $catlass_commit"
else
  fail "Catlass is missing or not at $catlass_commit"
fi

if (cd "$artifacts" && sha256sum -c SHA256SUMS >/dev/null); then
  pass 'all Git-stored portable artifacts match SHA256SUMS'
else
  fail 'portable artifact SHA256 verification failed'
fi

if (cd "$docs" && sha256sum -c MANIFEST.sha256 >/dev/null); then
  pass 'migration snapshot files match MANIFEST.sha256'
else
  fail 'migration snapshot MANIFEST.sha256 verification failed'
fi

check_sha "$workload_sha" "$workload" 'fixed online workload'
check_sha "$candidate_sha" "$candidate_result" 'best candidate result'

baseline_archive=$artifacts/specslo-tp4-baseline-result-20260928.json.zst
if [[ -f $baseline_archive ]]; then
  actual=$(zstd -dc "$baseline_archive" | sha256sum | awk '{print $1}')
  if [[ $actual == "$baseline_sha" ]]; then
    pass "full TP4 baseline result decompressed SHA256=$actual"
  else
    fail "full TP4 baseline result expected=$baseline_sha actual=$actual"
  fi
else
  fail "full TP4 baseline result archive missing: $baseline_archive"
fi

toolkit_info=$cann_root/aarch64-linux/ascend_toolkit_install.info
if [[ -f $toolkit_info ]] &&
   grep -qx 'version=9.0.0' "$toolkit_info" &&
   grep -qx 'innerversion=V100R001C10SPC001B250' "$toolkit_info"; then
  pass 'CANN toolkit is 9.0.0 / V100R001C10SPC001B250'
else
  fail "CANN toolkit identity mismatch or missing: $toolkit_info"
fi

driver_info=/usr/local/Ascend/driver/version.info
if [[ -f $driver_info ]] &&
   grep -qx 'Version=26.0.rc1' "$driver_info" &&
   grep -qx 'ascendhal_version=7.35.23' "$driver_info"; then
  pass 'Ascend driver is 26.0.rc1 / ascendhal 7.35.23'
else
  fail "Ascend driver identity mismatch or missing: $driver_info"
fi

if [[ -x $python ]]; then
  if "$python" - <<'PY'
from importlib.metadata import version

expected = {
    "torch": "2.10.0",
    "torch-npu": "2.10.0",
    "triton": "3.5.0",
    "triton-ascend": "3.2.1",
    "transformers": "5.14.1",
    "modelscope": "1.40.1",
}
errors = []
for package, wanted in expected.items():
    try:
        actual = version(package)
    except Exception as error:
        errors.append(f"{package}: missing ({error})")
        continue
    print(f"      {package}={actual}")
    if actual != wanted:
        errors.append(f"{package}: expected {wanted}, got {actual}")
if errors:
    raise SystemExit("; ".join(errors))
PY
  then
    pass 'critical Python package versions match the captured environment'
  else
    fail 'critical Python package versions do not match the captured environment'
  fi
else
  fail "Python executable missing: $python"
fi

model_root=/data/shared-models
model_missing=0
for model in Qwen3-0.6B Qwen3-32B; do
  if [[ -f $model_root/$model/config.json ]] &&
     ! find "$model_root/$model" -name '*.incomplete' -print -quit | grep -q .; then
    pass "$model is present and has no .incomplete files"
  else
    fail "$model is missing, incomplete, or lacks config.json"
    model_missing=1
  fi
done

if [[ $full_model_hash -eq 1 && $model_missing -eq 0 ]]; then
  if (cd / && sha256sum -c "$docs/CORE_MODEL_SHA256SUMS" >/dev/null); then
    pass 'both Qwen3 model trees are bit-exact'
  else
    fail 'Qwen3 full model SHA256 verification failed'
  fi
elif [[ $full_model_hash -eq 0 ]]; then
  warn 'full model hashes were not read; rerun with --full-model-hash before claiming bit-exact reproduction'
fi

if command -v npu-smi >/dev/null 2>&1; then
  npu_count=$(npu-smi info -l 2>/dev/null | awk '/Total Count/ {print $NF; exit}')
  if [[ $npu_count =~ ^[0-9]+$ ]] && (( npu_count >= 4 )); then
    pass "npu-smi reports $npu_count NPUs"
  else
    fail "fewer than four visible NPUs (reported: ${npu_count:-unknown})"
  fi
  warn 'NPU process occupancy is not auto-approved; inspect npu-smi immediately before each benchmark and do not use an occupancy script'
else
  fail 'npu-smi is unavailable'
fi

if [[ -f $repo/vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so ]]; then
  pass 'vllm_ascend extension exists (ABI still requires runtime tests)'
else
  warn 'vllm_ascend extension is not built yet; rebuild before NPU tests'
fi

printf '\nSummary: PASS=%d WARN=%d FAIL=%d\n' "$passes" "$warnings" "$failures"
if (( failures > 0 )); then
  printf 'Result: NOT READY. Resolve every FAIL before performance testing.\n'
  exit 1
fi
printf 'Result: preflight passed. Warnings remain explicit acceptance items.\n'
