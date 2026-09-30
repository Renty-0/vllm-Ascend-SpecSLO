#!/usr/bin/env bash
set -euo pipefail

# Run this after cloning the private reproduction tag.  It reconstructs the
# second source tree from GitHub and applies the exact vLLM-HUST core patch.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SPECSLO_ROOT=${SPECSLO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}
SPECSLO_DATA_ROOT=${SPECSLO_DATA_ROOT:-/root/data}
VLLM_HUST_ROOT=${VLLM_HUST_ROOT:-$SPECSLO_DATA_ROOT/vllm-hust}
VLLM_HUST_REMOTE=${VLLM_HUST_REMOTE:-git@github.com:vLLM-HUST/vllm-hust.git}

RUNTIME_CHECKPOINT=31b3cde3296c00867e8d9e23558598debed276a6
VLLM_HUST_BASE=933842ab0fdd53ee47dee8148bca1671d7d6da61
CATLASS_COMMIT=41bf90da655bba3c66d0acd7e00abe33960ecfd6
PATCH_PATH=$SPECSLO_ROOT/patches/vllm-hust/specslo-core.patch
PATCH_SHA256=133d96d1645225a6c9c40db445bf660a1f576787bd59265e5d0cc239f0c6db63

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

command -v git >/dev/null || fail "git is not installed"
command -v sha256sum >/dev/null || fail "sha256sum is not installed"
git -C "$SPECSLO_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1 || \
  fail "$SPECSLO_ROOT is not a Git worktree"
git -C "$SPECSLO_ROOT" cat-file -e "$RUNTIME_CHECKPOINT^{commit}" || \
  fail "private clone does not contain runtime checkpoint $RUNTIME_CHECKPOINT"
git -C "$SPECSLO_ROOT" merge-base --is-ancestor "$RUNTIME_CHECKPOINT" HEAD || \
  fail "runtime checkpoint is not an ancestor of the checked-out branch"
printf '%s  %s\n' "$PATCH_SHA256" "$PATCH_PATH" | sha256sum -c -

mkdir -p "$SPECSLO_DATA_ROOT"
if [[ ! -d "$VLLM_HUST_ROOT/.git" ]]; then
  if [[ -e "$VLLM_HUST_ROOT" ]] && \
      [[ -n "$(find "$VLLM_HUST_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    fail "$VLLM_HUST_ROOT exists and is not an empty Git repository"
  fi
  mkdir -p "$VLLM_HUST_ROOT"
  git -C "$VLLM_HUST_ROOT" init
  git -C "$VLLM_HUST_ROOT" remote add origin "$VLLM_HUST_REMOTE"
fi

if ! git -C "$VLLM_HUST_ROOT" remote get-url origin >/dev/null 2>&1; then
  git -C "$VLLM_HUST_ROOT" remote add origin "$VLLM_HUST_REMOTE"
fi
git -C "$VLLM_HUST_ROOT" fetch --depth=1 origin "$VLLM_HUST_BASE"

if git -C "$VLLM_HUST_ROOT" apply --reverse --check "$PATCH_PATH" >/dev/null 2>&1; then
  test "$(git -C "$VLLM_HUST_ROOT" rev-parse HEAD)" = "$VLLM_HUST_BASE" || \
    fail "patch is applied, but vLLM-HUST HEAD is not the expected base"
  printf 'vLLM-HUST patch is already applied; leaving the worktree unchanged.\n'
else
  if [[ -n "$(git -C "$VLLM_HUST_ROOT" status --short)" ]]; then
    fail "vLLM-HUST has unrelated changes before patch application"
  fi
  git -C "$VLLM_HUST_ROOT" checkout --detach FETCH_HEAD
  test "$(git -C "$VLLM_HUST_ROOT" rev-parse HEAD)" = "$VLLM_HUST_BASE" || \
    fail "failed to checkout exact vLLM-HUST base"
  git -C "$VLLM_HUST_ROOT" apply --check "$PATCH_PATH"
  git -C "$VLLM_HUST_ROOT" apply --3way "$PATCH_PATH"
fi

test "$(git -C "$VLLM_HUST_ROOT" rev-parse HEAD)" = "$VLLM_HUST_BASE" || \
  fail "unexpected vLLM-HUST HEAD after patch application"
git -C "$VLLM_HUST_ROOT" apply --reverse --check "$PATCH_PATH" || \
  fail "the checked-out vLLM-HUST tree does not contain the complete SpecSLO patch"

git -C "$SPECSLO_ROOT" submodule update --init --recursive
test "$(git -C "$SPECSLO_ROOT/csrc/third_party/catlass" rev-parse HEAD)" = \
  "$CATLASS_COMMIT" || fail "Catlass submodule commit mismatch"

printf '\nSource bootstrap passed.\n'
printf '  SpecSLO tree: %s\n' "$SPECSLO_ROOT"
printf '  SpecSLO HEAD: %s\n' "$(git -C "$SPECSLO_ROOT" rev-parse HEAD)"
printf '  vLLM-HUST:    %s\n' "$VLLM_HUST_ROOT"
printf '  vLLM base:    %s + specslo-core.patch\n' "$VLLM_HUST_BASE"
printf '  Catlass:      %s\n' "$CATLASS_COMMIT"
