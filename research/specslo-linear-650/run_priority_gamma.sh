#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]] || [[ ! $3 =~ ^[1-8]$ ]]; then
  echo 'usage: run_priority_gamma.sh <unique-tag> <acceptance-floor> <tight-target-burst:1..8>' >&2
  exit 2
fi

# A deferred loose request has no proposal or target verification in this
# round (effective gamma 0). Selected requests retain the qualified gamma-4
# full-window graph. The bounded burst releases loose rows to avoid starvation.
export SPECSLO_TIGHT_TARGET_BURST=$3
export SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=${SPECSLO_LOOSE_ADMISSION_TIGHT_GATE:-6}

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec bash "$repo/research/specslo-linear-650/run.sh" "$1" "$2"
