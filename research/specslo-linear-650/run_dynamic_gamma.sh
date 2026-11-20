#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo 'usage: run_dynamic_gamma.sh <unique-tag> <acceptance-floor> <min-gamma:1..3> <verification-budget>' >&2
  exit 2
fi
if [[ ! $3 =~ ^[123]$ ]] || [[ ! $4 =~ ^[1-9][0-9]*$ ]]; then
  echo 'dynamic gamma requires min-gamma 1..3 and a positive verification budget' >&2
  exit 2
fi

export SPECSLO_MIN_GAMMA=$3
export SPECSLO_VERIFICATION_BUDGET=$4
export SPECSLO_LOOSE_ADMISSION_TIGHT_GATE=${SPECSLO_LOOSE_ADMISSION_TIGHT_GATE:-6}

repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
exec bash "$repo/research/specslo-linear-650/run.sh" "$1" "$2"
