#!/usr/bin/env bash
set -euo pipefail
if [[ $# != 1 || ! $1 =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo 'usage: bash run.sh FRESH_TAG' >&2
  exit 2
fi
exec bash /root/SpecSLO/official-v021-baseline-20261019/env.sh bash /root/SpecSLO/official-v021-baseline-20261019/run-inner.sh "$1"
