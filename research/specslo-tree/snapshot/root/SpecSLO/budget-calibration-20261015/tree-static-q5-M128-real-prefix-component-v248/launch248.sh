#!/usr/bin/env bash
# Separate conditional LOCAL diagnostic entry; all online launchers stay blocked.
set -euo pipefail
task_root=/root/SpecSLO/budget-calibration-20261015/tree-causal80-small-cohort-role-aware-audit-full-v209
if [[ $# -ne 4 || $1 != --action || $3 != --tag ||
      $2 != static-q5-M128-real-prefix-component ||
      ! $4 =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo 'usage: launch_local_qualification.sh --action <static-q5-M128-real-prefix-component> --tag <tag>' >&2
  exit 2
fi
# An inherited permit/env variable cannot authorize a model launch. The inner
# Python checker revalidates actual local CPU, exclusive freeze and priors.
exec /usr/bin/env -i HOME=/root USER=root LOGNAME=root SHELL=/bin/bash LANG=C.UTF-8 \
  PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin \
  SPECSLO_V248_ROOT_FREEZE_SHA="${SPECSLO_V248_ROOT_FREEZE_SHA:?explicit ROOT248 freeze required}" \
  SPECSLO_V218_ROOT_FREEZE_SHA=fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260 \
  /bin/bash /root/SpecSLO/budget-calibration-20261015/serial-v1/runtime.sh \
  /bin/bash "/root/SpecSLO/budget-calibration-20261015/tree-static-q5-M128-real-prefix-component-v248/startup248.sh" "$@"
