#!/usr/bin/env bash
# Separate conditional LOCAL diagnostic entry; all online launchers stay blocked.
set -euo pipefail
task_root=/root/SpecSLO/budget-calibration-20261015/tree-causal80-small-cohort-role-aware-audit-full-v209
if [[ $# -ne 4 || $1 != --action || $3 != --tag ||
      $2 != alternate-chain-B1to8-M160-N32-node-component ||
      ! $4 =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo 'usage: launch_local_qualification.sh --action <alternate-chain-B1to8-M160-N32-node-component> --tag <tag>' >&2
  exit 2
fi
# An inherited permit/env variable cannot authorize a model launch. The inner
# Python checker revalidates actual local CPU, exclusive freeze and priors.
exec /usr/bin/env -i HOME=/root USER=root LOGNAME=root SHELL=/bin/bash LANG=C.UTF-8 \
  PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin \
  SPECSLO_V326_ROOT_FREEZE_SHA="${SPECSLO_V326_ROOT_FREEZE_SHA:?explicit ROOT326 freeze required}" \
  SPECSLO_V317_ROOT_FREEZE_SHA=36cbac2bca38b3b69b2ab4ed04f984bb48782fd51d5bb682d04f3d98ba76405f \
  SPECSLO_V218_ROOT_FREEZE_SHA=fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260 \
  /bin/bash /root/SpecSLO/budget-calibration-20261015/serial-v1/runtime.sh \
  /bin/bash "/root/SpecSLO/budget-calibration-20261015/tree-M160-S32-N32-target-math-cost-v326/startup326.sh" "$@"
