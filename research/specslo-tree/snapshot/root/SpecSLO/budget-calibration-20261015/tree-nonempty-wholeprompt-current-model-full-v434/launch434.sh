#!/usr/bin/env bash
# Separate conditional LOCAL diagnostic entry; all online launchers stay blocked.
set -euo pipefail
task_root=/root/SpecSLO/budget-calibration-20261015/tree-causal80-small-cohort-role-aware-audit-full-v209
if [[ $# -ne 4 || $1 != --action || $3 != --tag ||
      $2 != causal80-N9-to15-allB4-empty-active32-full463 ||
      ! $4 =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo 'usage: launch_local_qualification.sh --action <causal80-N9-to15-allB4-empty-active32-full463> --tag <tag>' >&2
  exit 2
fi
# An inherited permit/env variable cannot authorize a model launch. The inner
# Python checker revalidates actual local CPU, exclusive freeze and priors.
exec /usr/bin/env -i HOME=/root USER=root LOGNAME=root SHELL=/bin/bash LANG=C.UTF-8 \
  PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin \
  SPECSLO_V257_ROOT_FREEZE_SHA="${SPECSLO_V257_ROOT_FREEZE_SHA:?explicit ROOT257 freeze required}" \
  SPECSLO_V376_ROOT_FREEZE_SHA=ae723ddfddeb6d47f04605a3842de0c3690196b0addf39fa17c5bd1ca248765c \
  SPECSLO_V218_ROOT_FREEZE_SHA=fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260 \
  SPECSLO_V386_ROOT_FREEZE_SHA=0909f2ed2a3c5fbd0ef94a3e9122611c48bf4cc4e164afdbd558ce27afbb1b96 \
  SPECSLO_V385_ROOT_FREEZE_SHA=e62fc5f7740fa493906a7889a8af1baf2fd118696e8a271b5b5c493a7f892a1f \
  SPECSLO_V384_ROOT_FREEZE_SHA=a647343b81fac81459d6829644a59ab12425955647b11a78c15a4978871a1e5b \
  SPECSLO_V382_ROOT_FREEZE_SHA=f8b65aa81b78b9f33d3e0e97e30f62841f310160e52c321fd0b8a0243023d6c4 \
  SPECSLO_V381_ROOT_FREEZE_SHA=9ae5f9c8a807be352c0afd59dcce84a2b67eaa3ac48c644ffa9cb19f6039ebe5 \
  SPECSLO_V380_ROOT_FREEZE_SHA=a1a52c1d2b34e5ded30ad1a2a5e52baea39369ffdf26e6f6ffc0dd4e2920934e \
  SPECSLO_V379_ROOT_FREEZE_SHA=318f60b9d142a4c67c6c6f50e0b91e5f38aec40b365a0d4dea2a32bc1e28449c \
  SPECSLO_V378_ROOT_FREEZE_SHA=e7beb04cadd68479440ec26a1e8b3a39949d395e7f6485af658bc219c5853bc3 \
  SPECSLO_V361_ROOT_FREEZE_SHA=a1dbe700b6fa857631be31878df58fd7b129de38491ae027a8823799ce81bf3f \
  SPECSLO_V344_ROOT_FREEZE_SHA=f4aa9b02429020c9a1bdbaaa4d168253588b681897dce1796713aac72196b6c4 \
  SPECSLO_V391_ROOT_FREEZE_SHA=a9a4bfaa81f9bc8f95a75a533612fb1bd8b8202638a6f65c6d35b8185fe8fa98 \
  /bin/bash /root/SpecSLO/budget-calibration-20261015/serial-v1/runtime.sh \
  /bin/bash "/root/SpecSLO/budget-calibration-20261015/tree-nonempty-wholeprompt-current-model-full-v434/startup434.sh" "$@"
