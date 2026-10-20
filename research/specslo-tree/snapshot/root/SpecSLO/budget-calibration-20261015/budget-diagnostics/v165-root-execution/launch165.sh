#!/usr/bin/env bash
# Separate conditional LOCAL diagnostic entry; all online launchers stay blocked.
set -euo pipefail
task_root=/root/SpecSLO/budget-calibration-20261015/tree-live-shared4-union-full-v165
if [[ $# -ne 4 || $1 != --action || $3 != --tag ||
      $2 != live-shared4-union-full463 ||
      ! $4 =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]]; then
  echo 'usage: launch_local_qualification.sh --action <live-shared4-union-full463> --tag <tag>' >&2
  exit 2
fi
# An inherited permit/env variable cannot authorize a model launch. The inner
# Python checker revalidates actual local CPU, exclusive freeze and priors.
exec /usr/bin/env -i HOME=/root USER=root LOGNAME=root SHELL=/bin/bash LANG=C.UTF-8 \
  PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin \
  /bin/bash /root/SpecSLO/budget-calibration-20261015/serial-v1/runtime.sh \
  /bin/bash "/root/SpecSLO/budget-calibration-20261015/budget-diagnostics/v165-root-execution/startup165.sh" "$@"
