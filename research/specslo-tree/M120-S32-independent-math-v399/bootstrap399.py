"""Use the reviewed398 dependency paths; no environment or model installation."""
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
B = P.parent
FROZEN = '/root/data/specslo-baselines/nano-pearl-ascend-v1'
STARTUP = B/'tree-active64-whole-proposal-segment-limit-v398/startup398.sh'


def install():
    value = next(line.removeprefix('export PYTHONPATH=') for line in STARTUP.read_text().splitlines()
                 if line.startswith('export PYTHONPATH='))
    for name, replacement in (('$task_root', str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209')),
                              ('$repo', FROZEN+'/source/vllm-ascend-hust'),
                              ('$frozen', FROZEN), ('${PYTHONPATH:-}', '')):
        value = value.replace(name, replacement)
    if '$' in value:
        raise RuntimeError('ROOT399 explicit reviewed startup paths only')
    sys.path[:0] = [str(P), str(B/'tree-M120-S32-frugal-budget-projection-host-v397'),
                    *filter(None, value.split(':'))]
