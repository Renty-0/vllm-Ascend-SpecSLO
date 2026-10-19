"""Validate actual shell/guard entrypoints without executing a model."""
import json
import subprocess
from pathlib import Path

import guard333


def main():
    here = Path(__file__).resolve().parent
    launch = (here / 'launch333.sh').read_text()
    startup = (here / 'startup333.sh').read_text()
    for name in ('launch333.sh', 'startup333.sh'):
        subprocess.run(['bash', '-n', str(here / name)], check=True, capture_output=True)
    assert str(here / 'startup333.sh') in launch
    assert str(here / 'runtime333.py') in startup
    assert str(here / 'check333.py') in startup
    assert 'TRITON_CACHE_DIR=/root/SpecSLO/budget-calibration-20261015/tree-M160-S32-N32-target-math-cost-v326/triton-cache' in startup
    assert guard333.MAX_AGGREGATE_BYTES==2560*1024**2
    assert 'tree-dynamic-root-alternate-paired-prefix-grid-full-v333' not in launch + startup
    search = next(line.split('=', 1)[1] for line in startup.splitlines()
                  if line.startswith('export PYTHONPATH='))
    for path in search.split(':'):
        if path.startswith('/root/'):
            assert Path(path).is_dir(), path
    guard333.EXPECTED_FREEZE = 'f' * 64
    argv = guard333.launch_argv('HOST333-path-check')
    assert argv[-5] == str(here / 'launch333.sh')
    assert argv[-4:] == ['--action', guard333.ACTION, '--tag', 'HOST333-path-check']
    value = dict(status='HOST_actual_guard_shell_runtime_and_import_paths_passed',
                 actual_candidate_directory=str(here), old_alias_not_required=True,
                 all_literal_import_directories_exist=True, NPU_used=False,
                 model_launch_executed=False)
    target = here / 'HOST_LAUNCH333_RESULT.json'
    with target.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
