"""Validate actual shell/guard entrypoints without executing a model."""
import json
import subprocess
from pathlib import Path

import guard311


def main():
    here = Path(__file__).resolve().parent
    launch = (here / 'launch311.sh').read_text()
    startup = (here / 'startup311.sh').read_text()
    for name in ('launch311.sh', 'startup311.sh'):
        subprocess.run(['bash', '-n', str(here / name)], check=True, capture_output=True)
    assert str(here / 'startup311.sh') in launch
    assert str(here / 'runtime311.py') in startup
    assert str(here / 'check311.py') in startup
    assert f'TRITON_CACHE_DIR={here}/triton-cache/$tag' in startup
    assert 'tree-dynamic-root-alternate-paired-prefix-grid-full-v311' not in launch + startup
    search = next(line.split('=', 1)[1] for line in startup.splitlines()
                  if line.startswith('export PYTHONPATH='))
    for path in search.split(':'):
        if path.startswith('/root/'):
            assert Path(path).is_dir(), path
    guard311.EXPECTED_FREEZE = 'f' * 64
    argv = guard311.launch_argv('HOST311-path-check')
    assert argv[-5] == str(here / 'launch311.sh')
    assert argv[-4:] == ['--action', guard311.ACTION, '--tag', 'HOST311-path-check']
    value = dict(status='HOST_actual_guard_shell_runtime_and_import_paths_passed',
                 actual_candidate_directory=str(here), old_alias_not_required=True,
                 all_literal_import_directories_exist=True, NPU_used=False,
                 model_launch_executed=False)
    target = here / 'HOST_LAUNCH311_RESULT.json'
    with target.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
