"""Validate actual shell/guard entrypoints without executing a model."""
import json
import subprocess
from pathlib import Path

import guard316


def main():
    here = Path(__file__).resolve().parent
    launch = (here / 'launch316.sh').read_text()
    startup = (here / 'startup316.sh').read_text()
    for name in ('launch316.sh', 'startup316.sh'):
        subprocess.run(['bash', '-n', str(here / name)], check=True, capture_output=True)
    assert str(here / 'startup316.sh') in launch
    assert str(here / 'runtime316.py') in startup
    assert str(here / 'check316.py') in startup
    assert f'TRITON_CACHE_DIR={here}/triton-cache/$tag' in startup
    assert 'tree-dynamic-root-alternate-paired-prefix-grid-full-v316' not in launch + startup
    search = next(line.split('=', 1)[1] for line in startup.splitlines()
                  if line.startswith('export PYTHONPATH='))
    for path in search.split(':'):
        if path.startswith('/root/'):
            assert Path(path).is_dir(), path
    guard316.EXPECTED_FREEZE = 'f' * 64
    argv = guard316.launch_argv('HOST316-path-check')
    assert argv[-5] == str(here / 'launch316.sh')
    assert argv[-4:] == ['--action', guard316.ACTION, '--tag', 'HOST316-path-check']
    value = dict(status='HOST_actual_guard_shell_runtime_and_import_paths_passed',
                 actual_candidate_directory=str(here), old_alias_not_required=True,
                 all_literal_import_directories_exist=True, NPU_used=False,
                 model_launch_executed=False)
    target = here / 'HOST_LAUNCH316_RESULT.json'
    with target.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
