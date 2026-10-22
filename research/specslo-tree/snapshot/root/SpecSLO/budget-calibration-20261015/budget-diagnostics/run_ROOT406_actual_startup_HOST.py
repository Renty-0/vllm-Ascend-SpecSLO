"""Run CPU checks after the exact retained runtime clears its environment."""
import json
import os
from pathlib import Path
import subprocess
import sys

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B / 'tree-frugal-S32-startup-import-repair-v406'
F = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')
REPO = F / 'source/vllm-ascend-hust'
PYTHON = '/root/miniconda3/envs/vllm-hust-dev/bin/python'
SCRIPTS = ('check406_imports_HOST.py',)


def inside():
    target = sys.argv[2]
    assert target in SCRIPTS
    lines = [line for line in (P / 'startup406.sh').read_text().splitlines()
             if line.startswith('export PYTHONPATH=')]
    assert len(lines) == 1
    value = lines[0].removeprefix('export PYTHONPATH=')
    for before, after in [('$repo', str(REPO)), ('$frozen', str(F)),
                          ('$task_root', str(B / 'tree-causal80-small-cohort-role-aware-audit-full-v209')),
                          ('${PYTHONPATH:-}', '')]:
        value = value.replace(before, after)
    assert '$' not in value
    os.environ['PYTHONPATH'] = value
    sys.path[:0] = value.split(':')
    import torch
    import torch_npu
    torch.set_num_threads(1)
    def blocked(*args, **kwargs):
        raise RuntimeError('HOST406 attempted to initialize an NPU')
    torch.npu._lazy_init = blocked
    torch_npu.npu._lazy_init = blocked
    import runpy
    runpy.run_path(str(P / target), run_name='__main__')


def main():
    environment = dict(HOME='/root', USER='root', LOGNAME='root', SHELL='/bin/bash', LANG='C.UTF-8',
        PATH='/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
        PYTHONDONTWRITEBYTECODE='1',
        SPECSLO_V257_ROOT_FREEZE_SHA='b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2',
        SPECSLO_V376_ROOT_FREEZE_SHA='ae723ddfddeb6d47f04605a3842de0c3690196b0addf39fa17c5bd1ca248765c',
        SPECSLO_V218_ROOT_FREEZE_SHA='fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260')
    results = []
    for target in SCRIPTS:
        args = ['/bin/bash', str(B / 'serial-v1/runtime.sh'), PYTHON, '-B',
                str(REPO / 'examples/run_with_vllm_ascend_extension.py'), '--extension',
                str(REPO / 'vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so'),
                '--required-op', 'allreduce_add_rmsnorm_chained', str(Path(__file__).resolve()), '--inside', target]
        completed = subprocess.run(args, cwd=str(REPO), env=environment, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        log = P / f'HOST406-{target}.log'
        log.write_text(completed.stdout)
        results.append(dict(script=target, exit_code=completed.returncode, log=str(log)))
        print(json.dumps(results[-1]), flush=True)
        if completed.returncode:
            print(completed.stdout[-3000:], flush=True)
            raise RuntimeError('ROOT406 actual clean HOST check failed')
    (P / 'HOST406_CLEAN_STARTUP_RUN.json').write_text(json.dumps(dict(
        status='PASS_HOST406_ACTUAL_RUNTIME_CLEAN_PACKING_AND_LIVE_BINDING',
        results=results, NPU_used=False, model_constructors=0,
        existing_runtime_and_only_existing_freeze_environment_fields=True), indent=2) + '\n')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--inside':
        inside()
    else:
        main()
