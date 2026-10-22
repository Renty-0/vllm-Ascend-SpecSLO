"""Run CPU checks after the exact retained runtime clears its environment."""
import json
import os
from pathlib import Path
import subprocess
import sys

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B / 'tree-nonempty-refill-accounting-witness-host-v429'
F = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')
REPO = F / 'source/vllm-ascend-hust'
PYTHON = '/root/miniconda3/envs/vllm-hust-dev/bin/python'
SCRIPTS = ('check429_HOST.py',)
OPTIONAL = ()


def inside():
    target = sys.argv[2]
    assert target in (*SCRIPTS, *OPTIONAL)
    lines = [line for line in (B / 'tree-parent4-canonical-tail-observer-binding-v425/startup425.sh').read_text().splitlines()
             if line.startswith('export PYTHONPATH=')]
    assert len(lines) == 1
    value = lines[0].removeprefix('export PYTHONPATH=')
    for before, after in [('$repo', str(REPO)), ('$frozen', str(F)),
                          ('$task_root', str(B / 'tree-causal80-small-cohort-role-aware-audit-full-v209')),
                          ('${PYTHONPATH:-}', '')]:
        value = value.replace(before, after)
    assert '$' not in value
    import shlex
    for line in (B / 'tree-parent4-canonical-tail-observer-binding-v425/startup425.sh').read_text().splitlines():
        if line.startswith('export '):
            for assignment in shlex.split(line.removeprefix('export ')):
                if '=' not in assignment:
                    continue
                key, setting = assignment.split('=', 1)
                if key.startswith('VLLM_ASCEND_') or key in ('TASK_QUEUE_ENABLE', 'HCCL_OP_EXPANSION_MODE', 'HCCL_DETERMINISTIC', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
                    assert '$' not in setting and '`' not in setting
                    os.environ[key] = setting
        elif line.startswith('unset '):
            for key in shlex.split(line.removeprefix('unset ')):
                if key.startswith('VLLM_ASCEND_'):
                    os.environ.pop(key, None)
    os.environ['PYTHONPATH'] = value
    sys.path[:0] = [str(P),*value.split(':')]
    import torch
    import torch_npu
    torch.set_num_threads(1)
    def blocked(*args, **kwargs):
        raise RuntimeError('HOST429 attempted to initialize an NPU')
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
    environment.update({'SPECSLO_V386_ROOT_FREEZE_SHA': '0909f2ed2a3c5fbd0ef94a3e9122611c48bf4cc4e164afdbd558ce27afbb1b96', 'SPECSLO_V385_ROOT_FREEZE_SHA': 'e62fc5f7740fa493906a7889a8af1baf2fd118696e8a271b5b5c493a7f892a1f', 'SPECSLO_V384_ROOT_FREEZE_SHA': 'a647343b81fac81459d6829644a59ab12425955647b11a78c15a4978871a1e5b', 'SPECSLO_V382_ROOT_FREEZE_SHA': 'f8b65aa81b78b9f33d3e0e97e30f62841f310160e52c321fd0b8a0243023d6c4', 'SPECSLO_V381_ROOT_FREEZE_SHA': '9ae5f9c8a807be352c0afd59dcce84a2b67eaa3ac48c644ffa9cb19f6039ebe5', 'SPECSLO_V380_ROOT_FREEZE_SHA': 'a1a52c1d2b34e5ded30ad1a2a5e52baea39369ffdf26e6f6ffc0dd4e2920934e', 'SPECSLO_V379_ROOT_FREEZE_SHA': '318f60b9d142a4c67c6c6f50e0b91e5f38aec40b365a0d4dea2a32bc1e28449c', 'SPECSLO_V378_ROOT_FREEZE_SHA': 'e7beb04cadd68479440ec26a1e8b3a39949d395e7f6485af658bc219c5853bc3', 'SPECSLO_V361_ROOT_FREEZE_SHA': 'a1dbe700b6fa857631be31878df58fd7b129de38491ae027a8823799ce81bf3f', 'SPECSLO_V344_ROOT_FREEZE_SHA': 'f4aa9b02429020c9a1bdbaaa4d168253588b681897dce1796713aac72196b6c4', 'SPECSLO_V218_ROOT_FREEZE_SHA': 'fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260', 'SPECSLO_V391_ROOT_FREEZE_SHA': 'a9a4bfaa81f9bc8f95a75a533612fb1bd8b8202638a6f65c6d35b8185fe8fa98'})
    results = []
    selected = tuple(sys.argv[1:]) or SCRIPTS
    assert all(target in (*SCRIPTS, *OPTIONAL) for target in selected)
    for target in selected:
        args = ['/bin/bash', str(B / 'serial-v1/runtime.sh'), PYTHON, '-B',
                str(REPO / 'examples/run_with_vllm_ascend_extension.py'), '--extension',
                str(REPO / 'vllm_ascend/vllm_ascend_C.cpython-311-aarch64-linux-gnu.so'),
                '--required-op', 'allreduce_add_rmsnorm_chained', str(Path(__file__).resolve()), '--inside', target]
        completed = subprocess.run(args, cwd=str(REPO), env=environment, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        log = P / f'HOST429-{target}.log'
        log.write_text(completed.stdout)
        results.append(dict(script=target, exit_code=completed.returncode, log=str(log)))
        print(json.dumps(results[-1]), flush=True)
        if completed.returncode:
            print(completed.stdout[-3000:], flush=True)
            raise RuntimeError('ROOT429 actual clean HOST check failed')
    if selected != SCRIPTS:
        return
    (P / 'HOST429_CLEAN_STARTUP_RUN.json').write_text(json.dumps(dict(
        status='PASS_HOST429_ACTUAL_RUNTIME_CLEAN_PACKING_AND_LIVE_BINDING',
        results=results, NPU_used=False, model_constructors=0,
        existing_runtime_and_only_existing_freeze_environment_fields=True), indent=2) + '\n')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--inside':
        inside()
    else:
        main()
