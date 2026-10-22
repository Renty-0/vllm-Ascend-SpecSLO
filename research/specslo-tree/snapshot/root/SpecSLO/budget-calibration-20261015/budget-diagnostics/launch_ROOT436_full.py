"""Conditional future launch; requires completed, independently audited435.

This file does not select a winner or push Git. All436 NN proofs are fresh.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-nonempty-qualified-RPC-full-v436'
DONOR = B / 'tree-causal80-small-cohort-role-aware-audit-full-v209'
PREVIOUS = 'qualification-receive-parent4-E4-global64-root-v435-current75-full463-ownAR-v1'
TAG = 'nonempty-refill-parent4-E4-global64-root-v436-current75-plus6-full463-ownAR-v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prior-terminal-audit', type=Path, required=True)
    parser.add_argument('--prior-result', type=Path, required=True)
    parser.add_argument('--expected-prior-receipt-sha256', required=True)
    args = parser.parse_args()
    previous_path = DONOR / f'receipt-{PREVIOUS}.json'
    assert sha(previous_path) == args.expected_prior_receipt_sha256
    previous = json.loads(previous_path.read_text())
    assert previous['status'] == 'complete' and not previous['errors']
    assert previous['sources_unchanged'] and previous['owned_remaining'] == []
    assert previous['devices_after']['all_idle'] is True
    audit = json.loads(args.prior_terminal_audit.read_text())
    assert audit['schema'] == 'ROOT435.independent-terminal-compute-audit.v1'
    assert audit['status'] == 'terminal_completed_compute_audited' and not audit['errors']
    assert audit['original_receipt_sha256'] == args.expected_prior_receipt_sha256
    assert audit['whole_workload_AR_qualified'] is True
    assert audit['current75_raw_case_records'] == [75] * 4
    result = json.loads(args.prior_result.read_text())
    assert result['status'] == 'complete_independently_audited_single_run'
    assert Path(result['source_terminal_audit']).resolve() == args.prior_terminal_audit.resolve()
    assert result['source_terminal_audit_sha256'] == sha(args.prior_terminal_audit)
    assert result['whole463_own_AR_qualified'] and result['active_request_ceiling'] == 64
    actual = audit['metrics']['metrics']['host_stream_paper']
    assert result['Goodput'] == actual['Goodput'] and result['attained'] == actual['attained']
    assert not (result['measured_above_historical_floors'] and result['attained'] / 463 >= .8), \
        'A screened winning435 requires the predeclared matched comparison next.'
    for name in ('ROOT436_REAL_PRECONSTRUCTOR_RESULT.json',
                 'ROOT436_FROZEN_COLD_PARENT_RESULT.json',
                 'ROOT436_TERMINAL_ADMISSION_HOST_RESULT.json'):
        value = json.loads((D / name).read_text())
        assert value['status'].startswith('PASS_HOST436') and value['NPU_used'] is False
    contract = json.loads((D / 'ROOT436_TERMINAL_ADMISSION_HOST_RESULT.json').read_text())
    assert contract['terminal_auditor_sha256'] == sha(D / 'audit_ROOT436_terminal_full_HOST.py')
    record = json.loads((P / 'SOURCE436_RECEIPT.json').read_text())
    assert record['freeze_sha256'] == '4390b843830383905243d8dfb9aa37328a2c4a0076184cb171605b6cca39e5a3'
    assert record['archive_sha256'] == 'e9d88c023243b1eb67cc410979b126eb9acf047f3bcad0ea1aa4f6ef2144ddc9'
    env = dict(os.environ)
    env.update(USER='root', LOGNAME='root', SHELL='/bin/bash', LANG='C.UTF-8',
        PATH='/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin',
        PYTHONDONTWRITEBYTECODE='1')
    for line in (P / 'launch436.sh').read_text().splitlines():
        for value in shlex.split(line.rstrip(' \\')):
            if re.fullmatch(r'SPECSLO_V[0-9]+_ROOT_FREEZE_SHA=[a-f0-9]{64}', value):
                key, setting = value.split('=', 1)
                env[key] = setting
    env['SPECSLO_V257_ROOT_FREEZE_SHA'] = 'b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2'
    lines = [v for v in (P / 'startup436.sh').read_text().splitlines()
             if v.startswith('export PYTHONPATH=')]
    assert len(lines) == 1
    path = lines[0].removeprefix('export PYTHONPATH=')
    frozen = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1')
    for old, new in [('$repo', str(frozen / 'source/vllm-ascend-hust')),
                     ('$frozen', str(frozen)), ('$task_root', str(DONOR)), ('${PYTHONPATH:-}', '')]:
        path = path.replace(old, new)
    assert '$' not in path
    env['PYTHONPATH'] = path
    sys.path[:0] = path.split(':')
    for key, setting in env.items():
        if key.startswith('SPECSLO_V'):
            os.environ[key] = setting
    import guard436
    guard436.EXPECTED_FREEZE = record['freeze_sha256']
    guard436.EXPECTED_ARCHIVE = record['archive_sha256']
    guard436.prepared()
    storage = guard436.storage()
    assert storage['aggregate_private_bytes'] + 256 * 1024**2 <= 3584 * 1024**2, \
        'Retain room for a full bounded436 tag before any new model launch.'
    argv = [sys.executable, '-B', str(P / 'guard436.py'), '--tag', TAG,
        '--expected-freeze-sha256', record['freeze_sha256'],
        '--expected-archive-sha256', record['archive_sha256']]
    probe = subprocess.run(['npu-smi', 'info'], env=env, capture_output=True, text=True, timeout=30)
    assert probe.returncode == 0
    request = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), argv=argv,
        tag=TAG, source_freeze=record['freeze_sha256'], source_archive=record['archive_sha256'],
        prior_terminal_audit_sha256=sha(args.prior_terminal_audit),
        prior_result_sha256=sha(args.prior_result), prior_GP=result['Goodput'],
        actual_global_active_limit=64, all75_and6_fresh_NN_required=True,
        prelaunch_private_storage=storage, full256MiB_tag_reserved_before_launch=True,
        original227_681_and_warm_twozero_seal_full463_all3ownAR_required=True,
        prior_NN_permission_inherited=False, private_tag_MiB=256, aggregate_MiB=3584,
        original_worker_and_generation_timeout_seconds=6000,
        parent_qualification_receive_seconds=18000,
        Git_commit_or_push_requested_by_this_launcher=False)
    with (D / 'ROOT436_FULL_CURRENT75_PLUS6_LAUNCH_REQUEST.json').open('x') as f:
        json.dump(request, f, indent=2)
        f.write('\n')
    os.execve(sys.executable, argv, env)


if __name__ == '__main__':
    main()
