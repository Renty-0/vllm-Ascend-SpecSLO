"""Read retained terminal failure evidence; never launch models or grant qualification."""
import datetime
import fcntl
import gzip
import hashlib
import json
import subprocess
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
TAG = 'parent4-child4-current-M120-root-v391-early392-bounded-v1'
ROOT = B / 'tree-bounded-parent4-shared4-eager-diagnostic-v199'
ACTUAL = ROOT / 'actual' / TAG


def require(value, message):
    if value is not True:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def main():
    receipt_path = ROOT / f'receipt-{TAG}.json'
    receipt = read(receipt_path)
    require(receipt['status'] == 'failed' and receipt['child']['exit_code'] == 1,
            'Expected terminal child exception, not a successful or running job')
    require(receipt['sources_unchanged'] is True and receipt['owned_remaining'] == [],
            'Source and owned process cleanup')
    require(receipt['devices_after']['all_idle'] is True, 'Terminal four-card idle')
    require(receipt['post_verifier']['actual_result']['status'] == 'pass',
            'Actual retained environment verifier')
    require(read(ACTUAL / 'worker-exit.json') == dict(closed=True, all_exited=True),
            'All four workers exited')
    with (B / '.ROOT_NPU_EXECUTION.lock').open('a+') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle, fcntl.LOCK_UN)

    parts = (
        ('tree-parent4-child4-original-bounded-v386', 386),
        ('tree-parent4-full-archive-guard-v387', 387),
        ('tree-parent4-owned-exit-path-v388', 388),
        ('tree-parent4-atomic-output-storage-guard-v389', 389),
        ('tree-parent4-current-M120-target-protocol-v391', 391),
    )
    pinned = {
        386: '0909f2ed2a3c5fbd0ef94a3e9122611c48bf4cc4e164afdbd558ce27afbb1b96',
        387: 'b15f3876538e5f1a04e2d8c16950e16e264034205783f45d4d08e6dcad3be1a4',
        388: '1d660aacce3653ea25492c29e91cebd8dbf5862b5a81af1f09c3473c5b4557a9',
        389: 'e233543cad3a82dc6defce6f17f7d32c30a6e96a58a39c9ec12a7298a49cde29',
        391: 'a9a4bfaa81f9bc8f95a75a533612fb1bd8b8202638a6f65c6d35b8185fe8fa98',
    }
    sources = {}
    for folder, version in parts:
        path = B / folder / f'FREEZE{version}.json'
        require(digest(path) == pinned[version], 'Frozen source manifest digest')
        for filename, expected in read(path)['files'].items():
            require(filename not in sources or sources[filename] == expected,
                    'Overlapping source hash agreement')
            sources[filename] = expected
        sources[str(path)] = pinned[version]
    require(len(sources) == 4288, 'Actual complete source scope')
    for filename, expected in sources.items():
        path = Path(filename)
        require(path.is_file() and not path.is_symlink() and digest(path) == expected,
                'Original frozen source bytes: ' + filename)

    workers = []
    failure_stage = 'V199/pass0/N16-parent16-P4-E1-B4-R0-M20/real-parent16-B4/greedy/first/prepare'
    cause = "ModuleNotFoundError: No module named 'projection247'"
    for rank in range(4):
        error = read(ACTUAL / f'rank{rank}-last-error.json')['error']
        require(failure_stage in error and cause in error,
                'All ranks retained the same first-case stage exception')
        basic = read(ACTUAL / 'variable-B4-own-full-process' / f'rank{rank}-component.json.gz')
        causal = read(ACTUAL / 'causal80-own-current-process' / f'rank{rank}-component.json.gz')
        require(basic['completed'] is basic['component_numerical_qualified'] is True
                and basic['errors'] == [], 'Fresh original B4 completed')
        require(causal['completed'] is causal['math_independent_qualified'] is True
                and causal['errors'] == [] and causal['Native_flags_assigned'] is False,
                'Fresh original causal80 completed')
        workers.append(dict(rank=rank, original_B4_completed=True,
                            original_causal80_completed=True,
                            error_last_line=error.splitlines()[-1]))
    require(not list((ACTUAL / 'bounded199').glob('*.json.gz')),
            'No complete new bounded case was retained')
    prefill = []
    for rank in (1, 2, 3):
        value = read(ACTUAL / 'startup-progress' / f'prefill-rank{rank}.json')
        require(value['status'] == 'complete' and value['completed'] == value['total'] == 201,
                'Actual complete original prefill')
        value = read(ACTUAL / 'startup-progress' / f'finite-target-rank{rank}.json')
        require(value['status'] == 'complete' and len(value['native_keys']) == 44,
                'Actual original target inventory')
        prefill.append(dict(rank=rank, prefill=201, target_inventory=44))
    draft = read(ACTUAL / 'startup-progress/finite-draft-rank0.json')
    require(draft['status'] == 'complete' and draft['explicit_keys'] == 86
            and draft['existing_native_generic_keys_retained'] == 28,
            'Actual original 114-entry draft inventory')

    devices = []
    for card in (2, 5, 6, 7):
        argv = ['npu-smi', 'info', '-t', 'proc-mem', '-i', str(card), '-c', '0']
        process = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        require(process.returncode == 0 and 'No process in device.' in process.stdout,
                'Independent card idle check')
        devices.append(dict(card=card, idle=True))
    value = dict(
        schema='ROOT391.retained-terminal-failure-audit.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='EXPECTED_IMPORT_FAILURE_RETAINED_AND_CLEANUP_VERIFIED',
        receipt=str(receipt_path), receipt_sha256=digest(receipt_path),
        actual_output=str(ACTUAL), source_members_rehashed=len(sources),
        failure_stage=failure_stage, failure_cause=cause,
        source_manifest_sha256=pinned[391], workers=workers,
        original_inventory=prefill, original_draft_inventory=114,
        bounded_cases_completed=0, sources_unchanged=True, environment_verified=True,
        worker_exit_verified=True, owned_remaining=[], independently_idle_devices=devices,
        guardian_error_context='Generic nonzero-child gate says within1800seconds; actual child exit_code=1, elapsed about1232s, not evidence of an1800s timeout.',
        observer392_full_model_qualification=False,
        original173_proofs_recomputed_by_HOST=False,
        new_target_hidden_KV_proof_completed=False,
        production_permission=False, full463=False, Raw=None, Goodput=None,
        pause_requested_by_user=True, next_NPU_job_launched=False,
    )
    (D / 'ROOT391_TERMINAL_RETAINED_AUDIT.json').write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: value[key] for key in (
        'status', 'source_members_rehashed', 'bounded_cases_completed',
        'failure_cause', 'owned_remaining', 'independently_idle_devices')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
