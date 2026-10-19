"""Freeze fresh311 model/cost checks only after the299 full audit completes."""
import datetime
import hashlib
import io
import json
import tarfile
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-single-prepare-target-math-cost-v311'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert not (P / 'FREEZE311.json').exists()
    receipt = B / 'tree-causal80-small-cohort-role-aware-audit-full-v209/receipt-capacity-risk-span-root-v299-B1to8-full463-ownAR-v1.json'
    terminal = read(receipt)
    assert terminal['status'] == 'complete' and not terminal['errors']
    assert terminal['sources_unchanged'] is True and not terminal['owned_remaining']
    assert terminal['devices_after']['all_idle'] is True
    independent_path = D / 'ROOT299_TERMINAL_INDEPENDENT_FULL_AUDIT.json'
    independent = read(independent_path)
    assert independent['status'] == 'terminal_completed_compute_audited' and not independent['errors']
    assert independent['whole_workload_AR_qualified'] is True
    assert independent['original_receipt_sha256'] == sha(receipt)
    assert read(P / 'HOST311_RESULT.json')['cases'] == 408
    assert read(P / 'HOST_GRID311_RESULT.json')['case_count'] == 408
    assert read(P / 'HOST_SPLIT311_SPANS_RESULT.json')['actual_cases'] == 408
    assert read(P / 'HOST_DEVICE311_RESULT.json')['invalid_bindings_rejected'] == 5
    assert read(P / 'HOST_LAUNCH311_RESULT.json')['status'].startswith('HOST_')
    previous = B / 'tree-concrete-device-span-dual-cache-target-cost-v296'
    for stem, ending in (('FIXTURE', '.json'), ('DECLARED', '_CASES.json')):
        assert sha(P / f'{stem}311{ending}') == sha(previous / f'{stem}296{ending}')
    files = {}
    for directory, version in ((previous, 296), (B / 'tree-capacity-request-risk-span-full-v299', 299)):
        freeze = directory / f'FREEZE{version}.json'
        for path, digest in read(freeze)['files'].items():
            assert path not in files or files[path] == digest
            files[path] = digest
        files[str(freeze)] = sha(freeze)
    for directory in (P, B / 'tree-single-validated-prefix-prepare-host-v305',
                      B / 'tree-single-validated-prefix-descriptors-host-v304'):
        for path in directory.iterdir():
            if path.is_file() and path.suffix in ('.py', '.sh', '.json'):
                if path.suffix == '.py':
                    compile(path.read_text(), str(path), 'exec')
                files[str(path)] = sha(path)
    for path in (Path(__file__), D / 'build_ROOT311_single_prepare_target_cost_HOST.py',
                 D / 'audit_ROOT311_target_math_and_cost_HOST.py', receipt, independent_path):
        files[str(path)] = sha(path)
    for path, digest in files.items():
        assert sha(path) == digest, path
    review = dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  change='Only305 HOST validation/staging replaces296 preparation',
                  geometry='M120/S24/uniformQ5/KV2560/C96',
                  raw_operators_and_original_attention_and_model_precision_unchanged=True,
                  fresh_target_cases_per_rank=408, target_ranks=[1, 2, 3],
                  independent_committed_prefixes=72,
                  configurations=11, budget_arms=[4, 5, 6, 7, 8], rotated_rounds=20,
                  original_startup_inventory_not_run_by_this_diagnostic=True,
                  NPU_qualification_inherited=False, whole463_AR_qualified=False,
                  real_draft_cost_measured=False, Goodput=None, adopted=False)
    freeze = P / 'FREEZE311.json'
    with freeze.open('x') as stream:
        json.dump(dict(schema='ROOT311.frozen-target-math-cost.v1', files=files,
                       review=review, model_qualification_inherited=False), stream, indent=2)
        stream.write('\n')
    inventory = {**files, str(freeze): sha(freeze)}
    archive = P / 'SOURCE311.tar.gz'
    with tarfile.open(archive, 'x:gz') as saved:
        for path, digest in sorted(inventory.items()):
            data = Path(path).read_bytes()
            assert hashlib.sha256(data).hexdigest() == digest
            member = tarfile.TarInfo(path.lstrip('/'))
            member.mode, member.size, member.mtime = 0o600, len(data), 0
            saved.addfile(member, io.BytesIO(data))
    with tarfile.open(archive, 'r:gz') as saved:
        assert len(saved.getmembers()) == len(inventory)
        for member in saved:
            assert member.isfile()
            assert hashlib.sha256(saved.extractfile(member).read()).hexdigest() == inventory['/' + member.name]
    pins = dict(freeze_sha256=sha(freeze), archive_sha256=sha(archive), members=len(inventory))
    (P / 'SOURCE311_RECEIPT.json').write_text(json.dumps(pins, indent=2) + '\n')
    print(json.dumps(pins))


if __name__ == '__main__':
    main()
