"""Independently preserve the actual pre-measurement V116 failure."""
import datetime
import hashlib
import json
from pathlib import Path

B = Path(__file__).resolve().parent.parent
P = B / 'tree-s35-combined-controls-launchfix-v116'
TAG = 's35-CPUclean-familycap-paired-full463-v116-1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    receipt_path = P / ('local-receipt-' + TAG + '.json')
    r = json.loads(receipt_path.read_text())
    frozen = json.loads((P / 'PREPARED_SOURCE_HASHES_V116.json').read_text())
    expected = {**frozen['files'], **frozen['external_files']}
    for p, digest in expected.items():
        assert sha(p) == digest, p
    assert r['status'] == 'failed' and r['child']['exit_code'] == 1
    assert r['physical_cards'] == [2, 5, 6, 7]
    assert r['experimental_sources_unchanged'] is r['external_sources_unchanged'] is True
    assert r['owned_remaining'] == []
    assert r['devices_before']['all_idle'] is r['devices_after']['all_idle'] is True
    assert r['pre_verifier']['exit_code'] == r['post_verifier']['exit_code'] == 0
    assert r['formal_Goodput'] is r['online_qualified'] is False
    assert sha(r['source_snapshot']) == r['source_snapshot_sha256']
    artifacts = {str(receipt_path): sha(receipt_path)}
    startup = {}
    for rank in range(4):
        f = P / 'startup-progress' / ('MC2-preflight-rank' + str(rank) + '.json')
        d = json.loads(f.read_text())
        assert d['rank'] == rank and d['local_passed'] is True
        assert d['failed_checks'] == [] and d['all4rank_contract_passed'] is True
        artifacts[str(f)] = sha(f)
    for rank in (1, 2, 3):
        f = P / 'startup-progress' / ('prefill-rank' + str(rank) + '.json')
        d = json.loads(f.read_text())
        assert d['status'] == 'complete' and d['completed'] == d['total'] == 201
        artifacts[str(f)] = sha(f)
        f = P / 'startup-progress' / ('finite-target-rank' + str(rank) + '.json')
        d = json.loads(f.read_text())
        assert d['status'] == 'complete' and d['fixtures'] == len(d['native_keys']) == 44
        assert d['original_native_flags_unchanged'] is True
        artifacts[str(f)] = sha(f)
        startup[str(rank)] = dict(prefill=201, finite_target=44)
    f = P / 'startup-progress' / 'finite-draft-rank0.json'
    d = json.loads(f.read_text())
    assert d['status'] == 'complete' and d['explicit_keys'] == 86
    assert d['existing_native_generic_keys_retained'] == 28
    assert d['original_native_flags_unchanged'] is True
    artifacts[str(f)] = sha(f)
    startup['0'] = dict(finite_draft=86, retained_generic=28)
    errors = []
    out = P / 'local-output' / TAG
    for rank in range(4):
        f = out / ('rank' + str(rank) + '-last-error.json')
        d = json.loads(f.read_text())
        assert d['rank'] == rank
        raw = f.read_text()
        assert 'independent-normal-draft-original-numeric-and-exact-ABI' in raw
        assert 'fatal-no-publication-notice' in raw
        if rank == 0:
            assert 'Independent normal draft has an undeclared real request count' in raw
        errors.append(dict(rank=rank, path=str(f), sha256=sha(f)))
        artifacts[str(f)] = sha(f)
    f = out / 'worker-exit.json'
    d = json.loads(f.read_text())
    assert d['closed'] is d['all_exited'] is True
    artifacts[str(f)] = sha(f)
    assert not list(out.glob('**/*completed*.json'))
    report = dict(schema='ROOT.V116.actual-failure-audit.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        status='passed_failure_forensics', experiment_status='failed',
        original_receipt=str(receipt_path), artifact_sha256=artifacts,
        unchanged_frozen_source_files=len(frozen['files']),
        unchanged_external_files=len(frozen['external_files']),
        actual_completed_startup=startup, actual_MC2_preflight_all4=True,
        first_scope='V101-S19-N15-progressive', first_scope_N=15,
        root_cause='independent_history_v75.recompute_normal_draft rejects N15 outside its all4 stage; obsolete allowed counts (17,32)',
        secondary_failure='rank0 fatal cleanup enters stage sequence 5 while targets enter independent-normal-draft comparison; next cleanup stage remains mismatched',
        actual_completed_S19_or_S35_scopes=0, actual_S35_device_qualified=False,
        full463_started=False, formal_Goodput=False, performance_claim=False,
        four_rank_errors=errors, original_sources_unchanged=True,
        all_owned_exited=True, all_four_cards_idle_after=True,
        correction_requires_new_epoch=True)
    target = B / 'budget-diagnostics' / 'ROOT_V116_ACTUAL_FAILURE_AUDIT.json'
    with target.open('x') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=report['status'], path=str(target), sha256=sha(target),
        actual_S35_device_qualified=False, formal_Goodput=False)))


if __name__ == '__main__':
    main()
