"""Freeze307 only after296 terminal evidence passes independent audit."""
import datetime
import hashlib
import io
import json
import tarfile
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-capacity-risk-single-prepare-full-v307'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    assert not (P / 'FREEZE307.json').exists()
    parent_receipt=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/receipt-capacity-risk-span-root-v299-B1to8-full463-ownAR-v1.json'
    actual_parent=read(parent_receipt)
    assert actual_parent['status']=='complete'and not actual_parent['errors']
    assert actual_parent['sources_unchanged']is True and not actual_parent['owned_remaining']
    assert actual_parent['devices_after']['all_idle']is True
    assert actual_parent['actual_full_audit']['whole_workload_AR_qualified']is True

    audit_path = D / 'ROOT296_INDEPENDENT_TARGET_MATH_AND_COST_AUDIT.json'
    audit = read(audit_path)
    assert audit['status'].startswith('PASS_') and audit['target_cases_per_rank'] == 408
    assert audit['owned_cleanup_complete'] is True and audit['all4_idle_at_terminal'] is True
    required = ('HOST_FULL307_PROTOCOL_RESULT.json', 'HOST_COST307_RESULT.json',
                'HOST_BUDGET307_RESULT.json', 'HOST_PLANS307_RESULT.json',
                'HOST_PRIORITY307_RESULT.json', 'HOST_ADMISSION307_RESULT.json',
                'HOST_LIVE307_POST_SNAPSHOT_RESULT.json', 'HOST_FAULT307_RESULT.json',
                'HOST_SNAPSHOT307_RESULT.json', 'HOST_EAGER307_RESULT.json',
                'HOST_AUDIT307_RESULT.json', 'HOST_RISK_ALLOCATION307_RESULT.json',
                'HOST_LAUNCH307_RESULT.json')
    for name in required:
        assert read(P / name)['status'].startswith('HOST_'), name
    assert read(P / 'HOST_FULL307_PROTOCOL_RESULT.json')['actual_observer_cases'] == 408
    assert read(P / 'HOST_LIVE307_POST_SNAPSHOT_RESULT.json')['actual_policy_cases'] == 132
    assert read(P / 'HOST_RISK_ALLOCATION307_RESULT.json')['causal_risk_audit']['observed_verdicts'] == 3072
    previous = B / 'tree-dynamic-root-alternate-eager-compat-full-v287'
    assert sha(P / 'FIXTURE307.json') == sha(previous / 'FIXTURE287.json')
    files = {}
    for directory, epoch in (
        ('tree-dynamic-root-alternate-eager-compat-full-v287', 287),
        ('tree-capacity-request-risk-span-full-v299', 299),
        ('tree-concrete-device-span-dual-cache-target-cost-v296', 296),
    ):
        freeze = B / directory / f'FREEZE{epoch}.json'
        for path, digest in read(freeze)['files'].items():
            if path in files:
                assert files[path] == digest, path
            files[path] = digest
        files[str(freeze)] = sha(freeze)
    review = dict(schema='ROOT307.capacity-request-conditional-risk-span-full.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        changes=['Reserve prospective B4 normal capacity before branch upgrades',
                 'Per-request conditional hazards with leave-own-out class prior and censoring',
                 'All64 raw operators unchanged; single full immutable descriptor validation and no unused1792-row prefix staging'],
        gain_hyperparameters=dict(class_prior_strength=12, request_prior_strength=2, decay=31./32.),
        backend='305 HOSTprepare with exact frozen296 raw operators and original-attention M120/S24/Q5/KV2560',
        global_selected_candidates=192, normal_candidates=96, future_candidates=96,
        actual_target_primary_plus_side_groups=24, active_admission_cap=48,
        LIVE_normal_budget_domain=[1, 8], eager_parent_and_child_support='unchanged original B3 only',
        original_source_auditor_changes='Only two legacy B<=5 bounds become declared8; nonempty dict normalized bool',
        all_original_startup_component_gates_required=True,
        fresh408target_45draft_20postbranch_required=True,
        fresh_warm_twozero_seal_full463_ownAR_required=True,
        models_BF16_original_sampler_arrivals_SLO_unchanged=True,
        device_qualification_inherited=False, performance_qualified=False,
        Goodput=None, adopted=False)
    (P / 'INTEGRATION_REVIEW307.json').write_text(json.dumps(review, indent=2) + '\n')
    for directory in (P, B / 'tree-spare-capacity-budget-host-v290',
                      B / 'tree-request-conditional-risk-gain-host-v294',
                      B / 'tree-single-validated-prefix-descriptors-host-v304',
                      B / 'tree-single-validated-prefix-prepare-host-v305'):
        for path in directory.iterdir():
            if path.is_file() and path.suffix in {'.py', '.sh', '.json'}:
                if path.suffix == '.py':
                    compile(path.read_text(), str(path), 'exec')
                files[str(path)] = sha(path)
    for path in (Path(__file__), D / 'build_ROOT307_fast_prepare_full_HOST.py',
                 D / 'audit_ROOT296_target_math_and_cost_HOST.py', audit_path, parent_receipt, D / 'build_ROOT305_single_validated_prepare_HOST.py'):
        files[str(path)] = sha(path)
    for path, digest in files.items():
        assert sha(path) == digest, path
    freeze = P / 'FREEZE307.json'
    with freeze.open('x') as stream:
        json.dump(dict(schema='ROOT307.frozen-full-ownAR-required.v1', files=files,
                       review=review, prior_frozen_sources_edited=False,
                       NPU_qualification_inherited=False, only_delta_from299='prepare305 validation/staging; policies andrawkernels unchanged', adopted=False), stream, indent=2)
        stream.write('\n')
    inventory = {**files, str(freeze): sha(freeze)}
    archive = P / 'SOURCE307.tar.gz'
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
            assert member.isfile() and hashlib.sha256(saved.extractfile(member).read()).hexdigest() == inventory['/' + member.name]
    pins = dict(freeze_sha256=sha(freeze), archive_sha256=sha(archive), members=len(inventory))
    (P / 'SOURCE307_RECEIPT.json').write_text(json.dumps(pins, indent=2) + '\n')
    print(json.dumps(pins))


if __name__ == '__main__':
    main()
