"""Freeze the control-plane-only normal24/global64 delta and original full gates."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
P = B / 'tree-global64-normal24-typed-admission-v402'
PARENT = B / 'tree-active64-whole-proposal-segment-limit-v398'
REPO = Path('/root/SpecSLO/checkpoints/github-SpecSLO-Tree')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert not (P / 'FREEZE402.json').exists()
    dest = REPO / 'research/specslo-tree/global64-normal24-v402'
    assert not dest.exists()
    prerequisite_names = [
        'HOST402_RESULT.json', 'HOST402_SCOPE_RESULT.json', 'HOST402_CONSTRUCTION_RESULT.json',
        'HOST402_CAPACITY_RESULT.json', 'HOST402_PACKING_RESULT.json', 'HOST402_LIVE_BINDING_RESULT.json',
        'HOST402_TYPED_ALLOCATION_RESULT.json', 'HOST402_CLEAN_STARTUP_RUN.json']
    for name in prerequisite_names:
        row = json.loads((P / name).read_text())
        assert row['status'].startswith('PASS_HOST402') and row['NPU_used'] is False
    parent_freeze = PARENT / 'FREEZE398.json'
    assert sha(parent_freeze) == 'dfb6cfd44d2f524d7ad28892ec055d50d1035910fb71dbbc5486dc060cb5e73d'
    inherited = json.loads(parent_freeze.read_text())['files']
    assert all(sha(name) == digest for name, digest in inherited.items())
    plan = dict(status='FROZEN_HOST_PASS_DEVICE_NOT_LAUNCHED', global_active_limit=64,
                current_target=dict(N=24, C=96, M=120, primary_side_groups=24),
                actual_normal_work_admission=dict(previous_limit=32, policy_limit=24,
                    original_priority_age_and_owner_order=True, deferred_requests_remain_active=True),
                original_typed_gain_model_and_role_limits_unchanged=True,
                normal_candidates=96, future_eager_candidates=96, global_selected_candidates=192,
                original_B3_parent_B3_child_eager_filter_and_window_gates_unchanged=True,
                original_optimized_draft_neural_code_unchanged=True,
                original_allocator_admission_unchanged=False,
                numerical_backend_changed=False, new_Native_qualification_flags_assigned=False,
                fresh_original227_and_additional681_and_full463_own_AR_required=True,
                warm_twozero_seal_required=True,
                reference_control398=dict(Raw=599.5623377435959, Goodput=214.96187487135404,
                    attained=166, requests=463, active_limit=64),
                NPU_launched=False, model_qualification=False, Raw=None, Goodput=None)
    (P / 'PLAN402.json').write_text(json.dumps(plan, indent=2) + '\n')
    derivation = json.loads((P / 'DERIVATION402.json').read_text())
    derivation.update(status='HOST_PASS_PREPARED_FOR_FRESH_FULL_QUALIFICATION',
                      actual_normal402_audit_added=True, actual_engine_and_shaper_LIVE_binding_HOST_passed=True,
                      explicit_old_allocator_unchanged_claim_removed=True,
                      current_typed_source_findings=str(D / 'ROOT402_CURRENT_TYPED_POLICY_FINDINGS.json'))
    (P / 'DERIVATION402.json').write_text(json.dumps(derivation, indent=2) + '\n')
    own = [f for f in P.iterdir() if f.is_file()]
    tools = [Path(__file__).resolve(), D / 'prepare_ROOT402_full_integration_HOST.py',
             D / 'run_ROOT402_actual_startup_HOST.py', D / 'ROOT402_CURRENT_TYPED_POLICY_FINDINGS.json',
             D / 'archive_ROOT402_old_private_evidence.py', D / 'ROOT402_VERIFIED_PRIVATE_ARCHIVAL_RESULT.json',
             D / 'ROOT398_FINAL_FULL_RESULT.json']
    files = dict(inherited)
    files[str(parent_freeze)] = sha(parent_freeze)
    for path in own + tools:
        assert str(path) not in files
        files[str(path)] = sha(path)
        if path.suffix == '.py':
            ast.parse(path.read_bytes(), filename=str(path))
        if path.suffix == '.sh':
            subprocess.run(['bash', '-n', str(path)], check=True)
    freeze = dict(schema='ROOT402.normal24-global64-fresh-full-source-freeze.v1', files=files,
                  parent398_freeze_sha256=sha(parent_freeze),
                  parent398_device_qualification_inherited=False, global_active_limit=64,
                  normal_work_limit=24, actual_allocator_admission_changed=True,
                  original_typed_gain_model_and_neural_source_bodies_preserved=True)
    (P / 'FREEZE402.json').write_text(json.dumps(freeze, indent=2) + '\n')
    archive_files = dict(files, **{str(P / 'FREEZE402.json'): sha(P / 'FREEZE402.json')})
    archive = P / 'SOURCE402.tar.gz'
    with tarfile.open(archive, 'w:gz') as saved:
        for name in sorted(archive_files):
            saved.add(name, arcname=name.lstrip('/'), recursive=False)
    with tarfile.open(archive, 'r:gz') as saved:
        members = saved.getmembers()
        assert len(members) == len(archive_files)
        assert {m.name for m in members} == {name.lstrip('/') for name in archive_files}
        for member in members:
            assert member.isfile()
            assert hashlib.sha256(saved.extractfile(member).read()).hexdigest() == archive_files['/' + member.name]
    receipt = dict(freeze_sha256=sha(P / 'FREEZE402.json'), archive_sha256=sha(archive),
                   archive_bytes=archive.stat().st_size, archived_members=len(archive_files),
                   NPU_launched=False, model_qualification=False)
    (P / 'SOURCE402_RECEIPT.json').write_text(json.dumps(receipt, indent=2) + '\n')
    import sys
    sys.path.insert(0, str(P))
    import guard402
    guard402.EXPECTED_FREEZE, guard402.EXPECTED_ARCHIVE = receipt['freeze_sha256'], receipt['archive_sha256']
    guard402.prepared()
    # Evaluate the exact real supervisor metadata expression without executing it.
    namespace = guard402.execution_namespace()
    explicit_source = D / 'v154-root-execution/supervise_v154.py'
    function = next(n for n in ast.parse(explicit_source.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'execute_locked')
    expression = next(n.value for n in function.body if isinstance(n, ast.Assign)
                      and any(isinstance(t, ast.Name) and t.id == 'row' for t in n.targets))
    assert not {key.arg for key in expression.keywords if key.arg is not None} & set(namespace['SCOPE'])
    dest.mkdir()
    exported = {}
    for path in own + [P / 'FREEZE402.json', P / 'SOURCE402_RECEIPT.json']:
        shutil.copyfile(path, dest / path.name)
        exported[path.name] = dict(source=str(path), sha256=sha(path))
    (dest / 'tools').mkdir()
    for path in tools:
        shutil.copyfile(path, dest / 'tools' / path.name)
        exported['tools/' + path.name] = dict(source=str(path), sha256=sha(path))
    (dest / 'EXPORT_MANIFEST.json').write_text(json.dumps(exported, indent=2) + '\n')
    for relative, value in exported.items():
        assert sha(dest / relative) == value['sha256']
    (dest / 'VALIDATION.json').write_text(json.dumps(dict(
        status='FROZEN_HOST_PASS_FULL_DEVICE_VALIDATION_PENDING', global_active_limit=64,
        normal_work_limit=24, original_typed_priority_gain_and_role_accounting_preserved=True,
        HOST_typed_cases=800, HOST_exact_narrow_cases=587, HOST_capped_class_age_cases=213,
        HOST_original_controller_equivalence_and_segment_cases=1000,
        actual_clean_startup_and_live_source_binding=True,
        original_neural_gates_not_replaced=True, source_archive_private_not_public=True,
        format_ci='unavailable: pre-commit not installed; existing environment retained',
        NPU_launched=False, full463_completed=False, Raw=None, Goodput=None), indent=2) + '\n')
    print(json.dumps(dict(status='FROZEN_EXPORTED_HOST402_READY', **receipt)))


if __name__ == '__main__':
    main()
