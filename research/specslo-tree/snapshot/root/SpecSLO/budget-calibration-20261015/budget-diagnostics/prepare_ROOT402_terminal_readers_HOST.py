"""Bind post-run readers to the exact402 freeze and changed admission claim."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

D = Path(__file__).resolve().parent
B = D.parent
P = B / 'tree-global64-normal24-typed-admission-v402'


def one(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


def main():
    spec = importlib.util.spec_from_file_location('_ROOT402_reviewed_text_rename',
                                                D / 'prepare_ROOT402_full_integration_HOST.py')
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    receipt = json.loads((P / 'SOURCE402_RECEIPT.json').read_text())
    source = helper.rename_text((D / 'audit_ROOT398_terminal_full_HOST.py').read_text())
    source = one(source, 'dfb6cfd44d2f524d7ad28892ec055d50d1035910fb71dbbc5486dc060cb5e73d', receipt['freeze_sha256'])
    source = one(source, '5c79ae661393d9bb347dd6f53ff447d34e09fa121bc90d4d43844cdb63357f6a', receipt['archive_sha256'])
    audit = D / 'audit_ROOT402_terminal_full_HOST.py'
    assert not audit.exists()
    ast.parse(source)
    audit.write_text(source)
    source = helper.rename_text((D / 'read_ROOT398_common_result_HOST.py').read_text())
    source = one(source, "and a['completed_compute_audit']['ROOT402_original_best257_draft_and_allocator_unchanged'] is True,",
                 "and a['completed_compute_audit']['ROOT402_original_best257_draft_and_allocator_unchanged'] is False\n"
                 "         and a['completed_compute_audit']['ROOT402_best257_draft_neural_math_unchanged'] is True\n"
                 "         and a['completed_compute_audit']['ROOT402_normal_work_policy_limit'] == 24\n"
                 "         and len(a['completed_compute_audit']['ROOT402_actual_normal_admission']) == 4,")
    source = one(source, "        Raw=raw, Goodput=gp, attained=actual['attained'], requests=463,",
                 "        Raw=raw, Goodput=gp, attained=actual['attained'], requests=463,\n"
                 "        actual_normal_work_policy_limit=24,\n"
                 "        actual_class_age_admission=a['completed_compute_audit']['ROOT402_actual_normal_admission'],\n"
                 "        prior_global64_control_GP=214.96187487135404,\n"
                 "        GP_over_prior_global64_control=gp/214.96187487135404,")
    reader = D / 'read_ROOT402_common_result_HOST.py'
    assert not reader.exists()
    ast.parse(source)
    reader.write_text(source)
    prior = B / 'tree-causal80-small-cohort-role-aware-audit-full-v209/receipt-active64-whole-ticket-S24-segments-root-v398-full463-ownAR-v1.json'
    output = D / 'MUST_NOT_EXIST_ROOT402_OLD398_RESULT.json'
    assert not output.exists()
    tested = subprocess.run([sys.executable, '-B', str(audit), '--tag',
        'active64-whole-ticket-S24-segments-root-v398-full463-ownAR-v1',
        '--expected-receipt-sha256', hashlib.sha256(prior.read_bytes()).hexdigest(),
        '--output', str(output)], capture_output=True, text=True)
    assert tested.returncode != 0 and not output.exists()
    assert 'terminal402 active64 supervisor' in tested.stderr
    result = dict(status='PASS_HOST402_TERMINAL_READERS_BOUND_AND_PRIOR398_REJECTED',
                  expected_freeze_sha256=receipt['freeze_sha256'],
                  expected_archive_sha256=receipt['archive_sha256'],
                  source_auditor_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
                  source_reader_sha256=hashlib.sha256(reader.read_bytes()).hexdigest(),
                  different_qualified_control_receipt_rejected_before_metrics=True,
                  Raw=None, Goodput=None, NPU_used=False)
    (D / 'HOST402_TERMINAL_READERS_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(status=result['status'], NPU_used=False)))


if __name__ == '__main__':
    main()
