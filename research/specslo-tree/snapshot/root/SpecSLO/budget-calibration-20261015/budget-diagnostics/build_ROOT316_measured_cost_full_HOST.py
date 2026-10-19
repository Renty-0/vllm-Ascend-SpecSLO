"""Prepare cost-aware full316 only after311 has fresh independent evidence."""
import hashlib
import io
import json
import re
import tokenize
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B / 'budget-diagnostics'
OLD = B / 'tree-capacity-risk-single-prepare-full-v307'
P = B / 'tree-measured-cost-risk-single-prepare-full-v316'
POLICY = B / 'tree-current-prepare-cost-eligibility-host-v312'


def rename(text, before, after, *, python=True):
    # Version identifiers never change numeric constants or unrelated SHAs.
    pattern = re.compile(r'(?<![0-9])' + re.escape(before) + r'(?![0-9])')
    if not python:
        pattern = re.compile(r'(?<=[A-Za-z_])' + re.escape(before) + r'(?![0-9])')
        pieces = re.split(r'([0-9a-f]{64})', text)
        return ''.join(piece if re.fullmatch(r'[0-9a-f]{64}', piece)
                       else pattern.sub(after, piece) for piece in pieces)
    converted = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        value = token.string
        if token.type != tokenize.NUMBER:
            pieces = re.split(r'([0-9a-f]{64})', value)
            value = ''.join(piece if re.fullmatch(r'[0-9a-f]{64}', piece)
                            else pattern.sub(after, piece) for piece in pieces)
        converted.append(token._replace(string=value))
    return tokenize.untokenize(converted)


def main():
    audit_path = D / 'ROOT311_INDEPENDENT_TARGET_MATH_AND_COST_AUDIT.json'
    audit = json.loads(audit_path.read_text())
    assert audit['status'].startswith('PASS_') and audit['target_cases_per_rank'] == 408
    assert audit['owned_cleanup_complete'] is audit['all4_idle_at_terminal'] is True
    assert len(audit['target_costs']) == 33
    POLICY.mkdir(exist_ok=False)
    base = B / 'tree-measured-branch-cost-eligibility-host-v309'
    for source in base.glob('*.py'):
        text = rename(source.read_text(), '309', '312')
        (POLICY / source.name.replace('309', '312')).write_text(text)
    path = POLICY / 'cost312.py'
    text = path.read_text()
    old_evidence = D / 'ROOT296_INDEPENDENT_TARGET_MATH_AND_COST_AUDIT.json'
    old_digest = hashlib.sha256(old_evidence.read_bytes()).hexdigest()
    new_digest = hashlib.sha256(audit_path.read_bytes()).hexdigest()
    assert text.count(str(old_evidence)) == 1 and text.count(old_digest) == 1
    path.write_text(text.replace(str(old_evidence), str(audit_path)).replace(old_digest, new_digest))
    path = POLICY / 'budget312.py'
    text = path.read_text()
    before = 'or not 0 <= primary <= expanded + 1e-12:'
    after = 'or primary < 0 or expanded < 0:'
    assert text.count(before) == 1
    path.write_text(text.replace(before, after).replace('finite monotone actual policy gains required',
                                                      'finite nonnegative actual policy gains required'))
    P.mkdir(exist_ok=False)
    for source in OLD.iterdir():
        if not source.is_file() or (source.suffix not in ('.py', '.sh') and source.name != 'FIXTURE307.json'):
            continue
        target = P / source.name.replace('307', '316')
        if source.suffix == '.json':
            target.write_bytes(source.read_bytes())
        else:
            target.write_text(rename(source.read_text().replace(OLD.name, P.name), '307', '316',
                                     python=source.suffix == '.py'))
    for name in ('check316.py', 'runtime316.py'):
        path = P / name
        text = path.read_text()
        anchor = 'B = P.parent\n'
        assert text.count(anchor) == 1
        path.write_text(text.replace(anchor, anchor + 'sys.path.append(' + repr(str(POLICY)) + ')\n'))
    path = P / 'startup316.sh'
    text = path.read_text()
    assert text.count('export PYTHONPATH=') == 1
    path.write_text(text.replace('export PYTHONPATH=', 'export PYTHONPATH=' + str(POLICY) + ':'))
    (P / 'budget316.py').write_text(
        '"""Measured311 target costs constrain actual risk-predicted branch eligibility."""\n'
        'from budget312 import *\n')
    path = P / 'audit316.py'
    text = path.read_text()
    before = "any(call['N']>8 and any(b>4 for b in call['budgets']) for call in calls)"
    after = "any(any(b>4 for b in call['budgets']) for call in calls)"
    assert text.count(before) == 1
    text = text.replace(before, after)
    text = text.replace('actual larger-N B5 scope exercised naturally',
                        'actual cost-selected branch executed; large-N capability separately proven by fresh408 math')
    path.write_text(text)
    for directory in (P, POLICY):
        for source in directory.glob('*.py'):
            compile(source.read_text(), str(source), 'exec')
    review = dict(status='SOURCE_ONLY_measured_cost316_not_frozen_or_NPU_executed',
        target_cost_evidence=dict(path=str(audit_path), sha256=new_digest),
        only_serving_deltas_from299=['305 HOST validation/staging',
                                    'priority-weighted branch progress must beat measured311 relative target cost'],
        all_original_precision_sampler_arrivals_SLO_and_budget_limits_unchanged=True,
        physical_geometry='M120/S24/uniformQ5/KV2560;C96/future96/global192',
        workload_branch_scope_predeclared='Require genuine actual branch at any N; ROI may correctly skip large-N branches. Fresh408 math still proves the entire larger-N capability.',
        full463_ownAR_all3targets_and_actual_branch_compaction_still_required=True,
        all_startup_draft_followup_warm_twozero_seal_gates_still_required=True,
        real_draft_cost_missing_from_proxy=True, no_Goodput_guarantee=True,
        finite_noisy_negative_marginal_gains_rejected_as_unprofitable=True,
        new_model_qualification_inherited=False, NPU_used=False, Goodput=None, adopted=False)
    (P / 'SOURCE_ONLY316_REVIEW.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps(review))


if __name__ == '__main__':
    main()
