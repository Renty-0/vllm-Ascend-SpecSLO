"""Prepare a fresh full candidate; no model scope is inherited or granted."""
import json
from pathlib import Path

B = Path('/root/SpecSLO/budget-calibration-20261015')
OLD = B / 'tree-dynamic-root-alternate-eager-compat-full-v287'
P = B / 'tree-capacity-request-risk-span-full-v299'


def change(path, before, after, count=1):
    source = path.read_text()
    assert source.count(before) == count, (path, before, source.count(before))
    path.write_text(source.replace(before, after))


def main():
    P.mkdir(exist_ok=False)
    for source in OLD.iterdir():
        if not source.is_file() or (source.suffix not in {'.py', '.sh'} and source.name != 'FIXTURE287.json'):
            continue
        target = P / source.name.replace('287', '299')
        if source.suffix == '.json':
            target.write_bytes(source.read_bytes())
        else:
            target.write_text(source.read_text()
                              .replace('tree-dynamic-root-alternate-paired-prefix-grid-full-v287', P.name)
                              .replace(OLD.name, P.name).replace('287', '299'))
    paths = [B / name for name in (
        'tree-concrete-device-span-dual-cache-target-cost-v296',
        'tree-spare-capacity-budget-host-v290',
        'tree-request-conditional-risk-gain-host-v294',
        'tree-prefix-precopy-dual-cachewrite-host-v288',
        'tree-contiguous-prefix-span-copy-host-v292')]
    for name in ('check299.py', 'runtime299.py'):
        path = P / name
        source = path.read_text()
        anchor = "B = P.parent\n"
        assert source.count(anchor) == 1
        extra = 'sys.path.extend(' + repr([str(p) for p in paths]) + ')\n'
        path.write_text(source.replace(anchor, anchor + extra))
    change(P / 'startup299.sh', 'export PYTHONPATH=',
           'export PYTHONPATH=' + ':'.join(str(p) for p in paths) + ':')
    (P / 'static_state299.py').write_text(
        '"""Frozen296 kernels; fresh full-instance neural qualification required."""\n'
        'from static_state296 import ensure_zero,bind_checked_prepare,install,require_copy_bindings\n')
    (P / 'oracle299.py').write_text('"""Fresh execution required for all408 original neural cases."""\nfrom oracle296 import *\n')
    (P / 'audit_static_calls299.py').write_text('"""Audit actual296 capture families; no inherited neural proof."""\nfrom audit_static_calls296 import *\n')
    (P / 'budget299.py').write_text('"""Use290 capacity reservation with unchanged bound218 repair solver."""\nfrom budget290 import *\n')
    for name in ('scope299.py', 'audit299.py'):
        path = P / name
        path.write_text(path.read_text().replace('|ROOT280:copy-grid:', '|ROOT296:copy-grid:'))
    for name in ('live299.py', 'debug_snapshot299.py'):
        path = P / name
        path.write_text(path.read_text().replace('from paired274 import ', 'from risk299 import '))
    change(P / 'live299.py',
           "if plan.candidate_budget>=5:result=paired(engine._ROOT299_policy,request,**args)\n    elif plan.candidate_budget==4:result=observe_primary(engine._ROOT299_policy,request,**args)",
           "if plan.candidate_budget>=1:result=paired(engine._ROOT299_policy,request,**args)")
    change(P / 'extension299.py', "__import__('paired274').snapshot", "__import__('risk299').snapshot")
    change(P / 'extension299.py', "getattr(engine._branch_tail_state,'copy270_warm_widths',[])",
           "([engine._branch_tail_state.prefix296_copies.width] if not engine.is_draft else [])")
    change(P / 'audit299.py', "need(paired['generation']is not None and paired['by_generation'],",
           "need(paired['generation']is not None and bool(paired['by_generation']),")
    review = dict(status='SOURCE_ONLY_new299_not_frozen_not_NPU_run',
        mechanism='Capacity290 primary reservation + causal request risk294 + frozen span/dual-cache296',
        actual_models_sampler_and_arrivals_unchanged=True,
        original_startup_components_and_408target45draft20postbranch_required=True,
        all463_independent_own_AR_required=True,
        previous_qualification_inherited=False, Goodput=None, adopted=False)
    (P / 'SOURCE_ONLY299_REVIEW.json').write_text(json.dumps(review, indent=2) + '\n')
    print(json.dumps(review))


if __name__ == '__main__':
    main()
