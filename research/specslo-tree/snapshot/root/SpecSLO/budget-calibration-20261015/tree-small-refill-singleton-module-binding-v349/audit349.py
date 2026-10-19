"""Reopen actual independent small-refill and nextdraft data on all four roles."""
import json
from pathlib import Path

import runtime349
from prefill349 import audit as component_audit


def audit(directory):
    directory = Path(directory)
    result = component_audit(directory/'live-refill-small-cohort-own-current-process')
    reports = [json.loads((directory/f'rank{r}-numeric.json').read_text()) for r in range(4)]
    for rank, row in enumerate(reports):
        if row['rank'] != rank or row['audit'] != result or row['representative_sizes'] != [1,3,4,8]:
            raise RuntimeError('all actual roles and current independent audit required')
        if len(row['cases']) != 12 or len(row['warmup_cases']) != 4:
            raise RuntimeError('four actual warmups plus twelve actual prefill cases')
        if row['model_dtype'] != 'torch.bfloat16':
            raise RuntimeError('original BF16 precision retained')
    if json.loads((directory/'worker-exit.json').read_text()) != dict(closed=True, all_exited=True):
        raise RuntimeError('normal original worker shutdown required')
    if json.loads((directory/'completed.json').read_text()) != dict(replies=[
            ['calibrated', r, dict(numeric_cases=16, performance=False)] for r in range(4)],
            performance=False, adopted=False):
        raise RuntimeError('all original lifecycle replies required')
    return (dict(status='PASS_current_small_refill_graph_and_eager_model_component',
                 component=result, finite_prefill_cases=16, nextdraft_B4=True,
                 real_target_roles=3, real_draft_generation=True,
                 whole463_qualified=False, Goodput=None, adopted=False),
            'passed_current_small_refill_component_only',
            dict(real_target_roles=3, finite_prefill_cases=16, whole463_qualified=False,
                 no_performance_qualification=True))
