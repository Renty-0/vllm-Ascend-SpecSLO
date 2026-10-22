#!/usr/bin/env python3
"""Actual allocator HOST control: correct eager3 cap can leave a_need unfit."""
import ast
import json
from pathlib import Path
import sys
import time

from audit_v121_split_draft_target_budget_host import (
    D, P, CURRENT, execute_nodes, fixture_state, sha, traced_allocate)


def main():
    started = time.perf_counter()
    assert sys.version_info[:2] == (3,11)
    source = P/'budget_policy.py'
    policy_type = execute_nodes(source,ast.parse(source.read_text()).body)['TreeBudgetPolicy']
    model = json.loads((P/'tree-gain-model.json').read_text())
    policy = policy_type(model)
    high_gap = (policy.base['tight'][4]+policy.base['tight'][5])/2
    fit_gap = (policy.base['tight'][2]+policy.base['tight'][3])/2
    without3 = traced_allocate(policy_type(model),(),(0,),
        {0:fixture_state(0,high_gap,max_gamma=5)},192)
    enforce3 = traced_allocate(policy_type(model),(),(0,),
        {0:fixture_state(0,high_gap,max_gamma=3)},192)
    fits3 = traced_allocate(policy_type(model),(),(0,),
        {0:fixture_state(0,fit_gap,max_gamma=3)},192)
    assert without3['eager_C'] == 5
    assert enforce3['eager_C'] == 0 and enforce3['actual_original_locals']['needs'][0] is None
    assert enforce3['actual_original_locals']['left'] == 96
    assert fits3['eager_C'] == 3
    cfg = ast.parse((CURRENT/'startup_config_v75.py').read_text())
    calls = [n for n in ast.walk(cfg) if isinstance(n,ast.Call) and ast.unparse(n.func)=='PEARLConfig']
    assert len(calls)==1
    supplied = {k.arg:k.value for k in calls[0].keywords}
    kwargs = {k:ast.literal_eval(supplied[k]) for k in (
        'spec_rhythm_max_eager_tokens','spec_rhythm_verification_budget')}
    assert kwargs['spec_rhythm_max_eager_tokens']==5
    assert kwargs['spec_rhythm_verification_budget']==96
    assert 'spec_rhythm_draft_token_budget' not in supplied
    hashes = {str(p):sha(p) for p in (source,P/'tree-gain-model.json',
        CURRENT/'startup_config_v75.py',Path(__file__).resolve(),
        D/'audit_v121_split_draft_target_budget_host.py',
        D/'V121_SPLIT_DRAFT_TARGET_BUDGET_HOST_AUDIT.json')}
    assert not any(n=='torch' or n.startswith(('torch.','torch_npu','vllm')) for n in sys.modules)
    output = D/'V121_EAGER3_GAP_HOST_SUPPLEMENT.json'
    report = dict(schema='V121.eager3-current-gap.HOST-supplement.v1',
        status='HOST_CONSTRAINT_COUNTEREXAMPLES_ONLY',argv=sys.argv,python=sys.executable,
        elapsed_seconds=time.perf_counter()-started,tests_run=4,passed=True,
        source_hashes=hashes,current_V119_startup=dict(max_eager=5,C=96,draft_budget=None),
        high_gap=high_gap,g3=policy.base['tight'][3],
        original_cap5=without3,enforced_cap3=enforce3,fit_gap3=fits3,
        ignored_wrapper_caller_cap3_can_grant5_then_select3=True,
        fixture_state_cap3_is_a_HOST_input_control_not_a_runtime_state_mutation=True,
        shared_budget_is_not_sufficient_explanation_of_zero_eager=True,
        actual_window_or_gate_observations=False,performance_claim=False,NPU_started=False,
        frozen_sources_modified=False)
    with output.open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(output=str(output),sha256=sha(output),
        tests_run=4,elapsed_seconds=report['elapsed_seconds'])))


if __name__=='__main__':
    main()
