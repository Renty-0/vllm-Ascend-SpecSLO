"""Execute the frozen allocation closure, without importing Torch or a model."""
import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import types
from typing import Mapping

BASE = Path(__file__).resolve().parent.parent
PARENT = BASE / 'tree-s35-production-v101'
NATIVE = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/spec_rhythm.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    assert sys.version_info[:2] == (3, 11)
    paths = [Path(__file__), PARENT/'tree_entry.py', PARENT/'budget_policy.py', NATIVE]
    before = {str(p): sha(p) for p in paths}
    policy_ns = {}
    exec(compile((PARENT/'budget_policy.py').read_text(), str(PARENT/'budget_policy.py'), 'exec'), policy_ns)
    native_tree = ast.parse(NATIVE.read_text())
    plan_node = next(n for n in native_tree.body if isinstance(n, ast.ClassDef) and n.name == 'SpecRhythmBudgetPlan')
    plan_ns = dict(dataclass=dataclass, Mapping=Mapping)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[plan_node], type_ignores=[])), str(NATIVE), 'exec'), plan_ns)
    tree = ast.parse((PARENT/'tree_entry.py').read_text())
    install = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install')
    shape = next(n for n in install.body if isinstance(n, ast.FunctionDef) and n.name == 'shape')
    model = {'curves': {c: {str(n): 1.0+n/2 for n in range(1, 9)}
                        for c in ('coding', 'chat', 'summarization')}}
    policy = policy_ns['TreeBudgetPolicy'](model)
    namespace = dict(policy=policy, rhythm=types.SimpleNamespace(SpecRhythmBudgetPlan=plan_ns['SpecRhythmBudgetPlan']))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[shape], type_ignores=[])), str(PARENT/'tree_entry.py'), 'exec'), namespace)
    state = types.SimpleNamespace(slo_class='tight', max_gamma=5, output_token_budget=256,
        delivered_tokens=1, decode_elapsed_ms=0., decode_start_elapsed_ms=0., slo_tpot_ms=40.,
        projected_progress_gap=lambda wait: 1.)
    common = dict(plan_id=1, normal_request_indices=[], eager_request_indices=[0],
                  states={0: state}, projected_wait_ms=1., draft_token_budget=96)
    records = []
    for cap, reserve in ((1, 0), (3, 0), (3, 3), (5, 0)):
        result = namespace['shape'](None, **common, eager_token_cap=cap, eager_reserve_tokens=reserve)
        record = dict(configured_eager_token_cap=cap, configured_eager_reserve_tokens=reserve,
                      actual_typed_eager_budgets=dict(result.eager_budgets),
                      actual_allocated_draft_tokens=result.allocated_draft_tokens,
                      actual_verification_roof=result.verification_roof)
        assert record['actual_typed_eager_budgets'] == {0: 5}
        records.append(record)
    keys = sorted({n.slice.value for n in ast.walk(shape)
                   if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                   and n.value.id == 'kwargs' and isinstance(n.slice, ast.Constant)})
    source = ast.get_source_segment((PARENT/'tree_entry.py').read_text(), shape)
    assert 'eager_token_cap' not in source and 'eager_reserve_tokens' not in source
    after = {str(p): sha(p) for p in paths}
    assert before == after and 'torch' not in sys.modules
    report = dict(schema='ROOT.original-eager-budget-parameters-HOST-counterexample.v1',
        passed=True, actual_HOST_argv=sys.argv, source_hashes_before=before, source_hashes_after=after,
        actual_original_shape_executed=True, actual_original_policy_executed=True,
        actual_original_typed_plan_invariants_executed=True, original_kwargs_subscripts=keys,
        counterexamples=records, note='Synthetic eligible state proves ignored parameter flow. '
        'It does not measure actual online eligibility, grant counts, device cost or Goodput.',
        Torch_imported=False, NPU_started=False, model_initialized=False, production_installed=False,
        actual_runtime_eligibility_observed=False, formal_Goodput=False)
    output = BASE/'budget-diagnostics/ORIGINAL_EAGER_BUDGET_PARAMETERS_HOST_COUNTEREXAMPLE.json'
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(passed=True, output=str(output), sha256=sha(output), cases=records)))


if __name__ == '__main__':
    main()
