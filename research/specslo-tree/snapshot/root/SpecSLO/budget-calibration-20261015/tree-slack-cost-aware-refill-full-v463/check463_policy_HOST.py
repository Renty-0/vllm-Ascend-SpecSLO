"""Exercise real admission projections and CPU policy boundaries; no NN grant."""

from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import hashlib
import importlib.abc
import json
import math
import sys

P = Path(__file__).resolve().parent
B = P.parent


class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'acl'):
            raise RuntimeError('HOST policy tests cannot import device/model backends')


sys.meta_path.insert(0, NoModel())
sys.path.insert(0, str(B / 'tree-adaptive-own-history-full-v460'))
from host_env460 import setup
setup()
sys.path.insert(0, str(P))
import admission463 as policy
import generation463
import generation438
import witness463
from projection431 import patch as refill_patch


def fixture(old_count, new_count, *, slo=40.0, delivered=21, elapsed=560.0, queries=100):
    active = list(range(old_count))
    new = list(range(old_count, old_count + new_count))
    params = [SimpleNamespace(slo_tpot_ms=slo, arrival_ts=0.0) for _ in range(old_count + new_count)]
    states = [SimpleNamespace(delivered_tokens=delivered, decode_elapsed_ms=elapsed,
                              decode_start_elapsed_ms=0.0) for _ in params]
    local = [SimpleNamespace(prompt_length=queries) for _ in params]
    engine = SimpleNamespace(_dynamic_tree_generation=7)
    return engine, new, dict(active=active, request_params=params, states=states,
                             local_states=local, prefetched=set(), now=1.0, counters={})


policy.validate_model()
checks = 0
for old_count in (1, 4, 8, 16, 32):
    for slo in (40.0, 50.0, 150.0):
        for delivered in (1, 2, 20, 50, 100, 220):
            for elapsed_ratio in (0.0, 0.25, 0.75, 1.20):
                available_before_elapsed = policy.CREDIT_FRACTION * slo * (delivered - 1)
                elapsed = elapsed_ratio * available_before_elapsed
                for queries in (1, 86, 512, 2000):
                    for new_count in (1, 8, 16, 32):
                        engine, new, args = fixture(old_count, new_count, slo=slo,
                            delivered=delivered, elapsed=elapsed, queries=queries)
                        before = copy.deepcopy((args['request_params'], args['states'], args['local_states']))
                        selected = policy.choose(engine, new, **args)
                        available = max(0.0, available_before_elapsed - elapsed)
                        # Independent closed form for equal-length prompts,
                        # including rounding at the minimum-eight boundary.
                        bound = math.floor((available / policy.COST_MARGIN - policy.INTERCEPT_MS)
                                           / (policy.SLOPE_MS * queries))
                        expected = min(new_count, max(0, bound))
                        if expected < 8:
                            expected = 0
                        assert selected == new[:expected]
                        assert new == list(range(old_count, old_count + new_count))
                        assert [vars(v) for v in args['request_params']] == [vars(v) for v in before[0]]
                        assert [vars(v) for v in args['states']] == [vars(v) for v in before[1]]
                        assert [vars(v) for v in args['local_states']] == [vars(v) for v in before[2]]
                        if selected:
                            decision = engine._root463_pending_admission
                            assert decision['predicted_prefill_ms'] <= decision['available_pause_ms']
                            assert all(t <= args['now'] for t in decision['candidate_arrivals'])
                        checks += 1

# The new-cohort ceiling is not a reduced global ceiling: a loose workload
# with sufficient incumbent credit can select32 additional owners beside32.
engine, new, args = fixture(32, 32, slo=150.0, delivered=100, elapsed=1000.0)
assert len(policy.choose(engine, new, **args)) + len(args['active']) == 64
engine, new, args = fixture(0, 64)
assert policy.choose(engine, new, **args) == new[:32]
engine, new, args = fixture(16, 32)
args['active'] = list(reversed(args['active']))
new = new[::2] + new[1::2]
assert policy.choose(engine, new, **args) == new[:10]
engine, new, args = fixture(16, 32, delivered=1, elapsed=0.0)
assert policy.choose(engine, new, **args) == []

rejected = []
for kind in ('future', 'duplicate', 'nonfinite_elapsed', 'no_delivered_root', 'wrong_generation'):
    engine, new, args = fixture(16, 32)
    if kind == 'future':
        args['request_params'][new[0]].arrival_ts = 2.0
    elif kind == 'duplicate':
        new[1] = new[0]
    elif kind == 'nonfinite_elapsed':
        args['states'][0].decode_elapsed_ms = float('nan')
    elif kind == 'no_delivered_root':
        args['states'][0].delivered_tokens = 0
    else:
        engine._dynamic_tree_generation = 0
    try:
        policy.choose(engine, new, **args)
    except (RuntimeError, ValueError):
        rejected.append(kind)
    else:
        raise AssertionError('Invalid host admission accepted: ' + kind)

try:
    witness463.before(SimpleNamespace(), [], [], [], [], 1.0, [], SimpleNamespace())
except RuntimeError:
    rejected.append('HOST_cannot_supply_same_model_NN_permission')
else:
    raise AssertionError('HOST supplied the original same-model NN permission')

original_source = B / 'tree-parent4-canonical-tail-observer-binding-v425/HOST425.generated-LIVE.py'
base = generation438.project(refill_patch(original_source.read_text()))
changed = generation463.project(base)
base_tree, changed_tree = ast.parse(base), ast.parse(changed)
clock_names = {'time.time', 'time.perf_counter', '_dt_cpu_broadcast'}
def clocks(tree):
    return [ast.unparse(n.func) for n in ast.walk(tree)
            if isinstance(n, ast.Call) and ast.unparse(n.func) in clock_names]
assert clocks(base_tree) == clocks(changed_tree)
admit = next(n for n in ast.walk(changed_tree) if isinstance(n, ast.FunctionDef) and n.name == 'admit_available')
selection_index = next(i for i, n in enumerate(admit.body) if isinstance(n, ast.Assign)
                       and ast.unparse(n.value).startswith('_choose463('))
home_index = next(i for i, n in enumerate(admit.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'homes')
assert selection_index < home_index
assert not any(name.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend', 'acl') for name in sys.modules)
result = dict(status='PASS_HOST463_CREDIT_COST_PREFIX_POLICY_AND_WHOLE_LIVE_PROJECTION',
    closed_form_cases=checks, rejected_controls=rejected, whole_LIVE_AST_inverse=True,
    original_arrival_prefill_broadcast_clocks_unchanged=True,
    selector_precedes_home_cache_or_pending_mutations=True,
    preserves_original_priority_prefix=True, constrained_cohort_maximum=32,
    global64_reachable_for_sufficient_loose_credit=True,
    fixed_model=policy.validate_model(), model_imports=0, NPU_used=False,
    NN_permission=False, full463=False, Raw=None, Goodput=None,
    sources={str(p): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in (P/'admission463.py', P/'generation463.py', P/'witness463.py', Path(__file__).resolve())})
with (P/'HOST463_POLICY_RESULT.json').open('x') as stream:
    json.dump(result, stream, indent=2)
    stream.write('\n')
print(json.dumps(result))
