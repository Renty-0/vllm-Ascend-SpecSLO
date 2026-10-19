"""Exercise the actual bound repair and actual typed allocation, without NPU."""
import importlib.util
import json
import sys
import types
from pathlib import Path

P = Path(__file__).resolve().parent
B = P.parent
import check316
N = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    for name in ('vllm_ascend', 'vllm_ascend.spec_decode', 'vllm_ascend.spec_decode.pearl'):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules[name] = m
    load('vllm_ascend.spec_decode.pearl.roofline', N/'roofline.py')
    rhythm = load('vllm_ascend.spec_decode.pearl.spec_rhythm', N/'spec_rhythm.py')
    for f in ('tree-root-alternate-live-policy-host-v316','tree-spare-branch-group-budget-full-v260',
              'tree-packed-KV-preflight-rank-aware-own-AR-full-v218',
              'tree-causal80-real16-bankfix-full-v173',
              'tree-high-normal-cap4-native-families-v148'):
        sys.path.append(str(B/f))
    from collections import Counter
    from budget_policy import TreeBudgetPolicy
    from normal_admission_v140 import reset
    import split_shape_v132
    import cost218
    from budget316 import shape, prepared_bound_shape
    model = json.loads((B/'tree-high-normal-cap4-native-families-v148/tree-gain-model.json').read_text())
    cases = []
    repaired = []
    for n in range(1, 33):
        for available in (0, 24, 64, 96, 192):
            policy = TreeBudgetPolicy(model)
            reset(policy, 1)
            cache_state = dict(private_shape_bindings=0, by_generation={},
                original_cost_table=[dict(N=i, Q=q, M=m, S=s)
                    for (i,q),(m,s) in sorted(cost218.original_cost_table().items())])
            ns = dict(shape=split_shape_v132.shape)
            exec('def factory(policy):\n def candidate_shape(instance,**kwargs):return shape(policy,instance,**kwargs)\n return candidate_shape', ns)
            control = types.SimpleNamespace(candidate_shape=ns['factory'](policy),
                    engine=types.SimpleNamespace(_dynamic_tree_generation=1))
            original = cost218.bind_candidate_shape(control, cache_state)
            class Shaper:
                min_gamma = 1
                max_gamma = 8
            Shaper.shape = original
            shaper = Shaper()
            states = {i: rhythm.SpecRhythmRuntimeState(i, i%2,
                slo_tpot_ms=40 if i < (n+1)//2 else 50,
                slo_class='tight' if i < (n+1)//2 else 'normal',
                max_gamma=8, output_token_budget=256, delivered_tokens=1,
                decode_elapsed_ms=100) for i in range(n)}
            # Deliberately expose a mixed B3/B4 plateau. The real ROOT218
            # solver must repair it; a counter-only substitute cannot pass.
            policy.observed.update({('tight',3):(10000,30000),
                ('tight',4):(10000,28000),('tight',5):(10000,27000),
                ('normal',4):(10000,40000),('normal',5):(10000,35000),
                **{(c,j):(10000,20000)for c in('tight','normal')for j in(6,7,8)}})
            kwargs = dict(plan_id=0, normal_request_indices=tuple(states),
                eager_request_indices=(), states=states, projected_wait_ms=10.,
                context_len=127, draft_token_budget=available, batch_size=n,
                eager_token_cap=3, eager_reserve_tokens=0, verification_roof=96)
            first = shape(shaper, **kwargs)
            assert type(shaper).shape is original and control.candidate_shape is original
            assert all(1<=v<=8 for v in first.normal_budgets.values())
            assert sum(first.normal_budgets.values()) <= min(96,available)
            assert cache_state['by_generation']['1']['shape_calls'] == 1
            control.engine._dynamic_tree_generation = 2
            second = shape(shaper, **dict(kwargs, plan_id=1))
            assert cache_state['by_generation']['2']['shape_calls'] == 1
            assert cost218.audit_snapshot({'cost_aware_N9to16_budget':cache_state}, 2)
            solver = prepared_bound_shape(original).__globals__['shape_draft_work']
            prior = original.__globals__['shape'].__globals__['shape_draft_work']
            assert solver.__code__ is prior.__code__ and solver.__globals__ is prior.__globals__
            for name in ('control','state'):
                pos = prior.__code__.co_freevars.index(name)
                assert solver.__closure__[pos] is prior.__closure__[pos]
            if cache_state['by_generation']['1'].get('repaired_calls'):
                assert set(first.normal_budgets.values()) == {4}
                repaired.append(dict(N=n, available=available,
                    actual_generation1=dict(cache_state['by_generation']['1']),
                    actual_generation2=dict(cache_state['by_generation']['2'])))
            cases.append(dict(N=n, available=available,
                              first=list(first.normal_budgets.values()), second=list(second.normal_budgets.values())))
    assert repaired and any(r['N']==12 and r['available']==192 for r in repaired)
    assert not any(m.split('.')[0] in ('torch','torch_npu') for m in sys.modules)
    result = dict(status='HOST_actual_bound_repair_with_private_cap8_DTO_passed',
                  cases=cases, case_count=len(cases), actual_repaired_cases=repaired,
                  original_control_state_cells_preserved=True,
                  actual_repair_code_and_globals_preserved=True,
                  all_per_generation_audits_passed=True,
                  original_installed_class_shape_identity_preserved=True,
                  NPU_used=False, neural_qualified=False, Goodput=None)
    with (P/'HOST_COST316_RESULT.json').open('x') as f:
        json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','actual_repaired_cases')}))


if __name__=='__main__':
    main()
