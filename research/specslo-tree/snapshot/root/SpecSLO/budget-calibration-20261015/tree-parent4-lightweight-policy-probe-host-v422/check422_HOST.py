"""Actual original g/need allocator; no Torch, model or serving installation."""
import ast
import copy
import json
from pathlib import Path
import random
import sys

P=Path(__file__).resolve().parent
B=P.parent
BASE=B/'tree-high-normal-cap4-native-families-v148'


def main():
    sys.path.insert(0,str(B/'tree-frugal-S32-paid-budget-control-host-v401'))
    from check401_HOST import original_control_sources,state_set,BACKENDS
    rhythm=original_control_sources()
    sys.path[:0]=[str(B/'tree-parent4-selective-paid-allocation-host-v420'),str(BASE),str(B/'tree-normal24-derived-callback-repair-v405'),
                   str(B/'tree-parent4-normal24-SLO-mixed-protocol-v419'),
                   str(B/'tree-bounded-parent4-shared4-eager-diagnostic-v199')]
    import shape422 as shape420
    import admission405
    import budget_policy
    import split_shape_v132
    import split_budget_v132
    import budget419
    before_class=split_budget_v132.DraftWorkLimit
    # This is exactly the retained405 projected original allocator, without
    # constructing a Native model or modifying the original module/class.
    text=(BASE/'split_shape_v132.py').read_text()
    node=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef) and n.name=='shape')
    source=ast.get_source_segment(text,node).replace('if len(raw_normal)>32:','if len(raw_normal)>24:')
    namespace=dict(vars(split_shape_v132),select_normal=admission405.select,finish_normal=admission405.finish)
    exec(compile(source,'HOST422.actual-normal24-original-allocator','exec'),namespace)
    original=namespace['shape']
    model=json.loads((BASE/'tree-gain-model.json').read_text())
    rng=random.Random(420)
    proposed=fallbacks=0
    for trial in range(800):
        n=64
        states=state_set(rhythm,n,rng)
        for state in states.values():state.max_gamma=5
        normal_count=24 if trial%2==0 else rng.randrange(1,49)
        eager=tuple(range(48,48+(4 if trial%3 else rng.randrange(1,9))))
        normal=tuple(range(min(normal_count,48)))
        policy=budget_policy.TreeBudgetPolicy(model)
        shaper=rhythm.SpecRhythmBudgetShaper(min_gamma=1,max_gamma=8,verification_budget=96)
        admission405.reset(policy,1)
        unchanged=copy.deepcopy(states)
        original_policy=copy.deepcopy(policy)
        parents={i:4 if trial%5 else 3 for i in eager}
        kwargs=dict(plan_id=trial,normal_request_indices=normal,eager_request_indices=eager,
            states=states,projected_wait_ms=20.,context_len=1024,batch_size=64,
            draft_token_budget=192,eager_token_cap=3,eager_reserve_tokens=32,verification_roof=96)
        result=shape420.propose(policy,shaper,original,kwargs,parent_budgets=parents)
        record=admission405.snapshot(policy)
        assert record['call_count']==1 and states==unchanged
        assert policy.observed==original_policy.observed
        assert len(result.plan.normal_budgets)<=24
        assert sum(result.plan.normal_budgets.values())<=96
        assert sum(result.plan.eager_budgets.values())<=96
        assert result.plan.allocated_selected_candidates<=192
        if result.future4_owners:
            proposed+=1
            assert len(result.plan.normal_budgets)==24 and len(result.future4_owners)==4
            assert all(b==4 for b in (*result.plan.normal_budgets.values(),*result.plan.eager_budgets.values()))
            assert result.plan.allocated_selected_candidates==112
            assert all(parents[i]==4 for i in result.future4_owners)
        else:
            expected=original(original_policy,shaper,**dict(kwargs,
                eager_request_indices=tuple(i for i in eager if parents[i]==3)))
            assert result.plan==expected and policy.counts==original_policy.counts
            assert record['ages']==admission405.snapshot(original_policy)['ages']
            fallbacks+=1
    # Copy cost is measured on retained405-style growing HOST history only.
    # These CPU timings do not measure NPU kernels or serving performance.
    import hashlib,time,statistics
    donor=B/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual/normal24-global64-derived-controls-root-v405-full463-ownAR-v1/rank0-local-diagnostic.json'
    saved=json.loads(donor.read_text())['ROOT405_normal_admission']['normal_admission']['calls']
    policy._HOST422_history=saved
    encoded=json.dumps(saved,sort_keys=True).encode()
    unchanged_hash=hashlib.sha256(encoded).hexdigest()
    from shape422 import probe_policy,descriptor
    sample=probe_policy(policy)
    assert sample is not policy and sample.counts is not policy.counts
    assert sample._HOST422_history is policy._HOST422_history
    assert sample.base is policy.base and sample.observed is policy.observed
    sample.counts['HOST422-only-probe']+=1
    assert 'HOST422-only-probe' not in policy.counts
    costs={'deepcopy':[],'light':[]}
    for j in range(12):
        for kind in (('deepcopy','light') if j%2==0 else ('light','deepcopy')):
            start=time.perf_counter_ns()
            value=copy.deepcopy(policy) if kind=='deepcopy' else probe_policy(policy)
            costs[kind].append((time.perf_counter_ns()-start)/1e6)
    assert hashlib.sha256(json.dumps(policy._HOST422_history,sort_keys=True).encode()).hexdigest()==unchanged_hash
    copy_cost=dict(retained_history_rows=len(saved),retained_history_json_bytes=len(encoded),
                   median_CPU_ms={k:statistics.median(v) for k,v in costs.items()},
                   no_history_mutation=True,scope='HOST copy primitive only; no NPU or serving latency')
    assert proposed and fallbacks
    assert split_budget_v132.DraftWorkLimit is before_class
    assert before_class().eager_per_request==3
    assert not any(name.split('.')[0] in BACKENDS for name in sys.modules)
    value=dict(status='PASS_HOST422_ORIGINAL_G_NEED_PRIORITY_EXACT_PAID_N24_E4_PROPOSALS',
        copy_cost=copy_cost,source_projection=descriptor(),HOST_cases=800,supported_proposals=proposed,unchanged_original_fallbacks=fallbacks,
        same_original_g_and_online_observations=True,one_normal_age_finish_per_call=True,
        readonly_probe_does_not_change_real_policy=True,private_E4_class_only=True,
        public_production_cap3_class_unchanged=True,global_active_limit_required=64,
        normal_candidates=96,future_candidates=16,total_candidates=112,
        NPU_used=False,serving_installed=False,production_permission=False,Raw=None,Goodput=None)
    (P/'HOST422_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
