"""Actual allocator: add branch nodes only after preserving normal B4 space."""
import json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
def main():
    from check_cost290_HOST import load,N
    for name in('vllm_ascend','vllm_ascend.spec_decode','vllm_ascend.spec_decode.pearl'):
        module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
    load('vllm_ascend.spec_decode.pearl.roofline',N/'roofline.py')
    rhythm=load('vllm_ascend.spec_decode.pearl.spec_rhythm',N/'spec_rhythm.py')
    for folder in('tree-spare-branch-group-budget-full-v260','tree-high-normal-cap4-native-families-v148'):
        sys.path.append(str(B/folder))
    from budget_policy import TreeBudgetPolicy
    from normal_admission_v140 import reset
    from budget290 import prepared_shape,eligible
    model=json.loads((B/'tree-high-normal-cap4-native-families-v148/tree-gain-model.json').read_text())
    cases=[]
    for n in range(1,33):
        policy=TreeBudgetPolicy(model);reset(policy,1)
        for cls in('tight','normal','loose'):
            for b,g in enumerate([1.,2.,2.8,3.1,3.2,3.5,3.7,4.2],1):policy.observed[cls,b]=(1000000,round(1000000*g))
        states={i:rhythm.SpecRhythmRuntimeState(i,i%2,slo_tpot_ms=40 if i<12 else 150,
            slo_class='tight'if i<12 else'loose',max_gamma=8,output_token_budget=256,
            delivered_tokens=1,decode_elapsed_ms=140 if i<12 else 0)for i in range(n)}
        candidates=eligible(policy,states,tuple(states))
        assert len(candidates)==min(n,max(0,min(12,24-n)))
        assert candidates==frozenset(range(len(candidates)))
        plan=prepared_shape()(policy,types.SimpleNamespace(min_gamma=1,max_gamma=8),plan_id=1,
            normal_request_indices=tuple(states),eager_request_indices=(),states=states,
            projected_wait_ms=10.,context_len=127,draft_token_budget=192,batch_size=n,
            eager_token_cap=3,eager_reserve_tokens=0,verification_roof=96)
        budgets=dict(plan.normal_budgets)
        assert sum(budgets.values())<=96 and plan.allocated_selected_candidates<=192
        assert all(v<=8 if i in candidates else v<=4 for i,v in budgets.items())
        if n<=24:
            assert set(budgets)==set(states),plan
            assert budgets=={i:8 if i in candidates else 4 for i in states},budgets
            assert n+sum(v>4 for v in budgets.values())<=24
        if n==24:assert budgets==dict.fromkeys(states,4)
        cases.append(dict(N=n,branch_eligible=len(candidates),normal_grants=budgets,
            deferred=list(plan.deferred_normal_request_indices),selected_candidates=sum(budgets.values())))
    assert not any(m.split('.')[0]in('torch','torch_npu')for m in sys.modules)
    result=dict(status='HOST_actual_typed_allocator_preserves_B4_normal_capacity_before_B8_upgrades_passed',
        cases=cases,case_count=32,N24_all24_B4_instead_of12_B8=True,
        original_priority_original_bound_repair_and_C96_global192_unchanged=True,
        branch_nodes_only_use_spare_candidate_and_group_capacity=True,
        cost_ROI_gate_still_required=True,fixture_gain_not_empirical=True,NPU_used=False,Goodput=None,runtime_installed=False)
    with(P/'HOST_CAPACITY290_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='cases'}))
if __name__=='__main__':main()
