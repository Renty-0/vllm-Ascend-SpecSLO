"""Actual typed allocation/controller: positive B8, priority and hard budgets."""
import dataclasses,json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
def main():
    from check_cost284_HOST import load,N
    for name in('vllm_ascend','vllm_ascend.spec_decode','vllm_ascend.spec_decode.pearl'):
        module=types.ModuleType(name);module.__path__=[];sys.modules[name]=module
    load('vllm_ascend.spec_decode.pearl.roofline',N/'roofline.py')
    rhythm=load('vllm_ascend.spec_decode.pearl.spec_rhythm',N/'spec_rhythm.py')
    for folder in('tree-spare-branch-group-budget-full-v260','tree-high-normal-cap4-native-families-v148'):
        sys.path.append(str(B/folder))
    import split_budget_v132 as original
    from budget_policy import TreeBudgetPolicy
    from normal_admission_v140 import reset
    from budget284 import prepared_shape,work_limit_class,build_plan,filter_eager_parents
    limit=work_limit_class()();old=dataclasses.asdict(original.DraftWorkLimit());new=dataclasses.asdict(limit)
    assert new==dict(old,normal_per_request=8)and original.DraftWorkLimit().normal_per_request==5
    try:work_limit_class()(normal_per_request=9)
    except ValueError:pass
    else:raise AssertionError('cap9 accepted')
    model=json.loads((B/'tree-high-normal-cap4-native-families-v148/tree-gain-model.json').read_text())
    results=[]
    for n in(12,24,32):
        policy=TreeBudgetPolicy(model);reset(policy,1)
        for cls in('tight','normal','loose'):
            for b,g in enumerate([1.,2.,2.8,3.1,3.2,3.5,3.7,4.2],1):policy.observed[cls,b]=(1000000,round(1000000*g))
        urgent=min(12,n)
        states={i:rhythm.SpecRhythmRuntimeState(i,i%2,slo_tpot_ms=40 if i<urgent else 150,
            slo_class='tight'if i<urgent else'loose',max_gamma=8,output_token_budget=256,
            delivered_tokens=1,decode_elapsed_ms=140 if i<urgent else 0)for i in range(n)}
        shaper=types.SimpleNamespace(min_gamma=1,max_gamma=8)
        plan=prepared_shape()(policy,shaper,plan_id=1,normal_request_indices=tuple(states),
            eager_request_indices=(),states=states,projected_wait_ms=10.,context_len=127,
            draft_token_budget=192,batch_size=n,eager_token_cap=3,eager_reserve_tokens=0,verification_roof=96)
        assert dict(plan.normal_budgets)==dict.fromkeys(range(12),8)
        assert sum(plan.normal_budgets.values())==96 and plan.allocated_selected_candidates<=192
        assert set(plan.deferred_normal_request_indices)==set(range(12,n))
        results.append(dict(N=n,positive_B8_grants=12,selected_candidates=96,
            low_priority_deferred=list(plan.deferred_normal_request_indices)))
    controller_cases=[]
    for n in(12,16,20,24,32):
        for forks in(1,min(12,n)):
            for b in(5,6,7,8):
                states={i:rhythm.SpecRhythmRuntimeState(i,i%2,slo_tpot_ms=40 if i<forks else 150,
                    slo_class='tight'if i<forks else'loose',max_gamma=8,
                    output_token_budget=256,delivered_tokens=1,decode_elapsed_ms=10 if i<forks else 1000)
                    for i in range(n)}
                controller=rhythm.SpecRhythmPipelineController(states)
                controller.publish([controller.new_ticket(i,gamma=b if i<forks else 4,eager=False)for i in states])
                before={i:(t.proposal_id,t.gamma,t.required_prefix_epoch)for i,t in controller.ready.items()}
                result=build_plan(controller,list(states),active_indices_registered_unique=True,priority=True,
                    target_slo_first=True,merge_ready_homes=True,max_target_requests=32,verification_budget=96)
                values=result.target_candidate_budgets
                assert len(values)+sum(v>4 for v in values.values())<=24 and sum(values.values())<=96
                assert set(result.target_request_indices[:min(forks,len(values))])==set(range(min(forks,len(values))))
                assert {i:(t.proposal_id,t.gamma,t.required_prefix_epoch)for i,t in controller.ready.items()}==before
                assert set(values)|set(result.deferred_target_request_indices)==set(states)
                controller_cases.append(dict(N=n,branch_B=b,forks=forks,selected=len(values),
                    actual_candidates=sum(values.values()),actual_groups=len(values)+sum(v>4 for v in values.values())))
    controller=types.SimpleNamespace(ready={i:types.SimpleNamespace(proposal_id=i)for i in range(8)})
    payloads={i:dict(plan=types.SimpleNamespace(candidate_budget=i+1))for i in range(8)};counters={}
    assert filter_eager_parents(range(8),controller,payloads,counters)==[2]
    assert all(counters[f'spec_rhythm_eager_parent_B{b}_excluded_unqualified']==1 for b in(1,2,4,5,6,7,8))
    assert not any(m.split('.')[0]in('torch','torch_npu')for m in sys.modules)
    out=dict(status='HOST_positive_B8_priority_and_whole_ticket_caps_passed',
        positive_shapes=results,actual_controller_cases=controller_cases,
        explicit_private_normal_perowner_cap8=True,original_DTO_unchanged=True,
        C96_future96_global192_unchanged=True,actual_target_groups24_preserved=True,
        ready_ticket_identity_and_publication_unchanged=True,
        eager_parent_B1to8_recognized_only_B3_eligible=True,
        HOST_fixture_g_not_empirical=True,NPU_used=False,Goodput=None,adopted=False)
    with(P/'HOST_BUDGET284_RESULT.json').open('x')as f:json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items()if k not in('positive_shapes','actual_controller_cases')}))
if __name__=='__main__':main()
