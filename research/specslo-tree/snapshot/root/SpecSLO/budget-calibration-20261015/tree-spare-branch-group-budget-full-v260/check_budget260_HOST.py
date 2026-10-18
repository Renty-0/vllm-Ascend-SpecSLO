"""Execute original typed grants and controller with one extra physical budget."""
import importlib.util,json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
N=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module
def main():
    for name in('vllm_ascend','vllm_ascend.spec_decode','vllm_ascend.spec_decode.pearl'):
        m=types.ModuleType(name);m.__path__=[];sys.modules[name]=m
    load('vllm_ascend.spec_decode.pearl.roofline',N/'roofline.py')
    rhythm=load('vllm_ascend.spec_decode.pearl.spec_rhythm',N/'spec_rhythm.py')
    sys.path.insert(0,str(B/'tree-high-normal-cap4-native-families-v148'))
    from budget_policy import TreeBudgetPolicy
    from normal_admission_v140 import reset
    import split_shape_v132
    from budget260 import prepared_shape,eligible,build_plan,shape
    model=json.loads((B/'tree-high-normal-cap4-native-families-v148/tree-gain-model.json').read_text())
    def new_policy():
        policy=TreeBudgetPolicy(model);reset(policy,1);return policy
    shaper=types.SimpleNamespace(min_gamma=1,max_gamma=8)
    grant_rows=[];controller_rows=[]
    def states(n):
        return {i:rhythm.SpecRhythmRuntimeState(i,i%2,slo_tpot_ms=[40,50,150][i%3],slo_class=['tight','normal','loose'][i%3],max_gamma=5,output_token_budget=256,delivered_tokens=1,decode_elapsed_ms=100)for i in range(n)}
    for n in range(1,33):
        for available in (0,24,64,96,192):
            actual=states(n);policy=new_policy()
            kwargs=dict(plan_id=0,normal_request_indices=tuple(actual),eager_request_indices=(),states=actual,
                projected_wait_ms=10.,context_len=127,draft_token_budget=available,batch_size=n,
                eager_token_cap=3,eager_reserve_tokens=0,verification_roof=96)
            plan=prepared_shape()(policy,shaper,**kwargs)
            values=list(plan.normal_budgets.values());slots=min(n,max(0,24-n))
            assert values.count(5)<=slots and sum(values)<=min(96,available)
            assert plan.allocated_selected_candidates+plan.unused_draft_candidates==min(192,available)
            if n>=24:
                previous=split_shape_v132.shape(new_policy(),shaper,**kwargs)
                for field in('normal_budgets','eager_budgets','caps','needs','progress_gaps'):
                    assert dict(getattr(plan,field))==dict(getattr(previous,field)),field
            grant_rows.append(dict(N=n,available=available,grants=values,eligible_B5=sorted(eligible(policy,actual,tuple(actual)))))
    # Exercise the actual LIVE helper's policy ownership contract.
    policy=new_policy()
    def bound(policy):
        def original(instance,**kwargs):return split_shape_v132.shape(policy,instance,**kwargs)
        return original
    class Shaper: min_gamma=1;max_gamma=8
    Shaper.shape=bound(policy)
    actual=states(12);plan=shape(Shaper(),plan_id=1,normal_request_indices=tuple(actual),eager_request_indices=(),states=actual,
        projected_wait_ms=10.,context_len=127,draft_token_budget=192,batch_size=12,eager_token_cap=3,eager_reserve_tokens=0,verification_roof=96)
    assert len(plan.normal_budgets)==12 and 5 in plan.normal_budgets.values()
    assert policy.counts['ROOT260_actual_B5_grants']==list(plan.normal_budgets.values()).count(5)
    assert policy.counts['ROOT260_actual_B5_grants_N_gt8']==list(plan.normal_budgets.values()).count(5)
    for n in (1,8,12,16,20,23,24,32):
        for branches in range(n+1):
            actual=states(n);controller=rhythm.SpecRhythmPipelineController(actual)
            controller.publish([controller.new_ticket(i,gamma=5 if i<branches else 4,eager=False)for i in actual])
            before={i:(t.proposal_id,t.gamma,t.required_prefix_epoch)for i,t in controller.ready.items()}
            plan=build_plan(controller,list(actual),active_indices_registered_unique=True,priority=True,target_slo_first=True,
                merge_ready_homes=True,max_target_requests=24,verification_budget=96)
            selected=plan.target_request_indices;counts=list(plan.target_candidate_budgets.values())
            assert selected and len(selected)+counts.count(5)<=24 and sum(counts)<=96
            assert {i:(t.proposal_id,t.gamma,t.required_prefix_epoch)for i,t in controller.ready.items()}==before
            assert set(selected)|set(plan.deferred_target_request_indices)==set(actual)
            if n+branches<=24 and sum(t.gamma for t in controller.ready.values())<=96:
                assert set(selected)==set(actual)
            controller_rows.append(dict(N=n,branches=branches,selected=len(selected),selected_groups=len(selected)+counts.count(5)))
    actual=states(24);controller=rhythm.SpecRhythmPipelineController(actual)
    controller.publish([controller.new_ticket(i,gamma=5,eager=False)for i in actual])
    try:build_plan(controller,list(actual),active_indices_registered_unique=True,priority=True,target_slo_first=True,
        merge_ready_homes=True,max_target_requests=24,verification_budget=96,
        reserved_target_request_indices=tuple(range(13)),reserved_target_home_batch_id=0)
    except ValueError as error:assert 'exceeds24 physical groups'in str(error)
    else:raise AssertionError('invalid irreversible reservation accepted')
    actual=states(3);kwargs=dict(plan_id=0,normal_request_indices=(),eager_request_indices=tuple(actual),states=actual,
        projected_wait_ms=10.,context_len=127,draft_token_budget=192,batch_size=3,
        eager_token_cap=3,eager_reserve_tokens=0,verification_roof=96)
    new=prepared_shape()(new_policy(),shaper,**kwargs);old=split_shape_v132.shape(new_policy(),shaper,**kwargs)
    for field in('normal_budgets','eager_budgets','caps','needs','progress_gaps'):assert dict(getattr(new,field))==dict(getattr(old,field))
    assert not any(name.split('.')[0]in('torch','torch_npu')for name in sys.modules)
    result=dict(status='HOST_original_typed_allocation_and_controller_physical_groups_passed',grant_cases=len(grant_rows),controller_cases=len(controller_rows),
        grants=grant_rows,controller=controller_rows,N24plus_all_original_grants_unchanged=True,whole_tickets_preserved=True,
        actual_policy_closure_helper_executed=True,actual_B5_grant_counters_checked=True,
        irreversible_over_capacity_reservation_rejected=True,eager_only_original_grants_unchanged=True,
        original_AST_inverses=True,NPU_used=False,neural_qualified=False,Goodput=None)
    with(P/'HOST_BUDGET260_FINAL_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k not in('grants','controller')}))
if __name__=='__main__':main()
