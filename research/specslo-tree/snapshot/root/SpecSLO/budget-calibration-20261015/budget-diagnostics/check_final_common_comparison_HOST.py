"""Reject incomplete/mismatched comparisons; never launch a model."""
import copy,json
from pathlib import Path
from final_common_comparison import aggregate,ORDERS,EXPECTED_WORKLOAD

def fixture():
    scores=dict(fixed_gamma4=[500,510,490],qualified_dynamic_tree=[650,660,640],official_MC2_ON=[420,430,410])
    cells=[]
    for block,order in enumerate(ORDERS,1):
        for position,method in enumerate(order):
            validation=dict(source_and_environment_verified=True,physical_cards=[2,5,6,7],
                            all_owned_workers_exited=True,all4_idle_after=True)
            if method=='qualified_dynamic_tree':validation.update(all_original_and_new_gates_passed=True,
                whole_workload_AR_qualified=True,AR_equal_requests=463,AR_different_requests=0,candidate_freeze_sha256='a'*64)
            if method=='official_MC2_ON':validation.update(ordinary_MC2_actually_enabled=True,MC2_projection_modules_by_rank=[128]*4)
            cells.append(dict(block=block,position=position,method=method,tag=f'SYNTHETIC-HOST-{block}-{method}',validation=validation,
                metrics=dict(comparison_method=method,request_identity=dict(sha256=EXPECTED_WORKLOAD),
                             all=dict(requests=463,output_tokens=118528,Goodput=scores[method][block-1],Raw=1000,attained=463),
                             by_class={k:dict(requests=v)for k,v in dict(tight=271,normal=96,loose=96).items()},
                             source_descriptors=[dict(path=f'/SYNTHETIC-HOST-ONLY/{block}/{method}')])))
    return cells

passed=[]
x=fixture();r=aggregate(x)
assert r['stated_performance_and_full_AR_goals_met']and r['method_summaries']['qualified_dynamic_tree']['Goodput_median']==650
assert r['method_summaries']['qualified_dynamic_tree']['Goodput_all']==[650,660,640]
passed.append('all9_preserved_and_method_median_no_fastest_selection')
x=fixture()
for c in x:
    if c['method']=='official_MC2_ON':c['metrics']['all']['Goodput']+=30
assert not aggregate(x)['stated_performance_and_full_AR_goals_met'];passed.append('stronger_current_MC2_not_replaced_by_lower_history')
x=fixture()
for c in x:
    if c['method']=='fixed_gamma4':c['metrics']['all']['Goodput']+=160
assert not aggregate(x)['stated_performance_and_full_AR_goals_met'];passed.append('stronger_current_fixed_gamma4_required')

def rejected(name,mutate):
    cells=fixture();mutate(cells)
    try:aggregate(cells)
    except ValueError:passed.append(name)
    else:raise AssertionError(name+' was accepted')

rejected('partial8cells',lambda c:c.pop())
rejected('changed_order',lambda c:c.__setitem__(slice(0,2),[c[1],c[0]]))
rejected('reused_process_tag',lambda c:c[3].update(tag=c[0]['tag']))
rejected('reused_result_artifact',lambda c:c[3]['metrics'].update(source_descriptors=c[0]['metrics']['source_descriptors']))
rejected('changed_Poisson_or_prompt_identity',lambda c:c[0]['metrics']['request_identity'].update(sha256='b'*64))
rejected('dropped_outputs',lambda c:c[0]['metrics']['all'].update(output_tokens=118272))
rejected('incomplete_class',lambda c:c[0]['metrics']['by_class']['tight'].update(requests=270))
rejected('tree_AR_one_mismatch',lambda c:c[1]['validation'].update(AR_equal_requests=462,AR_different_requests=1))
rejected('MC2_off',lambda c:c[2]['validation'].update(ordinary_MC2_actually_enabled=False))
rejected('MC2_not_all_ranks',lambda c:c[2]['validation'].update(MC2_projection_modules_by_rank=[128,128,127,128]))
rejected('mixed_tree_source_versions',lambda c:c[4]['validation'].update(candidate_freeze_sha256='c'*64))
rejected('owned_workers_left',lambda c:c[0]['validation'].update(all_owned_workers_exited=False))
rejected('wrong_cards',lambda c:c[0]['validation'].update(physical_cards=[1,2,3,5]))

result=dict(status='HOST_synthetic_passed',tests=len(passed),passed=passed,NPU_used=False,model_launches=0,
            actual_experiment_cells=0,performance_result=False,goal_achieved=False)
Path(__file__).with_name('FINAL_COMMON_COMPARISON_HOST_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
