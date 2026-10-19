"""Strict finite model and current-cost reader with explicit qualification calls."""
import json
import math
from pathlib import Path
import statistics

from geometry341 import cases, arms, extent

P=Path(__file__).resolve().parent
B=P.parent


def need(condition,message):
    if not condition:raise RuntimeError('ROOT341 audit: '+message)


def audit(directory):
    directory=Path(directory)
    fixture=json.loads((B/'tree-M160-S32-N32-target-math-cost-v326/FIXTURE326.json').read_text())
    expected=cases();cost_arms=arms();names=[a['name']for a in cost_arms]
    need(len(expected)==42,'all42 planned same-cohort model cases')
    reports=[json.loads((directory/f'rank{r}-numeric.json').read_text())for r in range(4)]
    all_heads=[];costs=[]
    stable_keys=('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays',
        'aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks')
    for rank,report in enumerate(reports):
        need(report['rank']==rank and report['case_count']==len(report['cases'])==42,'complete original four roles')
        need(report['performance'] is report['adopted'] is report['real_draft_generation'] is False
             and report['whole_workload_AR_qualified'] is False,'target component only')
        checkpoint=json.loads((directory/f'rank{rank}-independent-bank-before-math341.json').read_text())
        need(checkpoint['rank']==rank and checkpoint['evidence']==report['bank'],'actual pre-math prefix checkpoint')
        if rank==0:
            need(all(x is None for x in report['cases']) and report['bank'] is report['causal_precision'] is None,
                 'draft makes no target model claims')
        else:
            need(report['model_dtype']=='torch.bfloat16' and report['all_prefixes_exact_after_cost'] is True,
                 'BF16 model and protected independent prefixes')
            bank=report['bank']
            need(bank['independent_neural_prefills']==96 and bank['all_triplet_roots_equal'] is bank['all_triplet_prefix_KV_equal'] is True,
                 'fresh96 independent neural prefixes')
            activation=report['activation']
            need(activation['actual_M']==120 and activation['groups']==24
                 and activation['all_original_wrapper_code_and_signatures_preserved'] is True
                 and activation['original_model_forward_unchanged'] is True
                 and activation['Native_flags_assigned'] is activation['deferred_read_done'] is False,
                 'actual original wrappers, M120 model and flushed MC2')
            precision=report['causal_precision']
            need(precision['actual_model_alias_bound'] is True and precision['runtime_qualification_flags_assigned'] is False,
                 'actual model callback and unchanged qualification flags')
            for key in ('eager_capture','task_update'):
                need(precision[key]['exact_source_inverse'] is precision[key]['original_live_module_globals_preserved'] is True,
                     'paired causal precision ABI preserves live graph state')
            need(len(report['MC2_routes']['routes'])==128,'all128 actual TP3 MC2 modules')
            before,after=report['original_engine_metrics_before'],report['original_engine_metrics_after']
            for key in ('mc2_dispatch_fallback','mc2_dispatch_exception','aclgraph_failed_captures',
                        'aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks'):
                need(after[key]==before[key],'no original model/graph failure '+key)
            for key in ('mc2_dispatch_native_epilogue_chained_success','mc2_dispatch_native_epilogue_chain_flush'):
                need(after[key]>before[key],'actual qualified original MC2 execution '+key)
            heads=[];observed=[];captures=0
            for case,row in zip(expected,report['cases'],strict=True):
                need(row['case']==case,'exact actual case identity')
                cross=row['same_cohort_FULL2560']
                need(cross==dict(control=case['family']=='FULL2560',greedy=True,hidden=True,all128KV=True),
                     'same-cohort original FULL2560 bitwise comparison')
                need([x['future']for x in row['comparisons']]==[0,17],'both independent future canaries')
                for comparison in row['comparisons']:
                    need(all(comparison[k] is True for k in ('exact_greedy','hidden_bitwise_equal',
                        'all128_KV_bitwise_equal','graph_eager_KV_bitwise_equal'))
                        and comparison['candidate_KV_copied'] is False,'all original strict node checks')
                    need(comparison['candidate_tokens']==comparison['reference_tokens']
                         and len(comparison['candidate_tokens'])==case['query_count'],'all independently evaluated nodes')
                    frames=comparison['frames']
                    need([x['depth']for x in frames]==list(range(5)) and all(
                        x['real_nodes']==case['N'] and x['physical_queries']==120 and x['future']==comparison['future']
                        and x['candidate_KV_copied'] is False and x['reference_eager_only'] is True for x in frames),
                         'all complete independent own-history frames')
                need(row['comparisons'][0]['reference_tokens']==row['comparisons'][1]['reference_tokens'],
                     'future values cannot affect consumed reference roots')
                heads.append(row['comparisons'][0]['candidate_tokens'])
                calls=row['observed_native_calls'];work=row['graph_work']
                need((len(calls),work['aclgraph_captures'],work['aclgraph_runtime_validation_replays'])in((1,0,0),(3,1,1)),
                     'explicit original actual/changed/restored first-graph qualification calls')
                for call in calls:
                    expected_extent=extent(case,fixture['prefixes'])
                    need(call['family']==case['family'] and call['M']==120 and call['groups']==24
                         and call['extent']==expected_extent and call['actual_seq_lengths_q']==list(range(5,121,5))
                         and call['used_aclgraph'] is True and call['fallback_reason'] is None,
                         'actual original graph execution and full query geometry')
                    causal=case['family']=='causal-inner1'
                    lengths=[len(fixture['prefixes'][owner])+5 for owner in case['owners']]+[5]*(24-case['N']) if causal else [expected_extent]*24
                    need(call['tree_attention'] is (not causal) and call['sequence_lens']==lengths
                         and call['mask_shape']==(None if causal else [24,1,10,4096]),'actual family and complete context extent')
                observed.extend(calls);captures+=work['aclgraph_captures']
            need(observed==report['cold_native_calls'],'every actual Native call accounted once')
            need(captures>=2 and after['aclgraph_captures']-before['aclgraph_captures']==captures
                 and after['aclgraph_runtime_validation_replays']-before['aclgraph_runtime_validation_replays']==captures,
                 'all actual new-family capture and changed-input work accounted')
            all_heads.append(heads)
        panel=report['paired_target_cost'];need(panel['arms']==cost_arms and panel['rounds']==len(panel['pairs'])==24,
                                               'all six real cost arms and24 rotations')
        need(panel['no_new_graph_work'] is True and all(panel['graph_counters_before'][k]==panel['graph_counters_after'][k]
             for k in stable_keys),'no timed graph capture, validation or fallback')
        samples={name:[]for name in names}
        for rotation,pair in enumerate(panel['pairs']):
            need(pair['order']==names[rotation%6:]+names[:rotation%6] and set(pair['milliseconds'])==set(names),
                 'balanced six-arm rotation')
            for name,value in pair['milliseconds'].items():
                if rank==0:need(value is None,'no invented draft elapsed')
                else:
                    need(type(value)in(int,float)and math.isfinite(value)and value>0,'positive actual target elapsed')
                    samples[name].append(value)
        if rank:costs.append(dict(rank=rank,median_ms={name:statistics.median(values)for name,values in samples.items()}))
    need(all_heads[0]==all_heads[1]==all_heads[2],'all target shard token decisions agree')
    need(json.loads((directory/'completed.json').read_text())['replies']==[
        ['calibrated',r,dict(numeric_cases=42,performance=False)]for r in range(4)],'normal original worker protocol')
    need(json.loads((directory/'worker-exit.json').read_text())==dict(closed=True,all_exited=True),'normal owned worker lifecycle')
    return (dict(status='complete_current_Q5_attention_model_component',cases_each_target=42,
        strict_cross_FULL2560=True,strict_independent_canaries=True,costs=costs,
        median_of_three_rank_medians_ms={name:statistics.median(row['median_ms'][name]for row in costs)for name in names},
        whole_workload_AR_qualified=False,actual_dynamic_B_or_tree_branch_qualified=False,
        real_draft_generation=False,Goodput=None,adopted=False),
        'passed_finite_Q5_attention_math_and_rotated_cost',dict(all_real_nodes_and_costs_recomputed=True))
