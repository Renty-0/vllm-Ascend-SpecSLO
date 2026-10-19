"""Reconstruct all693 model gates and every252-call per-bank cost panel."""
import hashlib
import json
import math
from pathlib import Path
import statistics


def need(value,message):
    if value is not True:
        raise RuntimeError('ROOT378 actual audit: '+message)


def read(path):
    return json.loads(Path(path).read_text())


def panel(value,prefixes,rank):
    from benchmark378 import arms
    from geometry377 import choose
    from bucket377 import extent
    target=rank!=0
    report=value['paired_physical_cost']
    cases=list(arms());modes=['fixed120','adaptive80_120']
    need(report['rank']==rank and report['target_math_on_rank'] is target
         and report['arms']==cases and report['modes']==modes and report['rounds']==20,
         'same six predeclared arms, modes and20 rounds')
    need(report['no_new_capture_validation_or_fallback'] is True
         and report['outputs_equal_between_modes'] is True
         and report['complete_serving'] is False and report['real_draft_generation'] is False
         and report['Raw'] is report['Goodput'] is None,'component-only actual graph and math gates')
    keys=('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays',
          'aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks')
    need(all(report['graph_counters_before'][k]==report['graph_counters_after'][k]for k in keys),
         'no new timed graph/validation/fallback work')
    expected=[(arm,mode)for arm in cases for mode in modes]
    samples={a['name']:{m:[]for m in modes}for a in cases}
    first={a['name']:{m:0 for m in modes}for a in cases}
    need(len(report['pairs'])==20,'all20 balanced rounds retained')
    for iteration,pair in enumerate(report['pairs']):
        order=cases[iteration%6:]+cases[:iteration%6]
        need(pair['iteration']==iteration and pair['arm_order']==[a['name']for a in order]
             and len(pair['samples'])==6,'arm rotation complete')
        for arm,observation in zip(order,pair['samples'],strict=True):
            mode_order=modes if(iteration+cases.index(arm))%2==0 else modes[::-1]
            need(observation['arm']==arm['name']and observation['mode_order']==mode_order
                 and set(observation['milliseconds'])==set(modes),'per-arm alternating mode order')
            first[arm['name']][mode_order[0]]+=1
            expected.extend((arm,m)for m in mode_order)
            for mode,elapsed in observation['milliseconds'].items():
                if target:
                    need(type(elapsed)in(int,float)and math.isfinite(elapsed)and elapsed>0,'positive real elapsed')
                    samples[arm['name']][mode].append(elapsed)
                else:need(elapsed is None,'draft has no target elapsed')
    need(all(count==10 for values in first.values()for count in values.values()),'each arm has10 first and10 second samples')
    calls=value['actual_cost_calls']
    if not target:
        need(calls==[]and value['all72_prefixes_after_cost'] is None,'draft no target call or prefix claim')
        return []
    need(value['all72_prefixes_after_cost'] is True and len(calls)==len(expected)==252,
         'all12 warm and240 timed actual dispatches and committed prefixes protected')
    for call,(arm,mode) in zip(calls,expected,strict=True):
        m=choose(arm['budgets'],enabled=mode=='adaptive80_120');s=m//5
        kv=extent(tuple(len(prefixes[i])+min(b,4)+1 for i,b in enumerate(arm['budgets'])))
        branch=arm['budget']==5
        need(call['N']==arm['N']and call['real_queries']==arm['N']+sum(arm['budgets'])
             and call['M']==m and call['attention_segments']==s
             and call['actual_seq_lengths_q']==list(range(5,m+1,5))and call['sequence_lens']==[kv]*s
             and call['mask_shape']==[s,1,11 if branch else 10,4096]
             and call['copy_enabled'] is branch and call['used_aclgraph'] is True
             and call['execution_mode']=='replay'and call['fallback_reason'] is None
             and call['primary_side_groups']==arm['N']+arm['budgets'].count(5)
             and call['adaptive377_enabled'] is(mode=='adaptive80_120'),
             'actual original replay has complete selected80/120 geometry')
        detail=call['uniform_KV_bucket377']
        need(detail['enabled'] is True and detail['extent']==kv
             and extent(tuple(detail['original_sequence_lens']))==kv,'same checked visible KV extent in both modes')
    return [dict(rank=rank,arm=a['name'],fixed120_median_ms=statistics.median(samples[a['name']]['fixed120']),
        adaptive_median_ms=statistics.median(samples[a['name']]['adaptive80_120']),
        adaptive_M=choose(a['budgets'],enabled=True))for a in cases]


def audit(directory):
    from oracle377 import cases
    from dispatch377 import audit_calls
    directory=Path(directory).resolve()
    completed=read(directory/'completed.json')
    need(completed['replies']==[['calibrated',r,dict(numeric_cases=693,performance=False)]for r in range(4)]
         and completed['performance'] is False and completed['adopted'] is False,
         'all four original lifecycle replies and complete component')
    need(read(directory/'worker-exit.json')==dict(closed=True,all_exited=True),
         'original owned worker shutdown completed')
    reports=[read(directory/f'rank{r}-numeric.json')for r in range(4)]
    expected=cases();heads=[];costs=[]
    need(len(expected)==231,'original227 plus4 maximum80 branch cases')
    for rank,report in enumerate(reports):
        target=rank!=0
        need(report['rank']==rank and report['actual_model_cases']==693
             and report['independent_prefills']==(216 if target else 0)
             and report['physical_M']==[80,120]and report['groups']==[16,24]and report['Q']==5
             and report['real_budget_domain']==[1,2,3,4,5]
             and report['independent_reference_FULL120'] is True
             and report['real_draft_generation'] is False and report['whole_workload_AR_qualified'] is False
             and report['performance'] is False and report['Raw'] is report['Goodput'] is None
             and report['model_dtype']=='torch.bfloat16','truthful actual role and original BF16 model')
        if target:
            need(report['MC2_routes'] is not None and len(report['MC2_routes']['routes'])==128,'actual original128 MC2 routes')
        need([x['name']for x in report['banks']]==['mixed','mid','short'],'all three independent prefix banks')
        rank_heads=[]
        for bank in report['banks']:
            def retained(kind):
                path=Path(bank[kind+'_path']).resolve()
                need(path.is_relative_to(directory)and hashlib.sha256(path.read_bytes()).hexdigest()==bank[kind+'_sha256'],
                     'actual retained '+kind+' proof digest')
                return read(path)
            proof=retained('math');cost=retained('cost');pref=[[0]*n for n in bank['prefix_lengths']]
            need(proof['rank']==rank and len(proof['math']['cases'])==231 and bank['cases']==231,'all231 current model cases')
            if target:
                evidence=proof['bank']
                need(evidence['independent_neural_prefills']==72 and evidence['all_triplet_roots_equal'] is True
                     and evidence['all_triplet_prefix_KV_equal'] is True
                     and proof['math']['all_independent_prefixes_unchanged'] is True,'all72 independent own-prefix neural banks')
                audit_calls(expected,proof['actual_calls'],pref)
                for case,observation in zip(expected,proof['math']['cases'],strict=True):
                    need(observation['case']==case and observation['exact_greedy'] is True
                         and observation['hidden_bitwise_equal'] is True and observation['KV_bitwise_equal'] is True
                         and observation['all128_KV_original_matcher_passed'] is True
                         and observation['graph_eager_KV_bitwise_equal'] is True
                         and observation['candidate_tokens']==observation['reference_tokens']
                         and observation['candidate_KV_copied'] is False,'strict independent FULL120 hidden/KV/greedy gates')
                    rank_heads.append(observation['candidate_tokens'])
            else:
                need(proof['bank'] is None and proof['dispatch'] is None and proof['actual_calls']==[]
                     and all(v is None for v in proof['math']['cases']),'draft has no target neural math')
            rows=panel(cost,pref,rank)
            costs.extend(dict(row,bank=bank['name'])for row in rows)
        if target:heads.append(rank_heads)
    need(heads[0]==heads[1]==heads[2],'all3 TP3 target heads agree on693 cases')
    value=dict(status='PASS_current_target693_adaptive80_120_and_same_instance_costs378',
        cases_each_target=693,independent_prefills_each_target=216,paired_costs=costs,
        TP3_ranks_are_not_independent_experiments=True,full463=False,whole463_AR_qualified=False,
        Raw=None,Goodput=None,adopted=False)
    return value,'passed_current_target693_adaptive_physical_math_and_costs',dict(
        hidden_KV_recomputed_by_HOST=False,all_reported_Native_gates_and_costs_reconstructed=True)
