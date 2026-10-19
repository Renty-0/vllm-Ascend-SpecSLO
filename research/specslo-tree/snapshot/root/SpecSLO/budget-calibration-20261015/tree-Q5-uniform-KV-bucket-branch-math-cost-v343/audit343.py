"""Independent finite tree math, dispatch and current-cost accounting."""
import json
import math
from pathlib import Path
import re
import statistics

import runtime343
from domain343 import cases,prelude,short_cases,short_prefixes,mid_prefixes,TOTAL_CASES
from bucket343 import case_extent
from dispatch343 import audit_calls


def need(value,message):
    if not value:raise RuntimeError('ROOT343 audit: '+message)


def math_rows(value,expected):
    need(value['target_math_on_rank'] is value['all_independent_prefixes_unchanged'] is True,
         'real target math and protected own histories')
    rows=value['cases'];need(len(rows)==len(expected),'all planned node cases')
    heads=[]
    for case,row in zip(expected,rows,strict=True):
        need(row['case']==case and row['real_predictions']==case['query_count'],'exact real-node geometry')
        need(all(row[k]is True for k in ('exact_greedy','hidden_bitwise_equal','KV_bitwise_equal',
             'all128_KV_original_matcher_passed','graph_eager_KV_bitwise_equal'))
             and row['candidate_KV_copied'] is False,'all original and strict node gates')
        need(row['candidate_tokens']==row['reference_tokens']
             and len(row['candidate_tokens'])==case['query_count'],'every actual node independently evaluated')
        frames=row['independent_reference_frames']
        need(sum(frame['mapped_real_nodes']for frame in frames)==case['query_count']
             and all(frame['physical_queries']==120 and frame['reference_eager_only']is True
                     and frame['candidate_KV_copied']is False for frame in frames),'complete original FULL2560 own-history frames')
        heads.append(row['candidate_tokens'])
    return heads


def cost_rows(panel,calls,prefixes,rank,label):
    target=rank!=0
    need(panel['rank']==rank and panel['target_math_on_rank']is target
         and panel['no_new_capture_or_validation_in_costs']is True,'real current role and untimed qualification')
    keys=('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays',
          'aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks')
    need(all(panel['graph_counters_before'][k]==panel['graph_counters_after'][k]for k in keys),
         'no new graph work or fallback in costs')
    geometry=[(n,f)for n in(1,4,8,12,16,20,23)for f in sorted({1,min(n,24-n,(96-4*n)//4)})]
    need([(r['N'],r['branch_owners'])for r in panel['rows']]==geometry,'all12 current cost panels')
    expected_calls=[];result=[]
    for row,(n,forks)in zip(panel['rows'],geometry,strict=True):
        selected=case_extent(dict(budgets=[4]*n),prefixes)
        need(row['rounds']==len(row['pairs'])==20 and row['physical_M120']is True
             and row['q5_groups']==24 and row['uniform_KV_extent']==selected,'all real cost rounds and uniform extent')
        need(row['original_greedy_output_stable']is(True if target else None),'repeated original greedy heads stable')
        sample={str(b):[]for b in range(4,9)}
        expected_calls.extend((n,forks,b)for b in range(4,9))
        for rotation,pair in enumerate(row['pairs']):
            names=[4,5,6,7,8];order=names[rotation%5:]+names[:rotation%5]
            need(pair['order']==order and set(pair['milliseconds'])==set(sample),'balanced five-budget rotation')
            expected_calls.extend((n,forks,b)for b in order)
            for key,elapsed in pair['milliseconds'].items():
                if target:
                    need(type(elapsed)in(int,float)and math.isfinite(elapsed)and elapsed>0,'positive measured target elapsed')
                    sample[key].append(elapsed)
                else:need(elapsed is None,'draft has no invented target cost')
        if target:result.append(dict(rank=rank,bank=label,N=n,branch_owners=forks,
            KV_extent=selected,median_ms={b:statistics.median(v)for b,v in sample.items()}))
    if not target:
        need(calls==[],'draft never invokes target graph');return result
    need(len(calls)==len(expected_calls)==1260,'all actual warm and timed Native calls')
    for call,(n,forks,b)in zip(calls,expected_calls,strict=True):
        budgets=[b]*forks+[4]*(n-forks);branch=b>4
        selected=case_extent(dict(budgets=budgets),prefixes)
        count=sum(len(prefixes[i])%128+(b-4)+1 for i in range(forks))if branch else 0
        grid=next(x for x in(128,256,512,1024,1792)if x>=count)if count else 0
        need(call['N']==n and call['real_queries']==n+sum(budgets)and call['M']==120
             and call['attention_segments']==24 and call['actual_seq_lengths_q']==list(range(5,121,5))
             and call['sequence_lens']==[selected]*24 and call['mask_shape']==[24,1,11 if branch else 10,4096]
             and call['copy_enabled']is branch and call['copy_grid']==grid
             and call['primary_side_groups']==n+(forks if branch else 0)
             and call['used_aclgraph']is True and call['execution_mode']=='replay'
             and call['fallback_reason']is None,'actual timed target/copy geometry')
        key=call['actual_capture_key']
        need((f'|ROOT296:copy-grid:{grid}'in key)if branch else('|ROOT296:copy-grid:'not in key),
             'actual original copy-family graph key')
        detail=call['uniform_KV_bucket343']
        need(detail['enabled']is True and detail['extent']==selected
             and 0<detail['original_max_context']<=selected,'actual runtime no-truncation bound')
    return result


def audit(directory):
    directory=Path(directory)
    fixture=json.loads((runtime343.original.P/'FIXTURE311.json').read_text())
    inputs=[('mixed',fixture['prefixes'],cases()),
        ('mid-residue-preserved',mid_prefixes(fixture['prefixes']),short_cases()),
        ('short-residue-preserved',short_prefixes(fixture['prefixes']),short_cases())]
    reports=[json.loads((directory/f'rank{r}-numeric.json').read_text())for r in range(4)]
    heads=[];costs=[]
    need(all(len(report['MC2_routes']['routes'])==128 for report in reports[1:]),'all128 actual TP3 routes')
    for rank,report in enumerate(reports):
        need(report['rank']==rank and report['actual_model_cases']==TOTAL_CASES==594
             and report['physical_M']==120 and report['groups']==24,'all actual four-role reports')
        need(report['model_dtype']=='torch.bfloat16'and report['performance']is report['adopted']is False
             and report['whole_workload_AR_qualified']is report['real_draft_generation']is False
             and report['original_reference_FULL2560']is True and report['Goodput']is None,'component only, original precision/reference')
        need(report['independent_prefills']==(216 if rank else 0)and len(report['banks'])==3,'three fresh banks')
        rank_heads=[]
        for row,(label,prefixes,selected)in zip(report['banks'],inputs,strict=True):
            need(row['name']==label and row['prefix_lengths']==[len(p)for p in prefixes],'exact teacher prefixes and declared cases')
            checkpoint=json.loads((directory/label/f'rank{rank}-bank-before-math.json').read_text())
            need(checkpoint['rank']==rank and checkpoint['evidence']==row['bank'],'fresh independent bank retained before math')
            if rank:
                bank=row['bank']
                need(bank['independent_neural_prefills']==72 and bank['all_triplet_roots_equal']is bank['all_triplet_prefix_KV_equal']is True,
                     'all72 independent neural prefixes')
                if label=='mixed':
                    maximum=row['maximum_extent_qualification']
                    rank_heads.append(math_rows(maximum['math'],prelude()))
                    need(maximum['dispatch']==audit_calls(prelude(),maximum['actual_calls'],prefixes,enabled=False),
                         'all6 natural max-extent capture/change/restore qualifications')
                else:need(row['maximum_extent_qualification']is None,'qualified graph inventory reused')
                rank_heads.append(math_rows(row['math'],selected))
                need(row['dispatch']==audit_calls(selected,row['actual_calls'],prefixes,enabled=True),
                     'every actual original physical permutation and bucket reconstructed')
                need(row['all72_prefixes_after_cost']is True,'all prefixes protected after all costs')
                before,after=row['model_metrics_before'],row['model_metrics_after']
                for key in('mc2_dispatch_fallback','mc2_dispatch_exception','aclgraph_captures',
                    'aclgraph_runtime_validation_replays','aclgraph_failed_captures',
                    'aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks'):
                    need(after[key]==before[key],'original model/graph guard unchanged '+key)
                for key in('mc2_dispatch_native_epilogue_chained_success','mc2_dispatch_native_epilogue_chain_flush'):
                    need(after[key]>before[key],'actual original flushed MC2 '+key)
            else:
                need(row['bank']is None and all(x is None for x in row['math']['cases'])
                     and row['actual_calls']==[],'draft makes no target numeric claims')
            need(re.fullmatch('[0-9a-f]{64}',row['reconstructable_full_cost_call_SHA'])is not None,
                 'full repeated cost mapping digest retained for exact regeneration')
            costs.extend(cost_rows(row['paired_target_cost'],row['actual_cost_calls'],prefixes,rank,label))
        if rank:heads.append(rank_heads)
    need(heads[0]==heads[1]==heads[2],'all three actual target token decisions agree')
    need(json.loads((directory/'worker-exit.json').read_text())==dict(closed=True,all_exited=True),'normal original worker lifecycle')
    need(json.loads((directory/'completed.json').read_text())==dict(replies=[
        ['calibrated',r,dict(numeric_cases=594,performance=False)]for r in range(4)],performance=False,adopted=False),
         'all four original completion replies')
    return (dict(status='PASS_current_M120_bucket_branch594_target_only',target_cases_each=594,
        target_costs=costs,original_reference_FULL2560=True,independent_prefixes_per_target=216,
        real_draft_generation=False,whole_workload_AR_qualified=False,Goodput=None,adopted=False),
        'passed_finite_branch_bucket_math_and_current_costs',dict(all_geometry_and_costs_recomputed=True))
