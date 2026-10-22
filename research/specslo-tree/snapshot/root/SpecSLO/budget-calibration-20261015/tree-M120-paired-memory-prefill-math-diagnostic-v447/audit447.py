"""Actual fresh445 scope and nine new initial/multigroup comparisons."""
from pathlib import Path
import json,math,statistics
KEYS=('prefill_hidden_exact','prefill_tokens_exact','prefill_all128_query_KV_exact',
    'decode_hidden_exact','decode_tokens_exact','decode_all128_query_KV_exact',
    'post_pair_prefill_hidden_stable','post_pair_prefill128_KV_exact')
CASES=['initial-L'+str(n)for n in(1,2,3,4,5,17,40)]+['continuation-L17','continuation-L40']
def audit(directory):
    from audit445 import audit as original
    directory=Path(directory);parent,_,_=original(directory);reports=[]
    for rank in(1,2,3):
        r=json.loads((directory/f'rank{rank}-bulk-prefill447.json').read_text())
        if r['rank']!=rank or r['physical_M']!=120 or r['physical_S']!=24 or r['Native_flags_assigned']is not False or r['production_permission']is not False:
            raise RuntimeError('ROOT447 actual diagnostic scope')
        if [v['rank']for v in r['by_target_rank']]!=[1,2,3]:raise RuntimeError('ROOT447 actual3 target roles')
        for v in r['by_target_rank']:
            before=json.loads((directory/f"rank{v['rank']}-bulk-prefill447-completed-numerical-records.json").read_text())
            if (v['tested_graph_variants']!=['baseline',*CASES]or before['cases']!=v['cases']
                or before['tested_graph_variants']!=v['tested_graph_variants']):raise RuntimeError('ROOT447 exact10 owned graph frames and before-timing comparisons')
            if v['all_parameter_storages_unchanged']is not True or v['MC2_manifest_unchanged']is not True:
                raise RuntimeError('ROOT447 unchanged weights and static routes')
            if [c['name']for c in v['cases']]!=CASES:raise RuntimeError('ROOT447 all9 actual cases')
            if v['maximum_bulk_diagnostic_graphs']!=2:raise RuntimeError('ROOT447 at most two diagnostic graphs in each bulk pair')
            for c in v['cases']:
                earlier=json.loads((directory/f"rank{v['rank']}-bulk-prefill447-{c['name']}-before-timing.json").read_text())
                if earlier['actual_graph_pair']!=['baseline',c['name']]or c['actual_graph_pair']!=earlier['actual_graph_pair']:
                    raise RuntimeError('ROOT447 actual two owned frames during each pair')
                if any(c[k]!=value for k,value in earlier['case'].items()):raise RuntimeError('ROOT447 preserved actual pre-timing NN comparisons')
                if len(c['paired_baseline_event_ms'])!=9 or any(not math.isfinite(t)or t<=0 for t in c['paired_baseline_event_ms']):
                    raise RuntimeError('ROOT447 actual9 per-variant paired baseline samples')
                if c['candidate_owner']==c['independent_serial_reference_owner']or len(c['query_KV'])!=128:
                    raise RuntimeError('ROOT447 independent complete reference KV')
                if c['prefill_tokens_exact']!=(c['actual_tokens']==c['reference_tokens']):raise RuntimeError('ROOT447 actual token arithmetic')
                if c['prefill_all128_query_KV_exact']!=all(x['equal']is True for x in c['query_KV']):raise RuntimeError('ROOT447 actual128-KV equality arithmetic')
                if c['prefill_hidden_exact']is True and c['prefill_hidden_max_abs']!=0.:raise RuntimeError('ROOT447 exact hidden has zero error')
                for x in c['query_KV']:
                    if x['equal']is True and(x['actual_sha256']!=x['reference_sha256']or x['max_abs']!=0.):raise RuntimeError('ROOT447 exact actual KV hash')
                if c['family']=='initial'and(c['prefix_length']!=0 or(c['candidate_owner'],c['independent_serial_reference_owner'])!=(72,73)):
                    raise RuntimeError('ROOT447 fresh empty-prefix owners')
                if c['family']=='continuation'and(c['prefix_length']!=127 or(c['candidate_owner'],c['independent_serial_reference_owner'])!=(16,40)):
                    raise RuntimeError('ROOT447 actual page-crossing independently prefetched owners')
            if v['all9_cases_math_exact']!=all(all(c[k]is True for k in KEYS)for c in v['cases']):raise RuntimeError('ROOT447 actual9-case exactness')
        if r['all3_all9_cases_math_exact']!=all(v['all9_cases_math_exact']for v in r['by_target_rank']):raise RuntimeError('ROOT447 all3 exactness arithmetic')
        for name,cost in r['max_rank_costs'].items():
            values=[max(v['whole_target_event_ms'][name][j]for v in r['by_target_rank'])for j in range(9)]
            if any(not math.isfinite(v)or v<=0 for v in values)or values!=cost['max_rank_samples_ms']or statistics.median(values)!=cost['p50_ms']:
                raise RuntimeError('ROOT447 actual9 max-rank event windows')
        if r['maximum_bulk_diagnostic_graphs']!=2:raise RuntimeError('ROOT447 actual bounded bulk pair lifetime')
        for index,name in enumerate(CASES):
            base=[max(v['cases'][index]['paired_baseline_event_ms'][j]for v in r['by_target_rank'])for j in range(9)]
            measured=r['paired_max_rank_costs'][name]
            if (base!=measured['baseline_samples_ms']or statistics.median(base)!=measured['baseline_p50_ms']
                or r['max_rank_costs'][name]['max_rank_samples_ms']!=measured['variant_samples_ms']
                or r['max_rank_costs'][name]['p50_ms']!=measured['variant_p50_ms']):
                raise RuntimeError('ROOT447 independently paired actual max-rank event arithmetic')
        reports.append(r)
    if any(r['by_target_rank']!=reports[0]['by_target_rank']for r in reports):raise RuntimeError('ROOT447 identical retained roles')
    return(dict(status='retained_actual447_initial_and_bulk_known_prefill_math_cost',fresh445_math_exact=parent['all3_all5_lengths_math_exact'],
        reports=reports,all3_all9_cases_math_exact=reports[0]['all3_all9_cases_math_exact'],
        Native_flags_assigned=False,production_permission=False,whole_workload_AR_qualified=False,
        Raw=None,Goodput=None,adopted=False),'initial_and_continuation_same_owner_known_spans_only_not_serving_qualification',dict(NN_reexecuted_by_HOST=False))
