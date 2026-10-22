"""Audit retained actual q5 padding-prefill math, including negative results."""
from pathlib import Path
import json,math,statistics

EXACT_KEYS=('decode80_hidden_exact','decode80_tokens_exact','decode128_query_KV_exact',
    'prefill_hidden_exact','prefill_tokens_exact','prefill_all128_query_KV_exact')
def audit(directory):
    directory=Path(directory);reports=[]
    if list(directory.glob('rank*-last-error.json')):raise RuntimeError('ROOT445 no retained worker errors')
    for rank in(1,2,3):
        r=json.loads((directory/f'rank{rank}-padding-prefill445.json').read_text())
        if r['rank']!=rank or r['physical_M']!=120 or r['physical_S']!=24 or r['decode_owners']!=16:
            raise RuntimeError('ROOT445 actual unchanged physical target scope')
        if r['Native_flags_assigned']is not False or r['production_permission']is not False:
            raise RuntimeError('ROOT445 no serving permission')
        if [v['rank']for v in r['by_target_rank']]!=[1,2,3]:raise RuntimeError('ROOT445 all3 real target roles')
        for v in r['by_target_rank']:
            before=json.loads((directory/f"rank{v['rank']}-padding-prefill445-before-timing.json").read_text())
            expected_frames=['baseline','chunk1','chunk2','chunk3','chunk4','chunk5']
            if (v['actual_owned_graph_frames']!=expected_frames
                or before['actual_owned_graph_frames']!=expected_frames
                or before['cases']!=v['cases']):
                raise RuntimeError('ROOT445 exact six owned graph frames and actual saved pre-timing comparisons')
            if (v['all_parameter_storages_unchanged']is not True or v['MC2_manifest_unchanged']is not True
                or v['original_decode_owners_unchanged']is not True):raise RuntimeError('ROOT445 unchanged parameters and16 decode owners')
            if len(v['cases'])!=5:raise RuntimeError('ROOT445 all5 chunk lengths')
            for length,c in enumerate(v['cases'],1):
                if c['chunk_length_per_owner']!=length or c['prefill_queries']!=8*length or c['all_finite']is not True:
                    raise RuntimeError('ROOT445 actual known prefill query counts and finite values')
                if c['real_prefill_owners']!=list(range(16,24))or c['independent_serial_reference_owners']!=list(range(40,48)):
                    raise RuntimeError('ROOT445 independently prefetched reference owners')
                if len(c['actual_rows'])!=8*length or len(c['query_KV'])!=128:
                    raise RuntimeError('ROOT445 complete actual chunk rows and128 KV arrays')
                if c['prefill_tokens_exact']!=(c['actual_prefill_tokens']==c['serial_reference_tokens']):
                    raise RuntimeError('ROOT445 retained actual token arithmetic')
                kv_equal=all(t['equal']is True for t in c['query_KV'])
                if kv_equal!=c['prefill_all128_query_KV_exact']:raise RuntimeError('ROOT445 retained128-KV equality arithmetic')
                for t in c['query_KV']:
                    if t['equal']is True and(t['actual_sha256']!=t['reference_sha256']or t['max_abs']!=0.):
                        raise RuntimeError('ROOT445 exact KV hashes and zero difference')
                if c['prefill_hidden_exact']is True and c['prefill_hidden_max_abs']!=0.:
                    raise RuntimeError('ROOT445 exact hidden means zero difference')
            exact=all(all(c[k]is True for k in EXACT_KEYS)for c in v['cases'])
            if exact!=v['all_math_exact']:raise RuntimeError('ROOT445 complete actual per-rank math aggregation')
        if r['all3_all5_lengths_math_exact']!=all(v['all_math_exact']for v in r['by_target_rank']):
            raise RuntimeError('ROOT445 actual all3 math aggregation')
        for name,cost in r['max_rank_costs'].items():
            if name not in ('baseline','chunk1','chunk2','chunk3','chunk4','chunk5'):raise RuntimeError('ROOT445 explicit graph variants')
            values=[max(v['whole_target_device_event_ms'][name][i]for v in r['by_target_rank'])for i in range(9)]
            if any(not math.isfinite(v)or v<=0 for v in values)or values!=cost['max_rank_samples_ms']or statistics.median(values)!=cost['p50_ms']:
                raise RuntimeError('ROOT445 actual9 event max-rank arithmetic')
        reports.append(r)
    if any(r['by_target_rank']!=reports[0]['by_target_rank']for r in reports):raise RuntimeError('ROOT445 identical retained all3 reports')
    for rank in range(4):
        r=json.loads((directory/f'rank{rank}-numeric.json').read_text())
        if r['rank']!=rank or r['model_dtype']!='torch.bfloat16'or r['new_profile_installed']is not False or r['current_profile_unchanged']is not True:
            raise RuntimeError('ROOT445 original protected BF16 capture')
        if rank and(r['independent_prefills']!=72 or r['all72_prefixes_protected']is not True):
            raise RuntimeError('ROOT445 original all72 committed prefixes intact')
    c=json.loads((directory/'completed.json').read_text())
    if sorted(c['replies'],key=lambda r:r[1])!=[['calibrated',rank,dict(numeric_cases=1,performance=False)]for rank in range(4)]:
        raise RuntimeError('ROOT445 all4 completed actual roles')
    return (dict(status='retained_actual445_padding_prefill_math_cost',reports=reports,
        all3_all5_lengths_math_exact=reports[0]['all3_all5_lengths_math_exact'],
        whole_workload_AR_qualified=False,Native_flags_assigned=False,
        production_permission=False,Raw=None,Goodput=None,adopted=False),
        'one_N16_decode_plus8_known_prompt_continuations_only_not_serving_qualification',
        dict(NN_reexecuted_by_HOST=False))
