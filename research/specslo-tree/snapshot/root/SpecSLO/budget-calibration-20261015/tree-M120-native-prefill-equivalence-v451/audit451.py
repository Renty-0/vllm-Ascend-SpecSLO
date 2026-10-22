"""Actual one-graph numerical records; failed cases cannot claim timings."""
from pathlib import Path
import json,math,statistics
from forward451 import KEYS
CASES=['initial-L'+str(v)for v in(1,2,3,4,5,17,40)]+['continuation-L17','continuation-L40']
def audit(directory):
    directory=Path(directory);reports=[]
    if list(directory.glob('rank*-last-error.json')):raise RuntimeError('ROOT451 no worker errors')
    for rank in(1,2,3):
        r=json.loads((directory/f'rank{rank}-native-prefill-equivalence451.json').read_text())
        if r['rank']!=rank or(r['physical_M'],r['physical_S'],r['diagnostic_graphs'])!=(120,24,1):raise RuntimeError('ROOT451 exact one-graph physical scope')
        if r['Native_flags_assigned']is not False or r['production_permission']is not False:raise RuntimeError('ROOT451 no serving permission')
        if [v['rank']for v in r['by_target_rank']]!=[1,2,3]:raise RuntimeError('ROOT451 actual3 target roles')
        for v in r['by_target_rank']:
            for key in('exactly_one_persistent_diagnostic_graph','all_reference_model_calls_before_capture',
                'no_graph_reset_or_empty_cache_during_epoch','static_input_pointers_unchanged',
                'all_parameter_storages_unchanged','all_parameter_bytes_SHA256_unchanged','MC2_manifest_unchanged'):
                if v[key]is not True:raise RuntimeError('ROOT451 exact current lifetime and bindings '+key)
            if v['ordinary_model_body_calls_after_capture']!=0:raise RuntimeError('ROOT451 no eager-body/capture interleave')
            wb=json.loads((directory/f"rank{v['rank']}-native-prefill451-weight-hashes-before.json").read_text())
            wa=json.loads((directory/f"rank{v['rank']}-native-prefill451-weight-hashes-after.json").read_text())
            if not wb or wb!=wa:raise RuntimeError('ROOT451 actual before/after parameter-byte fingerprints')
            pages=[p for row in v['page_inventory_after']for p in row if p>=0]
            if len(pages)!=len(set(pages))or any(v['page_inventory_before'][i]!=v['page_inventory_after'][i]for i in range(72)):
                raise RuntimeError('ROOT451 all allocated pages disjoint and72 old owners retained')
            if [c['name']for c in v['cases']]!=CASES:raise RuntimeError('ROOT451 all9 actual cases')
            for c in v['cases']:
                if set(x['slot']for x in c['actual_rows'])&set(v['decode_query_slots']):raise RuntimeError('ROOT451 no prompt/decode query write alias')
                if len(c['query_KV'])!=128 or c['candidate_owner']==c['independent_serial_reference_owner']:
                    raise RuntimeError('ROOT451 complete independent128 query KV')
                if c['prefill_tokens_exact']!=(c['actual_tokens']==c['reference_tokens']):raise RuntimeError('ROOT451 actual token arithmetic')
                if c['prefill_all128_query_KV_exact']!=all(x['equal']is True for x in c['query_KV']):raise RuntimeError('ROOT451 actual128-KV arithmetic')
                if c['prefill_hidden_exact']is True and c['prefill_hidden_max_abs']!=0.:raise RuntimeError('ROOT451 exact hidden has zero difference')
                gates=c['actual_all3_pre_timing_gate']
                if [x['rank']for x in gates]!=[1,2,3]or any(x['name']!=c['name']for x in gates):raise RuntimeError('ROOT451 actual all3 per-case gate')
                if c['timing_skipped']is not None:
                    if all(x['exact_before']is True for x in gates)or any(c['device_event_ms'].values()):raise RuntimeError('ROOT451 negative case has no timing claim')
                elif not all(x['exact_before']is True for x in gates):raise RuntimeError('ROOT451 positive timing requires actual all3 numerical gate')
            if v['ordinary_prefill_reference_owner_ids']!=[74,75,76,77,78]:raise RuntimeError('ROOT451 independent Native owners')
            saved_AR=json.loads((directory/f"rank{v['rank']}-native-prefill451-ordinary-AR-before-capture.json").read_text())
            expected_AR=True
            for family,ar in v['actual_independent_Native_AR'].items():
                reference=ar['independent_native_reference']
                if reference!=saved_AR[family]or reference['owner']not in(77,78)or reference['KV_copied_from_candidate']is not False:
                    raise RuntimeError('ROOT451 actual independent Native AR saved before capture')
                if len(ar['actual_tokens'])!=16 or len(reference['tokens'])!=16 or len(ar['unrelated_decode_checks'])!=15:
                    raise RuntimeError('ROOT451 full16-step true AR and15 decode checks')
                eq=ar['actual_tokens']==reference['tokens']
                stable=all(all(x is True for x in c.values())for c in ar['unrelated_decode_checks'])
                if ar['all16_tokens_exact']!=eq or ar['all_unrelated_decode_checks_exact']!=stable:raise RuntimeError('ROOT451 actual AR arithmetic')
                expected_AR=expected_AR and eq and stable
            if v['all_native_AR_tokens_exact']!=expected_AR:raise RuntimeError('ROOT451 independent Native AR aggregation')
            for c in v['cases']:
                wanted=['initial-full40']if c['family']=='initial'else['continuation-chunk40','continuation-full167']
                if sorted(c['ordinary_prefill_checks'])!=sorted(wanted):raise RuntimeError('ROOT451 ordinary independent prompt comparisons')
                for comparison in c['ordinary_prefill_checks'].values():
                    if comparison['hidden_exact']!=(comparison['hidden_actual_sha256']==comparison['hidden_reference_sha256']):raise RuntimeError('ROOT451 real native hidden hashes')
                    if comparison['tokens_exact']!=(c['actual_tokens']==comparison['tokens_reference']):raise RuntimeError('ROOT451 real native token comparison')
                    if len(comparison['query_KV'])!=128 or any(q['equal']!=(q['actual_sha256']==q['reference_sha256'])for q in comparison['query_KV']):raise RuntimeError('ROOT451 actual128 Native KV hashes')
                    if comparison['all128_KV_exact']!=all(q['equal']is True for q in comparison['query_KV']):raise RuntimeError('ROOT451 Native128-KV aggregation')
            exact=all(all(c[k]is True for k in(*KEYS,'post_replay_hidden_stable','post_replay_prefill128_KV_exact','post_replay_decode_hidden_exact','post_replay_decode_tokens_exact','post_replay_decode128_KV_exact'))for c in v['cases'])
            if exact!=v['all9_cases_math_exact']:raise RuntimeError('ROOT451 actual9-case aggregation')
        for index,name in enumerate(CASES):
            values=[v['cases'][index]for v in r['by_target_rank']]
            expected_gates=[dict(rank=v['rank'],name=name,exact_before=all(v['cases'][index][k]is True for k in KEYS))for v in r['by_target_rank']]
            if any(c['actual_all3_pre_timing_gate']!=expected_gates for c in values):raise RuntimeError('ROOT451 independently reconstructed actual all3 numerical gates')
            cost=r['paired_max_rank_costs'][name]
            if any(c['timing_skipped']is not None for c in values):
                if cost is not None:raise RuntimeError('ROOT451 no retained timing for a numerical mismatch')
                continue
            base=[max(c['device_event_ms']['baseline'][j]for c in values)for j in range(9)]
            variant=[max(c['device_event_ms'][name][j]for c in values)for j in range(9)]
            if any(not math.isfinite(x)or x<=0 for x in(*base,*variant))or base!=cost['baseline_samples_ms']or variant!=cost['variant_samples_ms']or statistics.median(base)!=cost['baseline_p50_ms']or statistics.median(variant)!=cost['variant_p50_ms']:
                raise RuntimeError('ROOT451 actual paired max-rank event arithmetic')
        if r['all3_all9_cases_math_exact']!=all(v['all9_cases_math_exact']for v in r['by_target_rank']):raise RuntimeError('ROOT451 all3 aggregation')
        if r['all3_native_AR_tokens_exact']!=all(v['all_native_AR_tokens_exact']for v in r['by_target_rank']):raise RuntimeError('ROOT451 Native all3 aggregation')
        reports.append(r)
    if any(r['by_target_rank']!=reports[0]['by_target_rank']for r in reports):raise RuntimeError('ROOT451 same retained evidence all3')
    for rank in range(4):
        v=json.loads((directory/f'rank{rank}-numeric.json').read_text())
        if v['rank']!=rank or v['model_dtype']!='torch.bfloat16'or v['current_profile_unchanged']is not True or v['new_profile_installed']is not False:raise RuntimeError('ROOT451 original protected BF16 lifecycle')
        if rank and(v['independent_prefills']!=72 or v['all72_prefixes_protected']is not True):raise RuntimeError('ROOT451 all72 original committed prefixes intact')
    completed=json.loads((directory/'completed.json').read_text())
    if sorted(completed['replies'],key=lambda r:r[1])!=[['calibrated',rank,dict(numeric_cases=1,performance=False)]for rank in range(4)]:raise RuntimeError('ROOT451 all4 completed roles')
    return(dict(status='retained_actual451_Native_prefill_and_AR_comparisons',reports=reports,
        all3_all9_cases_math_exact=reports[0]['all3_all9_cases_math_exact'],
        all3_native_AR_tokens_exact=reports[0]['all3_native_AR_tokens_exact'],
        production_permission=False,whole_workload_AR_qualified=False,Native_flags_assigned=False,
        Raw=None,Goodput=None,adopted=False),'one_persistent_graph_N16_known_prefill_plus_independent_Native_16_AR_only',dict(NN_reexecuted_by_HOST=False))
