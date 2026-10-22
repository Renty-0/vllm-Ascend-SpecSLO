"""Whole-model independent reference and balanced max-rank device costs."""
from pathlib import Path
import hashlib
import json
import math
import statistics


def audit(directory):
    directory=Path(directory);reports=[]
    if list(directory.glob('rank*-last-error.json')):raise RuntimeError('ROOT455 no worker errors')
    kinds=('control160','qualified160','current120')
    keys=('hidden_exact','tokens_exact','all128_query_KV_exact')
    for rank in(1,2,3):
        r=json.loads((directory/f'rank{rank}-whole455-final.json').read_text())
        if r['rank']!=rank or[v['rank']for v in r['by_target_rank']]!=[1,2,3]:raise RuntimeError('ROOT455 actual three roles')
        for v in r['by_target_rank']:
            if(v['physical_M160'],v['physical_S160'],v['real160_queries'])!=(160,32,160)or v['real160_owners']!=list(range(32)):
                raise RuntimeError('ROOT455 actual32 fully occupied input')
            if 160 not in v['actual_chained_rows']or 160 in v['actual_deferred_rows']:raise RuntimeError('ROOT455 actual constructor fully-flushed160')
            if v['model_configuration_flags_modified']is not False or v['Native_flags_assigned']is not False or v['production_permission']is not False:
                raise RuntimeError('ROOT455 reference changes no model flags or serving permissions')
            for key in('original_unfused_reference_uses_real_modules_and_weights','model_parameter_SHA_unchanged','mc2_manifest_unchanged_during_epoch','ordinary_prompt_prefill_unchanged'):
                if v[key]is not True:raise RuntimeError('ROOT455 original reference and bound weights '+key)
            wb=json.loads((directory/f"rank{v['rank']}-whole455-weights-before.json").read_text())
            wa=json.loads((directory/f"rank{v['rank']}-whole455-weights-after.json").read_text())
            if not wb or wb!=wa:raise RuntimeError('ROOT455 real parameter fingerprints')
            e=v['eager']
            if len(e['query_KV'])!=128 or[e['query_KV'][i]['index']for i in range(128)]!=list(range(128)):raise RuntimeError('ROOT455 all128 model KV arrays')
            if e['all128_query_KV_exact']!=all(q['equal']is True for q in e['query_KV']):raise RuntimeError('ROOT455 real eager128 arithmetic')
            if e['hidden_exact']is True and e['hidden_max_abs']!=0.:raise RuntimeError('ROOT455 exact eager hidden difference zero')
            if set(v['initial_graph_checks'])!=set(kinds)or set(v['final_graph_checks'])!=set(kinds):raise RuntimeError('ROOT455 all3 graph families')
            allchecks=[*v['initial_graph_checks'].values(),*v['post_timing_checks'],*v['final_graph_checks'].values()]
            exact=all(all(row[k]is True for k in keys)for row in allchecks)
            if v['all_graph_checks_exact']!=exact:raise RuntimeError('ROOT455 full graph check aggregation')
            if v['timing_skipped']is None:
                if any(len(v['timings_ms'][kind])!=9 for kind in kinds)or len(v['post_timing_checks'])!=27:
                    raise RuntimeError('ROOT455 balanced9 events per mode and27 last-timed replay checks')
                if {(row['repeat'],row['name'])for row in v['post_timing_checks']}!={(i,kind)for i in range(9)for kind in kinds}:
                    raise RuntimeError('ROOT455 complete paired post checks')
            elif any(v['timings_ms'].values())or v['post_timing_checks']:
                raise RuntimeError('ROOT455 no timing claim after an actual initial numerical mismatch')
        expected=[dict(rank=v['rank'],exact=all(v['eager'][k]is True for k in keys)and all(all(row[k]is True for k in keys)for row in v['initial_graph_checks'].values()))for v in r['by_target_rank']]
        if any(v['actual_all3_pre_timing_gate']!=expected for v in r['by_target_rank']):raise RuntimeError('ROOT455 independent all3 pre-timing gate')
        if any((v['timing_skipped']is None)!=all(g['exact']for g in expected)for v in r['by_target_rank']):raise RuntimeError('ROOT455 actual timing permission arithmetic')
        exact=all(all(v['eager'][k]is True for k in keys)and v['all_graph_checks_exact']is True for v in r['by_target_rank'])
        if r['all3_eager_and_graph_math_exact']!=exact:raise RuntimeError('ROOT455 whole-model all3 aggregation')
        costs=r['actual_whole_model_costs']
        if any(v['timing_skipped']is not None for v in r['by_target_rank']):
            if costs is not None:raise RuntimeError('ROOT455 negative math cannot claim model costs')
        else:
            samples={kind:[max(v['timings_ms'][kind][i]for v in r['by_target_rank'])for i in range(9)]for kind in kinds}
            if any(not math.isfinite(x)or x<=0 for values in samples.values()for x in values):raise RuntimeError('ROOT455 actual finite positive events')
            median={kind:statistics.median(values)for kind,values in samples.items()}
            if costs['paired_max_rank_samples_ms']!=samples or costs['p50_ms']!=median:raise RuntimeError('ROOT455 actual max-rank medians')
            if costs['qualified160_over_unfused160_speedup']!=median['control160']/median['qualified160']or costs['fully_occupied_request_rate160_over120']!=(32/median['qualified160'])/(24/median['current120']):
                raise RuntimeError('ROOT455 actual whole-model and occupied-request rate arithmetic')
        reports.append(r)
    if any(v['by_target_rank']!=reports[0]['by_target_rank']for v in reports[1:]):raise RuntimeError('ROOT455 all3 identical source evidence')
    for rank in range(4):
        v=json.loads((directory/f'rank{rank}-numeric.json').read_text())
        if v['rank']!=rank or v['model_dtype']!='torch.bfloat16'or v['current_profile_unchanged']is not True or v['new_profile_installed']is not False:
            raise RuntimeError('ROOT455 constructor-bound BF16 epoch unchanged afterward')
        if rank and(v['independent_prefills']!=72 or v['all72_prefixes_protected']is not True):raise RuntimeError('ROOT455 original72 prefixes intact')
    completed=json.loads((directory/'completed.json').read_text())
    if sorted(completed['replies'],key=lambda r:r[1])!=[['calibrated',rank,dict(numeric_cases=1,performance=False)]for rank in range(4)]:raise RuntimeError('ROOT455 all4 roles complete')
    return(dict(status='retained_current455_qualified160_whole_model_reference_and_cost',reports=reports,
        all3_eager_and_graph_math_exact=reports[0]['all3_eager_and_graph_math_exact'],actual_whole_model_costs=reports[0]['actual_whole_model_costs'],
        production_permission=False,whole_workload_AR_qualified=False,Native_flags_assigned=False,Raw=None,Goodput=None,adopted=False),
        'actual_constructor_profile160_unfused_whole_model_and_current120_device_cost_only',dict(NN_reexecuted_by_HOST=False))
