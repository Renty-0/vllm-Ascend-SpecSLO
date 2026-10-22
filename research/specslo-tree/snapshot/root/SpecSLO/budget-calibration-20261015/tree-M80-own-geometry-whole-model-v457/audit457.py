"""Whole-model independent reference and balanced max-rank device costs."""
from pathlib import Path
import hashlib
import json
import math
import statistics


def audit(directory):
    directory=Path(directory);reports=[]
    if list(directory.glob('rank*-last-error.json')):raise RuntimeError('ROOT457 no worker errors')
    kinds=('control80','qualified80','current120')
    keys=('hidden_exact','tokens_exact','all128_query_KV_exact')
    for rank in(1,2,3):
        r=json.loads((directory/f'rank{rank}-whole457-final.json').read_text())
        if r['rank']!=rank or[v['rank']for v in r['by_target_rank']]!=[1,2,3]:raise RuntimeError('ROOT457 actual three roles')
        for v in r['by_target_rank']:
            if(v['physical_M80'],v['physical_S80'],v['real80_queries'])!=(80,16,80)or v['real80_owners']!=list(range(16)):
                raise RuntimeError('ROOT457 actual16 fully occupied80 input')
            if v['global_active_ceiling']!=64:raise RuntimeError('ROOT457 unchanged global64')
            if 80 not in v['actual_chained_rows']or 80 in v['actual_deferred_rows']:raise RuntimeError('ROOT457 actual constructor fully-flushed80')
            if v['model_configuration_flags_modified']is not False or v['Native_flags_assigned']is not False or v['production_permission']is not False:
                raise RuntimeError('ROOT457 reference changes no model flags or serving permissions')
            for key in('original_unfused_reference_uses_real_modules_and_weights','model_parameter_SHA_unchanged','mc2_manifest_unchanged_during_epoch','ordinary_prompt_prefill_unchanged'):
                if v[key]is not True:raise RuntimeError('ROOT457 original reference and bound weights '+key)
            wb=json.loads((directory/f"rank{v['rank']}-whole457-weights-before.json").read_text())
            wa=json.loads((directory/f"rank{v['rank']}-whole457-weights-after.json").read_text())
            if not wb or wb!=wa:raise RuntimeError('ROOT457 real parameter fingerprints')
            e=v['eager']
            if len(e['query_KV'])!=128 or[e['query_KV'][i]['index']for i in range(128)]!=list(range(128)):raise RuntimeError('ROOT457 all128 model KV arrays')
            if e['all128_query_KV_exact']!=all(q['equal']is True for q in e['query_KV']):raise RuntimeError('ROOT457 real eager128 arithmetic')
            if e['hidden_exact']is True and e['hidden_max_abs']!=0.:raise RuntimeError('ROOT457 exact eager hidden difference zero')
            if set(v['initial_graph_checks'])!=set(kinds)or set(v['final_graph_checks'])!=set(kinds):raise RuntimeError('ROOT457 all3 graph families')
            allchecks=[*v['initial_graph_checks'].values(),*v['post_timing_checks'],*v['final_graph_checks'].values()]
            exact=all(all(row[k]is True for k in keys)for row in allchecks)
            if v['all_graph_checks_exact']!=exact:raise RuntimeError('ROOT457 full graph check aggregation')
            if v['timing_skipped']is None:
                if any(len(v['timings_ms'][kind])!=9 for kind in kinds)or len(v['post_timing_checks'])!=27:
                    raise RuntimeError('ROOT457 balanced9 events per mode and27 last-timed replay checks')
                if {(row['repeat'],row['name'])for row in v['post_timing_checks']}!={(i,kind)for i in range(9)for kind in kinds}:
                    raise RuntimeError('ROOT457 complete paired post checks')
            elif any(v['timings_ms'].values())or v['post_timing_checks']:
                raise RuntimeError('ROOT457 no timing claim after an actual initial numerical mismatch')
        expected=[dict(rank=v['rank'],exact=all(v['eager'][k]is True for k in keys)and all(all(row[k]is True for k in keys)for row in v['initial_graph_checks'].values()))for v in r['by_target_rank']]
        if any(v['actual_all3_pre_timing_gate']!=expected for v in r['by_target_rank']):raise RuntimeError('ROOT457 independent all3 pre-timing gate')
        if any((v['timing_skipped']is None)!=all(g['exact']for g in expected)for v in r['by_target_rank']):raise RuntimeError('ROOT457 actual timing permission arithmetic')
        exact=all(all(v['eager'][k]is True for k in keys)and v['all_graph_checks_exact']is True for v in r['by_target_rank'])
        if r['all3_eager_and_graph_math_exact']!=exact:raise RuntimeError('ROOT457 whole-model all3 aggregation')
        costs=r['actual_whole_model_costs']
        if any(v['timing_skipped']is not None for v in r['by_target_rank']):
            if costs is not None:raise RuntimeError('ROOT457 negative math cannot claim model costs')
        else:
            samples={kind:[max(v['timings_ms'][kind][i]for v in r['by_target_rank'])for i in range(9)]for kind in kinds}
            if any(not math.isfinite(x)or x<=0 for values in samples.values()for x in values):raise RuntimeError('ROOT457 actual finite positive events')
            median={kind:statistics.median(values)for kind,values in samples.items()}
            if costs['paired_max_rank_samples_ms']!=samples or costs['p50_ms']!=median:raise RuntimeError('ROOT457 actual max-rank medians')
            if costs['qualified80_over_unfused80_speedup']!=median['control80']/median['qualified80']or costs['fully_occupied_request_rate80_over120']!=(16/median['qualified80'])/(24/median['current120']):
                raise RuntimeError('ROOT457 actual whole-model and occupied-request rate arithmetic')
        if costs is not None and costs['current120_over_qualified80_frame_speedup']!=costs['p50_ms']['current120']/costs['p50_ms']['qualified80']:
            raise RuntimeError('ROOT457 actual frame speed arithmetic')
        for v in r['by_target_rank']:
            if v['actual_profile_sha256']!='ad9e0648810ed697679ee84b8bbc2939ea14bfb687d0215d27f403ad5526db1c':raise RuntimeError('ROOT457 unchanged original MC2 profile')
            if not isinstance(v['cross_M120_observation']['different_greedy_rows'],list):raise RuntimeError('ROOT457 retain cross-M120 differences')
        reports.append(r)
    if any(v['by_target_rank']!=reports[0]['by_target_rank']for v in reports[1:]):raise RuntimeError('ROOT457 all3 identical source evidence')
    for rank in range(4):
        v=json.loads((directory/f'rank{rank}-numeric.json').read_text())
        if v['rank']!=rank or v['model_dtype']!='torch.bfloat16'or v['current_profile_unchanged']is not True or v['new_profile_installed']is not False:
            raise RuntimeError('ROOT457 constructor-bound BF16 epoch unchanged afterward')
        if rank and(v['independent_prefills']!=72 or v['all72_prefixes_protected']is not True):raise RuntimeError('ROOT457 original72 prefixes intact')
    completed=json.loads((directory/'completed.json').read_text())
    if sorted(completed['replies'],key=lambda r:r[1])!=[['calibrated',rank,dict(numeric_cases=1,performance=False)]for rank in range(4)]:raise RuntimeError('ROOT457 all4 roles complete')
    return(dict(status='retained_current457_qualified80_whole_model_reference_and_cost',reports=reports,
        all3_eager_and_graph_math_exact=reports[0]['all3_eager_and_graph_math_exact'],actual_whole_model_costs=reports[0]['actual_whole_model_costs'],
        production_permission=False,whole_workload_AR_qualified=False,Native_flags_assigned=False,Raw=None,Goodput=None,adopted=False),
        'actual_original_profile80_same_geometry_unfused_reference_and_current120_cost_only',dict(NN_reexecuted_by_HOST=False))
