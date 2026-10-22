"""Retain genuine one-frame math and paired full-target events; no service."""
from pathlib import Path
import json,math,statistics

def audit(directory):
    directory=Path(directory);reports=[]
    if list(directory.glob('rank*-last-error.json')):raise RuntimeError('ROOT441 no retained worker errors')
    for rank in(1,2,3):
        r=json.loads((directory/f'rank{rank}-whole-target-layout441.json').read_text())
        if r['rank']!=rank or r['actual_input_M']!=120 or r['physical_S']!=24 or r['TP_world_size']!=3 or r['Native_flags_assigned']is not False or r['production_permission']is not False or r['all_parameter_storages_restored']is not True:raise RuntimeError('ROOT441 exact actual one-frame scope and restored data')
        if [x['rank']for x in r['by_target_rank']]!=[1,2,3]:raise RuntimeError('ROOT441 all3 actual target roles')
        for record in r['by_target_rank']:
            if record['changed_layer_count']!=64 or record['all_original_parameter_storages_restored']is not True or record['all_MC2_row_weight_bindings_unchanged']is not True or len(record['numerical']['query_KV'])!=128:raise RuntimeError('ROOT441 all64 layouts/all128 KV arrays and bindings')
            for key in('ND','NZ'):
                values=record['whole_target_forward_event_ms'][key]
                if len(values)!=9 or any(type(x)not in(int,float)or not math.isfinite(x)or x<=0 for x in values):raise RuntimeError('ROOT441 actual9 positive paired full-forward samples')
        for name in('ND','NZ'):
            samples=[max(x['whole_target_forward_event_ms'][name][i]for x in r['by_target_rank'])for i in range(9)]
            if samples!=r['max_rank_event_costs'][name]['actual_max_rank_samples_ms']or statistics.median(samples)!=r['max_rank_event_costs'][name]['p50_ms']:raise RuntimeError('ROOT441 independent actual max-rank arithmetic')
        reports.append(r)
    if any(r['by_target_rank']!=reports[0]['by_target_rank']for r in reports):raise RuntimeError('ROOT441 cross-rank retained evidence identical')
    for rank in range(4):
        r=json.loads((directory/f'rank{rank}-numeric.json').read_text())
        if r['rank']!=rank or r['model_dtype']!='torch.bfloat16'or r['new_profile_installed']is not False or r['current_profile_unchanged']is not True:raise RuntimeError('ROOT441 original protected real-frame capture')
        if rank and(r['independent_prefills']!=72 or r['all72_prefixes_protected']is not True or r['measurement']['all_original_parameter_storages_restored']is not True):raise RuntimeError('ROOT441 original72 prefixes protected')
    completed=json.loads((directory/'completed.json').read_text())
    if sorted(completed['replies'],key=lambda r:r[1])!=[['calibrated',rank,dict(numeric_cases=1,performance=False)]for rank in range(4)]:raise RuntimeError('ROOT441 all4 completed actual roles')
    result=dict(status='retained_actual441_whole_target_layout_math_and_device_cost',reports=reports,all3_hidden_exact=reports[0]['all3_hidden_exact'],all3_query_KV_exact=reports[0]['all3_query_KV_exact'],p50_speedup=reports[0]['p50_speedup'],whole_workload_AR_qualified=False,Native_flags_assigned=False,production_permission=False,Raw=None,Goodput=None,adopted=False)
    return result,'one_real_target_frame_only_not_serving_qualification',dict(NN_reexecuted_by_HOST=False)
