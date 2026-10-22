"""Actual six-row M160 operator qualification, never an end-to-end result."""
from pathlib import Path
import hashlib
import json
import math


def percentile(values,p):
    values=sorted(values);position=(len(values)-1)*p/100
    lo=math.floor(position);hi=math.ceil(position)
    if lo==hi:return values[lo]
    fraction=position-lo
    return values[lo]*(1.-fraction)+values[hi]*fraction


def audit(directory):
    directory=Path(directory);reports=[]
    if list(directory.glob('rank*-last-error.json')):raise RuntimeError('ROOT454 actual worker error')
    source=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/examples/measure_specslo_rank3_reduce_add_rmsnorm.py')
    qualifier_sha=hashlib.sha256(source.read_bytes()).hexdigest()
    expected=[(layer,kind)for layer in(0,31,63)for kind in('attention','down')]
    for rank in(1,2,3):
        r=json.loads((directory/f'rank{rank}-M160454-final.json').read_text())
        if r['rank']!=rank or[r['rank']for r in r['by_target_rank']]!=[1,2,3]:raise RuntimeError('ROOT454 three actual target roles')
        for v in r['by_target_rank']:
            if v['TP_rank']!=v['rank']-1 or v['original_qualifier_sha256']!=qualifier_sha:raise RuntimeError('ROOT454 real subgroup rank and unchanged qualifier')
            if v['actual_init_noops']!=1 or v['actual_destroy_noops']!=1 or v['payload_loader_calls']!=6:raise RuntimeError('ROOT454 owned TP3 lifecycle and six real payload loads')
            if v['model_weights_serialized']is not False or v['new_profile_installed']is not False or v['Native_flags_assigned']is not False or v['production_permission']is not False:
                raise RuntimeError('ROOT454 only live-input operator qualification')
            capture=v['real32_capture']
            if(capture['physical_M'],capture['physical_S'],capture['real_requests'],capture['real_query_rows'])!=(160,32,32,160):raise RuntimeError('ROOT454 all160 queries are real')
            if capture['owner_ids']!=list(range(32))or capture['additional_owner_ids']!=list(range(24,32))or capture['additional_owners_separately_prefilled']is not True:
                raise RuntimeError('ROOT454 actual independent32 cache owners')
            if len(capture['original24_slots'])!=120 or len(capture['additional40_slots'])!=40 or len(set(capture['original24_slots']+capture['additional40_slots']))!=160:
                raise RuntimeError('ROOT454 all160 real cache writes unique')
            if capture['ordinary_prompt_prefill_unchanged']is not True or capture['tensors_serialized']is not False:raise RuntimeError('ROOT454 original prompt path and no tensor dumps')
            if [(x['layer'],x['projection'])for x in capture['captured']['entries']]!=expected:raise RuntimeError('ROOT454 all actual layer/projection inputs')
            wb=json.loads((directory/f"rank{v['rank']}-M160454-weights-before.json").read_text())
            wa=json.loads((directory/f"rank{v['rank']}-M160454-weights-after.json").read_text())
            if not wb or wb!=wa or v['model_parameters_SHA_unchanged']is not True:raise RuntimeError('ROOT454 immutable model parameter bytes')
            if [(x['layer'],x['projection'])for x in v['results']]!=expected:raise RuntimeError('ROOT454 no omitted qualification job')
        if any(v['results']!=r['by_target_rank'][0]['results']for v in r['by_target_rank'][1:]):
            # Per-rank input hashes and pointers deliberately differ. Only
            # shared measured paths, decisions and counts must agree.
            for index in range(6):
                keys=('layer','projection','M','K','N','measurement','original_qualifier_admission_rejection','production_row_accepted','changed_input_replays')
                reference={k:r['by_target_rank'][0]['results'][index][k]for k in keys}
                if any({k:v['results'][index][k]for k in keys}!=reference for v in r['by_target_rank'][1:]):raise RuntimeError('ROOT454 collective job decision consensus')
        reports.append(r)
    if any(r['by_target_rank']!=reports[0]['by_target_rank']for r in reports[1:]):raise RuntimeError('ROOT454 same all3 retained evidence')
    summary=[]
    for row in reports[0]['by_target_rank'][0]['results']:
        path=directory/Path(row['measurement']).relative_to(directory)
        q=json.loads(path.read_text());k=3072 if row['projection']=='attention'else 8576
        if q['status']!='measured'or q['shape']!=dict(m=160,k=k,n=5120):raise RuntimeError('ROOT454 exact measured160 shape')
        meta=q['metadata'];correct=q['correctness'];samples=q['samples'];s=q['summary']
        if(meta['execution_mode'],meta['operations_per_graph'],meta['tensor_parallel_size'],meta['chained_flush'])!=('graph',1,3,True):raise RuntimeError('ROOT454 actual standalone flushed graph measurement')
        if meta['operator']!='torch.ops._C_ascend.allreduce_add_rmsnorm_chained':raise RuntimeError('ROOT454 original native-MatMul epilogue operator')
        if correct['changed_input_replays']!=1024 or correct['changed_input_delta']!=0.015625:raise RuntimeError('ROOT454 original changed-input gate')
        a,b=samples['baseline_latency_ms'],samples['fused_latency_ms']
        if len(a)!=31 or len(b)!=31 or any(not math.isfinite(v)or v<=0 for v in a+b):raise RuntimeError('ROOT454 all31 original positive samples')
        ap,bp=percentile(a,95),percentile(b,95)
        if s['baseline_p95_ms']!=ap or s['fused_p95_ms']!=bp or s['p95_speedup']!=ap/bp:raise RuntimeError('ROOT454 independently recomputed original p95')
        exact=correct['exact_all_ranks']is True and correct['changed_input_exact_all_ranks']is True
        exact=exact and all(correct[key]==0. for key in('max_abs_norm','max_abs_added','changed_input_max_abs_norm','changed_input_max_abs_added'))
        accepted=exact and bp*1.02<=ap
        if row['production_row_accepted']!=accepted or(row['original_qualifier_admission_rejection']is None)!=accepted:
            raise RuntimeError('ROOT454 actual unmodified numerical/p95 acceptance')
        profile=path.parent/'single-shape-qualified-profile.json'
        if profile.exists()!=accepted:raise RuntimeError('ROOT454 qualified profile emitted only by original gate')
        if accepted:
            p=json.loads(profile.read_text());measurements=p['measurements']
            if p['status']!='measured'or p['minimum_speedup']!=1.02 or len(measurements)!=1 or(measurements[0]['m'],measurements[0]['k'],measurements[0]['n'])!=(160,k,5120):raise RuntimeError('ROOT454 actual strict single-shape artifact')
        summary.append(dict(layer=row['layer'],projection=row['projection'],M=160,K=k,N=5120,
            exact_original_and1024_changed_inputs=exact,baseline_p95_ms=ap,fused_p95_ms=bp,p95_speedup=ap/bp,
            original_gate_accepted=accepted,raw=str(path),raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            profile=str(profile)if accepted else None))
    accepted_all=all(v['original_gate_accepted']for v in summary)
    if any(r['all6_original_operator_gates_accepted']!=accepted_all for r in reports):raise RuntimeError('ROOT454 all6 aggregate')
    for rank in range(4):
        v=json.loads((directory/f'rank{rank}-numeric.json').read_text())
        if v['rank']!=rank or v['model_dtype']!='torch.bfloat16'or v['current_profile_unchanged']is not True or v['new_profile_installed']is not False:raise RuntimeError('ROOT454 original BF16 lifecycle')
        if rank and(v['independent_prefills']!=72 or v['all72_prefixes_protected']is not True):raise RuntimeError('ROOT454 all72 committed prefixes intact')
    completed=json.loads((directory/'completed.json').read_text())
    if sorted(completed['replies'],key=lambda r:r[1])!=[['calibrated',rank,dict(numeric_cases=1,performance=False)]for rank in range(4)]:raise RuntimeError('ROOT454 all4 completed roles')
    return(dict(status='retained_current454_real32_M160_exact_operator_qualification',reports=reports,summary=summary,
        all6_original_operator_gates_accepted=accepted_all,production_permission=False,whole_workload_AR_qualified=False,
        Native_flags_assigned=False,Raw=None,Goodput=None,adopted=False),
        'genuine_N32_B4_inputs_M160_native_MatMul_epilogue_only',dict(NN_reexecuted_by_HOST=False))
