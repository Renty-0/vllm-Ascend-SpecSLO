"""Read actual local matrix events and descriptors, never infer Goodput."""
import json,math
from pathlib import Path

def audit(directory):
    directory=Path(directory);reports=[]
    if list(directory.glob('rank*-last-error.json')):raise RuntimeError('ROOT440 no retained role errors')
    for rank in range(1,4):
        path=directory/f'rank{rank}-linear-layout-cost440.json';r=json.loads(path.read_text())
        if r['rank']!=rank or r['TP_world_size']!=3 or r['physical_cards']!=[2,5,6,7]or r['production_permission']is not False or r['Native_flags_assigned']is not False or r['Raw']is not None or r['Goodput']is not None:raise RuntimeError('ROOT440 actual scope, no production qualification')
        if [x['projection']for x in r['roles']]!=['attention','gate_up','down']:raise RuntimeError('ROOT440 all3 real layer0 projection types')
        for role in r['roles']:
            records=role['by_target_rank']
            if [x['rank']for x in records]!=[1,2,3]:raise RuntimeError('ROOT440 all3 actual target ranks')
            for x in records:
                if x['actual_input_rows']!=120 or x['dtype']!='torch.bfloat16'or x['tensors_serialized']is not False:raise RuntimeError('ROOT440 actual immutable BF16 payloads')
                for name in('ND120','NZ120','NDpad128','NZpad128'):
                    samples=x['device_event_ms_per_local_matmul'][name]
                    if len(samples)!=9 or any(type(v)not in(int,float)or not math.isfinite(v)or v<=0 for v in samples)or x['comparison'][name]['finite']is not True:raise RuntimeError('ROOT440 all actual balanced positive device-event samples')
        reports.append(r)
    if any(r['roles']!=reports[0]['roles']for r in reports[1:]):raise RuntimeError('ROOT440 identical actual TP3 gathered report')
    exit_record=json.loads((directory/'worker-exit.json').read_text())
    wait=json.loads((directory/'worker-exit-wait.json').read_text())
    for rank in range(4):
        numeric=json.loads((directory/f'rank{rank}-numeric.json').read_text())
        if numeric['rank']!=rank or numeric['model_dtype']!='torch.bfloat16' or numeric['new_profile_installed']is not False or numeric['current_profile_unchanged']is not True or numeric['operator_qualification_only']is not True or numeric['full463']is not False:raise RuntimeError('ROOT440 exact original385 genuine capture lifecycle')
        if rank and (numeric['independent_prefills']!=72 or numeric['all72_prefixes_protected']is not True or numeric['measurement']['actual_projections']!=3):raise RuntimeError('ROOT440 original72 prefix protections and all3 genuine projections')
    completed=json.loads((directory/'completed.json').read_text())
    expected=[['calibrated',rank,dict(numeric_cases=1,performance=False)]for rank in range(4)]
    if sorted(completed['replies'],key=lambda r:r[1])!=expected:raise RuntimeError('ROOT440 complete original four-role replies')
    result=dict(status='PASS_ACTUAL440_LOCAL_LINEAR_LAYOUT_DIAGNOSTIC_ONLY',
        reports=reports,worker_exit=exit_record,worker_exit_wait=wait,
        performance_scope='only local layer0 matmul device costs; not complete target, new weights policy, or full463',
        Raw=None,Goodput=None,production_permission=False,Native_flags_assigned=False,adopted=False)

    return result, 'local_matrix_operator_cost_only_no_service_qualification', dict(NPU_model_outputs_recomputed_by_HOST=False)
