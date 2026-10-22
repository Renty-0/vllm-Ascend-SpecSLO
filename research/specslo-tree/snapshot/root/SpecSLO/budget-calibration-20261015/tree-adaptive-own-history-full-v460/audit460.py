"""Original438 full checks, actual adaptive geometry, complete new own-AR."""
from pathlib import Path
from types import FunctionType,SimpleNamespace
import gzip,hashlib,json
import audit209,audit_base425,audit425,audit428,audit430,audit433,audit432,audit434,audit438
from projection425 import project_function
from bucket460 import valid_record
from history439 import verify as verify_normal_history

def bind(fn,**values):
    value=FunctionType(fn.__code__,dict(fn.__globals__,**values),fn.__name__,fn.__defaults__,fn.__closure__)
    value.__kwdefaults__=fn.__kwdefaults__;value.__annotations__=dict(fn.__annotations__);return value

def original_geometry_audit(directory,windows):
    body=project_function(audit209.original_audit,(
        ('from mapper209 import expected_geometry','from geometry377 import envelope as expected_geometry'),
        ('m,s,padding=expected_geometry(n,q,f)',"m,s,padding=expected_geometry(n,q,f,physical=c['physical_M'])"),
    ))
    # Original active64 window validation remains the original425 binding.
    from window425 import windows as checked_windows
    body=bind(body,windows=checked_windows)
    return body(directory,windows)

def base_audit():
    return project_function(audit_base425.audit,(
        ("'from empty425 import audit_trace'","'from admission_audit432 import audit_trace'"),
        ('    if projected==audit_source:',
         "    projected=projected.replace('actual_empty_cohorts=', 'actual_admission_cohorts=').replace('empty_active_cohorts=', 'actual_admission_cohorts=')\n    if projected==audit_source:"),
        ('from projection255 import plan as physical_plan','from physical377 import plan as physical_plan'),
        ("projection=physical_plan(ends,call['real_query_count'])","projection=physical_plan(ends,call['real_query_count'],call['physical_M'])"),
        ("and row['M'] == call['physical_M'] == 120","and row['M'] == call['physical_M'] and row['M'] in(80,120)"),
        ("and row['attention_segments']==24 and row['primary_side_groups']==row['N']+branches<=24",
         "and row['attention_segments']==row['M']//5 and row['primary_side_groups']==row['N']+branches<=row['M']//5"),
        ("and row['fixed_KV_extent']==[2560]*24","and _ROOT460_valid_record(row)"),
        ("and row['actual_q_partition']==list(range(5,121,5))","and row['actual_q_partition']==list(range(5,row['M']+1,5))"),
        ('from audit_reference425 import audit as original_reference_audit','from audit_reference460 import audit as original_reference_audit'),
        ("step['metadata']['backend'] == 'static_FULL_q5_M120_S24_KV2560'",
         "step['metadata']['backend'] == 'independent_static_FULL_q5_actual_M80_or_M120_KV2560'"),
        ("step['metadata']['tree_mask_shape'] == [24, 1, 10, 4096]",
         "step['metadata']['tree_mask_shape'] == [step['physical_M']//5, 1, 10, 4096]"),
    ),dict(_ROOT460_valid_record=valid_record,original_with_new_geometry=original_geometry_audit))

def constructed():
    source=project_function(audit425.audit,(
        ("all(value == normal_snapshots[0] for value in normal_snapshots)\n         and normal_audits[0]['overflow_calls'] > 0",
         '_ROOT460_verify_normal_history(normal_snapshots,normal_audits)'),
    ),dict(projected_original_audit=base_audit,_ROOT460_verify_normal_history=verify_normal_history))
    current428=bind(audit428.audit,original=SimpleNamespace(audit=source))
    current430=bind(audit430.audit,original=SimpleNamespace(audit=current428))
    current433=bind(audit433.audit,original=SimpleNamespace(audit=current430))
    current432=bind(audit432.audit,constructed=lambda:(current433,base_audit))
    current434=bind(audit434.audit,original=SimpleNamespace(audit=current432))
    return bind(audit438.audit,original=current434)

def audit(directory,windows):
    value=constructed()(directory,windows)
    directory=Path(directory).resolve();heads=[]
    from oracle459 import cases
    from dispatch377 import audit_calls
    for rank in range(4):
        report=json.loads((directory/f'rank{rank}-local-diagnostic.json').read_text())
        proof=report['ROOT257_actual_geometry']['adaptive_same_instance_proof']
        if not(proof['cases']==693 and proof['same_model_instance']is True and proof['global_active_ceiling']==64
               and proof['Native_flags_assigned']is False and proof['parameter_SHA_before']==proof['parameter_SHA_after']):
            raise RuntimeError('ROOT460 actual current693 proof and original parameter bytes')
        rank_heads=[]
        for bank in proof['banks']:
            path=Path(bank['path']).resolve()
            if not path.is_relative_to(directory)or hashlib.sha256(path.read_bytes()).hexdigest()!=bank['sha256']:
                raise RuntimeError('ROOT460 actual bank evidence location and bytes')
            data=json.loads(path.read_text());rows=data['math']['cases']
            if len(rows)!=231 or bank['cases']!=231:raise RuntimeError('ROOT460 all231 actual bank cases')
            if not rank:
                if data['bank']is not None or data['actual_calls']or not all(r is None for r in rows):
                    raise RuntimeError('ROOT460 draft has no target NN claim')
                continue
            prefixes=[[0]*n for n in data['prefix_lengths']]
            audit_calls(cases(),data['actual_calls'],prefixes)
            if not(data['bank']['independent_neural_prefills']==72 and data['bank']['all_triplet_roots_equal']is True
                   and data['bank']['all_triplet_prefix_KV_equal']is True and data['math']['all_independent_prefixes_unchanged']is True):
                raise RuntimeError('ROOT460 all72 independent actual prefix banks')
            for case,row in zip(cases(),rows,strict=True):
                if not(row['case']==case and row['exact_greedy']is True and row['hidden_bitwise_equal']is True
                       and row['KV_bitwise_equal']is True and row['all128_KV_original_matcher_passed']is True
                       and row['graph_eager_KV_bitwise_equal']is True and row['candidate_KV_copied']is False
                       and row['candidate_tokens']==row['reference_tokens']):
                    raise RuntimeError('ROOT460 all actual same-physical node gates')
                if not all(frame['physical_queries']==case['M']for frame in row['independent_reference_frames']):
                    raise RuntimeError('ROOT460 actual independent reference physical family')
                rank_heads.append(row['candidate_tokens'])
        if rank:heads.append(rank_heads)
    if not heads[0]==heads[1]==heads[2]:raise RuntimeError('ROOT460 all3 target693 predictions agree')
    from geometry_history460 import build
    traces=[]
    for path in sorted(directory.glob('**/rank1-*-actual-trace.json.gz')):
        trace=json.loads(gzip.decompress(path.read_bytes()))
        geometry=build(trace,adaptive_required=True)
        traces.append(dict(path=str(path),SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),requests=geometry['requests'],target_calls=geometry['target_calls']))
    if not traces:raise RuntimeError('ROOT460 actual completed adaptive traces required')
    return dict(value,ROOT460_current_additional_own_physical_cases=693,ROOT460_all3_new_node_predictions_agree=True,
                ROOT460_actual_adaptive_geometry_histories=traces,ROOT460_all_original_gates_and_current_own_AR_retained=True,
                global_active_limit=64,whole_workload_AR_qualified=True)
