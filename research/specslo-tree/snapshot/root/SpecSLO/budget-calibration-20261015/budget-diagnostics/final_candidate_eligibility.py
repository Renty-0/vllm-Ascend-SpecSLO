"""Create a reviewable final-comparison selection only from qualified real data."""
import hashlib,json,math
from pathlib import Path
from final_common_comparison import collect,FIXED_FLOOR,OFFICIAL_FLOOR

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def need(ok,message):
    if ok is not True:raise ValueError(message)

def qualified(receipt,audit,metrics):
    need(receipt['status']=='complete'and receipt['sources_unchanged']is True
         and receipt['devices_after']['all_idle']is True and not receipt['owned_remaining'],
         'terminal complete source-bound candidate with all owned workers gone and four idle')
    for phase in ('pre_verifier','post_verifier'):
        item=receipt[phase]
        need(item['exit_code']==0 and item['actual_result']['status']=='pass',
             'original frozen source/environment check passed before and after')
    need(audit['status']=='terminal_completed_compute_audited'and audit['supervisor_status']=='complete'
         and not audit['errors']and audit['whole_workload_AR_qualified']is True,
         'independent complete clock and whole-workload own-AR audit required')
    reference=audit['independent_complete_own_AR']
    need(reference['same_backend_exact_output_qualified']is True and reference['references_per_target']==463,
         'all463 own histories qualified')
    rows=reference['by_target_rank']
    need([r['rank']for r in rows]==[1,2,3]
         and all(r['comparison']['requests']==r['comparison']['equal_requests']==463
                 and r['comparison']['mismatching_requests']==0
                 and r['comparison']['exact_all_tokens_equal']is True for r in rows),
         'all target ranks independently match every request exactly')
    gp=metrics['all']['Goodput']
    need(type(gp)in(int,float)and math.isfinite(gp)and gp>FIXED_FLOOR and gp>1.5*OFFICIAL_FLOOR,
         'common-clock candidate exceeds fixed gamma4 and historical strong MC2 1.5x floors before final repeats')
    return dict(qualified_for_final_comparison=True,Goodput=gp,Raw=metrics['all']['Raw'],
                historical_fixed_floor=FIXED_FLOOR,historical_strong_MC2_floor=OFFICIAL_FLOOR,
                historical_1p5_threshold=1.5*OFFICIAL_FLOOR,AR_equal_requests=463,
                AR_different_requests=0,final_repeats_still_required=True,goal_achieved=False)

def select(directory,receipt_path,audit_path,terminal_auditor,output):
    directory=Path(directory).resolve();receipt_path=Path(receipt_path);audit_path=Path(audit_path)
    terminal_auditor=Path(terminal_auditor);output=Path(output)
    need(not output.exists(),'fresh final selection artifact')
    receipt=json.loads(receipt_path.read_text());audit=json.loads(audit_path.read_text())
    tag=receipt['tag'];base=directory.parent
    actual=base/'tree-causal80-small-cohort-role-aware-audit-full-v209/actual'/tag
    fulls=list(actual.glob('windows/window*/full463/rank1-full463-actual-trace.json.gz'))
    need(len(fulls)==1,'one actual audited full463 admission arm, never choose a faster arm')
    raw=fulls[0]
    metrics=collect('qualified_dynamic_tree',raw)
    result=qualified(receipt,audit,metrics)
    need(audit['original_receipt_sha256']==sha(receipt_path),'actual audited terminal receipt bytes')
    manifests=list(directory.glob('FREEZE*.json'));archives=list(directory.glob('SOURCE*.tar.gz'));guards=list(directory.glob('guard*.py'))
    need(len(manifests)==len(archives)==len(guards)==1,'one explicitly frozen candidate manifest/archive/guard')
    freeze,archive,guard=manifests[0],archives[0],guards[0]
    need(audit['source_freeze_sha256']==sha(freeze)and audit['source_archive_sha256']==sha(archive),
         'same complete candidate source as independently audited')
    trace=json.loads(__import__('gzip').decompress(raw.read_bytes()))
    need(any(5 in c['budgets']for c in trace['normal_target_calls']),
         'actual real branching was executed, not merely a tree method label')
    need(trace['decode_counters']['spec_rhythm_kv_compaction_rows']>0,
         'actual noncontiguous accepted path KV work retained')
    bindings={str(p):sha(p)for p in [freeze,archive,guard,terminal_auditor,receipt_path,audit_path,raw]}
    payload=dict(schema='SpecSLO.final-qualified-tree-selection.v1',candidate_directory=str(directory),
                 guard_script=str(guard),freeze_manifest=str(freeze),source_archive=str(archive),
                 terminal_auditor_script=str(terminal_auditor),eligible_receipt=str(receipt_path),
                 eligible_audit=str(audit_path),eligible_raw_trace=str(raw),source_and_evidence_hashes=bindings,
                 full_trace_relative_path=str(raw.relative_to(actual)),
                 common_metrics=metrics,eligibility=result,selected_before_final_model_runs=True,
                 original_source_environment_checks=receipt['pre_verifier']['actual_result'])
    with output.open('x')as stream:json.dump(payload,stream,indent=2);stream.write('\n')
    return payload
