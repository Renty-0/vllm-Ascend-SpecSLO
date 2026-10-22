#!/usr/bin/env python3
"""Terminal V102 failure forensics; never changes its receipt or source.

This writes a failed-run evidence audit, never a complete cost/qualification
result. No torch, subprocess, model or NPU is imported/launched.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True
P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('_V102_independent_complete_auditor_helpers',P/'analyze_v102_actual_packed_eager01_cost.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def packed_records(report):
    rows=report['numerical'];expected=[(c,ch) for c in a.cases() for ch in (False,True,False)]
    a.require([(q['case'],q['changed_input_variant']) for q in rows]==expected,'exact18 completed fresh packed numeric records')
    for q in rows:
        c=q['case'];g=a.validate_case(c)
        a.true_fields(q,a.NUMERIC_FLAGS+('original_before_tail_real_first','root_head_outside_graph',
            'immediate_first_hidden_observer_numeric_only'),'actual packed independent hidden/28KV/tokens/confidence/ledger/restoration')
        a.require(q['method']=='packed_seed_tail3' and q['qualification_inherited'] is False and q['timing_qualified'] is False
            and q['changed']==q['changed_input_variant'] and q['first_M']==g['first_M'] and q['tail_M']==g['head_M'],'actual packed source/geometry/unchanged gates')
        a.require([(k['kind'],k['key']) for k in q['native_keys']]==a.expected_native_keys(c,'packed_seed_tail3'),'actual numeric Native hidden/tail keys')
        for key in q['native_keys']:
            a.require(key['runtime_validated'] is True,'actual original changed-input flag')
            if key['kind']=='draft':a.require(key['validated_real_row_count']>=g['total'],'all real draft rows qualified')
    return [dict(case=c,metadata_scope=a.validate_case(c),actual_changed_input_variants=[False,True,False],
        independent_numeric_records=3,original_sameM_hidden_KV_exact_public_ABI_and_whole_method_restoration_passed=True)
        for c in a.cases()]


def classify_first_error(error,report):
    text=error['error'];a.require(error['stage']=='independent-same-M-allphysical-hidden-KV','actual failing local stage')
    for needle in ('prepare_control','trace_observer.py','padding_reference.py',
        'PA padding must preserve token0/pos0/slot-1/context1 and fresh owner0 table'):
        a.require(needle in text,'first local error retains actual control/observer/padding stack')
    case=report['warmup'][-1]['case'];a.require(case==a.cost_cases()[0] and report['warmup'][-1]['method']=='original_fast4','first original-Fast numerical control context')
    owners=[i for i,r in enumerate(case['normal_repairs']) if r]
    return dict(stage=error['stage'],role_rank=0,actual_method='original_fast4',case=case,
        actual_repair_owner_ids=owners,first_live_repair_owner=owners[0],repair_real_rows=len(owners),repair_physical_M=a.bucket(len(owners),64),
        failing_adapter='trace_observer.run.hidden -> padding_reference.inspect',
        actual_error='PA padding must preserve token0/pos0/slot-1/context1 and fresh owner0 table',
        source_supported_root_cause='Original Native _pad_inputs copies first live metadata row table, here repair owner7. Frozen padding_reference.inspect instead compares against cache owner0. The V65 all-one fixture started with owner0 and did not expose this assumption.',
        raw_padding_tensor_metadata_persisted=False,
        diagnostic_limit='The retained error does not identify which boolean subpredicate failed or dump actual dummy tensors. Source ownership and Native padding rule explain the blocker; no model/KV numerical mismatch was reported at this point.',
        candidate_or_Native_model_failure_proved=False)


def audit_failed(tag):
    a.require(tag.startswith('packed-eager01-') and tag.replace('-','').isalnum(),'bounded failure tag')
    w=a.WORK;receipt_path=w/f'cost-receipt-{tag}.json';raw_receipt=receipt_path.read_bytes();r=json.loads(raw_receipt)
    a.require(r['schema']=='V102.ragged01-packed-cost-receipt.v1' and r['status']=='failed','actual terminal failed receipt required')
    a.require(r['child']['exit_code']!=0 and r['owned_remaining']==[] and len(r['errors'])==2,'child failed and independent post-output classification failed')
    a.require('Calibration failed; see calibration.log' in r['errors'][0] and 'Duplicate CANN diagnostic key' in r['errors'][1],'retain distinct primary child failure and final fusion parser failure')
    freeze,pre,cpu,ready=a.source_and_HOST(r)
    d=w/'context-cost-sweep-v1'/tag;reports=[a.read(d/f'rank{i}-packed01-cost.json') for i in range(4)]
    a.require(a.read(d/'prerequisites.json')==ready,'actual child source readiness bound to ROOT prelaunch evidence')
    coverage=[];evidence={str(receipt_path):a.sha(receipt_path)};packed=None
    for rank,report in enumerate(reports):
        a.require(report['rank']==rank and report['schema']=='V102.ragged01-packed-qualification-and-local-draft-cost.v1' and report['errors'],'actual incomplete report preserves errors')
        counts={key:len(report[key]) for key in ('numerical','control_numerical','protocol','fixed_point','timing','warmup')}
        a.require(counts==dict(numerical=18,control_numerical=0,protocol=0,fixed_point=0,timing=0,warmup=7)
            and report['sealed'] is False and report['graph_sealed'] is False,'exact actual coverage, never inferred complete/sealed')
        seen=packed_records(report)
        if packed is None:packed=seen
        a.require(report['numerical']==reports[0]['numerical'],'all-rank shared actual draft numerical proof agreement')
        a.memory(report['startup_memory'],rank)
        preflight=a.read(d/f'rank{rank}-preflight.json');a.require(len(preflight)==4,'original preflight all-gather')
        a.memory(preflight[rank]['memory'],rank)
        if rank:a.static_mc2(preflight[rank]['MC2'])
        first_six=report['warmup'][:6]
        a.require([(v['case'],v['method']) for v in first_six]==[(c,'packed_seed_tail3') for c in a.cases()],'actual6 packed warmup fixtures before numerical controls')
        for value in report['warmup']:
            a.require(len(value['calls'])==3,'actual unpublished F/T/F warmup calls')
            for out,ch in zip(value['calls'],(False,True,False)):
                if rank:a.require(out is None,'target ranks do not invent draft warmup')
                else:a.public_signature(out,value['case'],value['method'],changed=ch)
        coverage.append(dict(rank=rank,actual_counts=counts,sealed=False,actual_draft_latency_samples=0,
            target_latency_samples=0,numeric_proof_shared_from_actual_draft_rank=True))
        for suffix in ('packed01-cost','packed01-progress','preflight'):
            path=d/f'rank{rank}-{suffix}.json';a.require(path.is_file(),'actual raw bounded evidence retained');evidence[str(path)]=a.sha(path)
    first_path=d/'rank0-last-error.json';first=a.read(first_path);cause=classify_first_error(first,reports[0]);evidence[str(first_path)]=a.sha(first_path)
    a.require(not (d/'completed.json').exists() and not list(d.glob('rank*-MC2-*.json')),'no completed replies or new target MC2 protocol records')
    worker=a.read(d/'worker-exit.json');a.require(worker==dict(closed=True,all_exited=True),'original engine and all4 workers exited')
    a.require(r['pre_verifier']['exit_code']==0 and r['devices_before']['all_idle'] is True,'actual original pre-lock and idle passed')
    absent=[k for k in ('experimental_sources_unchanged','external_sources_unchanged','post_verifier','devices_after','known_runtime_outputs') if k not in r]
    a.require(set(absent)=={'experimental_sources_unchanged','external_sources_unchanged','post_verifier','devices_after','known_runtime_outputs'},'post collector exception skipped later receipt fields; none are silently supplied')
    a.number(r['elapsed_seconds'],'actual child failed duration',positive=True);a.require(r['elapsed_seconds']<2400 and r['private_bytes']<256*1024**2,'original failed run stayed inside bounds')
    runtime=a.top_level(w,set(freeze['files'])|{a.FREEZE},started=datetime.datetime.fromisoformat(r['started_UTC']).timestamp(),completed=receipt_path.stat().st_mtime)
    a.require(len(runtime)==1 and runtime[0]['object_records']==3 and runtime[0]['unique_session_graph_keys']==1,'actual duplicate3/same-session CANN output, not arbitrary source exemption')
    evidence[runtime[0]['path']]=runtime[0]['sha256']
    log_path=w/f'cost-calibration-{tag}.log';text=log_path.read_text()
    a.require('PEARL rank 3 failed' in text and 'original_fast4' in first['error'],'original all-rank error propagation preserved')
    for path in (log_path,w/f'{tag}-pre-verifier.log',d/'config.json',d/'prerequisites.json',d/'worker-exit.json',w/a.FREEZE,w/'CPU_RESULT_V102.json',a.PREAUDIT):evidence[str(path)]=a.sha(path)
    a.require(receipt_path.read_bytes()==raw_receipt and 'torch' not in sys.modules,'failed receipt unchanged; no torch/model imported')
    return dict(schema='ROOT.V102.terminal-failed-raw-forensics.v1',audited_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        audit_of_failure_evidence_passed=True,actual_run_status='failed',receipt_rewritten=False,
        receipt=str(receipt_path),receipt_sha256=hashlib.sha256(raw_receipt).hexdigest(),
        first_local_failure=cause,independent_second_failure='Frozen strict duplicate-key JSON hook rejects valid-format3-rank CANN fusion output; its exception skips original postchecks.',
        actual_raw_coverage=coverage,completed_packed_numerical_fixtures=packed,
        independent_packed_sample_F_T_F_groups_completed=6,original_Fast_numerical_controls_completed=0,
        original_local_protocol_completed=0,complete_zero_traces_completed=0,original_seal_reached=False,
        actual_paired_cost_comparisons=0,actual_rank0_latency_samples=0,target_latency_ms=None,actual_delivered_g=None,
        new_target_MC2_protocol_raw_records=0,original_MC2_static_preflight_three_targets_checked=True,
        local88_source_archive_and_current_bytes_unchanged=True,external620_source_environment_inputs_unchanged=True,
        exact_source_freeze_sha256=a.FREEZE_SHA,exact_CPU_result_sha256=a.CPU_SHA,actual_prepared_HOST_tests=33,
        source_archive=str(r['source_snapshot']),source_archive_sha256=r['source_snapshot_sha256'],
        original_pre_verifier_exit_code=0,original_post_verifier_exit_code=None,
        receipt_devices_after=None,receipt_owned_remaining=[],actual_worker_exit=worker,
        missing_receipt_fields=absent,missing_fields_are_not_zero_or_pass=True,
        later_ROOT_device_observation=dict(source='ROOT task message after independent npu-smi read',
            approximate_UTC='2026-10-16T17:16:00+00:00',all4idle_reported=True,
            occurred_before_subsequent_V103_session=93606,raw_artifact_path=None,
            same_time_as_original_receipt_postcheck=False,independently_requeried_by_this_auditor=False),
        actual_child_elapsed_seconds=r['elapsed_seconds'],last_polled_private_bytes=r['private_bytes'],
        known_runtime_outputs=runtime,raw_evidence_paths_sha256=evidence,
        repair_recommendations=[
            'New source only: inspector validates actual first live owner against authoritative cache table and Native first-live-row padding rule; retain zero token/pos,slot-1,context1,read-only and full-hidden/KV guards. Do not force owner0 or weaken to self-consistency.',
            'Add meaningful original Native CPU metadata test repair owners7..12 with different owner0/7 pages, unchanged legacy inspector failing and corrected inspector passing; malformed table/slot/context must still fail.',
            'New source only: classify exact fusion_result.json using object_pairs preserving3 duplicate top session records with strict session/graph/pass/count schemas; raw hash/mtime retained, no writer attribution; unknown static/code and diagnostic disguise rejected.',
            'Keep V102 failed. Next version reruns missing original controls, all12 protocols, both complete zero traces and original seal before any paired timing; do not inherit missing device qualifications or fabricate cost.'
        ],full_local_qualification_complete=False,production_qualified=False,independent_own_AR_qualified=False,formal_Goodput=False,
        N32_B5_C163_scope='completed local draft numerical boundary only, not C96 online service',
        auditor_sha256=a.sha(__file__),complete_auditor_helper_sha256=a.sha(P/'analyze_v102_actual_packed_eager01_cost.py'),
        actual_HOST_argv=[sys.executable,*sys.argv],torch_imported=False,NPU_launched=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--tag',required=True);args=parser.parse_args()
    try:
        result=audit_failed(args.tag);path=P/'ROOT_V102_ACTUAL_FAILED_PACKED_EAGER01_FORENSICS.json'
        a.require(not path.exists(),'refuse overwriting retained actual failed forensics')
        with path.open('x') as out:out.write(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        print(json.dumps(dict(failure_evidence_audit_passed=True,actual_run_status='failed',forensics=str(path),packed_numerical_records=18,paired_cost_samples=0)))
        return 0
    except (OSError,KeyError,ValueError,RuntimeError,AssertionError,StopIteration) as e:
        print(json.dumps(dict(failure_evidence_audit_passed=False,result_written=False,error=str(e))),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
