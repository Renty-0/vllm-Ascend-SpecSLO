"""Independent HOST audit of a terminal V108 run; no Torch/model/NPU import."""
import argparse
import copy
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tarfile
import traceback

BASE=Path('/root/SpecSLO/budget-calibration-20261015')
P=BASE/'tree-scoped-validation-cost-v108'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition,message):
    if not condition:raise RuntimeError(message)


def artifact(path):
    path=Path(path);return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)


def archive_matches(path,expected):
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers()
        require(len(members)==len(expected) and len({m.name for m in members})==len(members)
            and all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members),
            'Source archive membership differs')
        actual={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}
    require(actual==expected,'Source archive member bytes differ')
    return dict(**artifact(path),members=len(actual),exact_member_bytes=True)


def shell_rows(rows):
    if type(rows) is dict:result=dict(rows)
    else:
        require(type(rows) is list and all(type(r) is dict and 'path' in r and 'sha256' in r for r in rows),
            'Malformed actual sourced-shell records')
        result={r['path']:r['sha256'] for r in rows}
        require(len(result)==len(rows),'Duplicate actual sourced-shell paths')
    require(all(type(path) is str and path.startswith('/') and type(value) is str
        and re.fullmatch('[0-9a-f]{64}',value) for path,value in result.items()),'Malformed sourced-shell path/SHA')
    return result


def run(tag):
    require(shell_rows({'/actual/shell':'a'*64})==shell_rows([dict(path='/actual/shell',sha256='a'*64)]),
        'HOST shell dict/list normalizer positive failed')
    for bad in ({'/actual/shell':'invalid'},[dict(path='/actual/shell',sha256='a'*64)]*2):
        try:shell_rows(bad)
        except RuntimeError:pass
        else:raise RuntimeError('HOST shell malformed/duplicate negative was accepted')
    sys.path.insert(0,str(P))
    import check_v108
    from audit_scoped_validation_v108 import audit_worker_reports
    from source_snapshot_v108 import runtime_outputs
    receipt_path=P/f'receipt-{tag}.json';receipt=json.loads(receipt_path.read_text())
    directory=P/'actual'/tag
    require(receipt['status']=='complete' and receipt['errors']==[] and receipt['child']['exit_code']==0
        and receipt['local_execution_complete'] is True and receipt['owned_remaining']==[],
        'Terminal complete/worker exit/owned cleanup not proved')
    for label in ('before','after'):
        state=receipt['devices_'+label]
        require(state['all_idle'] is state['four_910B2'] is True
            and {r[0] for r in state['parsed_hardware']}=={2,5,6,7}
            and state['hardware']['exit_code']==0
            and set(state['processes'])=={'2','5','6','7'}
            and all(r['exit_code']==0 and r['idle'] is True for r in state['processes'].values()),
            'Retained hardware/cleanup snapshot invalid')
    for label in ('pre','post'):
        verifier=receipt[label+'_verifier']
        require(verifier['exit_code']==0 and sha(verifier['log'])==verifier['log_sha256']
            and Path(verifier['log']).stat().st_size==verifier['log_bytes'],'Actual verifier log differs')
    require(receipt['experimental_sources_unchanged'] is receipt['external_sources_unchanged'] is True,
        'Supervisor did not retain unchanged sources')
    ready=check_v108.require_ready()
    freeze=json.loads(check_v108.FREEZE.read_text());cpu=json.loads(check_v108.CPU.read_text())
    require(receipt['external_input_hashes']==freeze['external_files']
        and all(sha(path)==digest for path,digest in receipt['source_hashes'].items()),'Actual source map differs')
    require(cpu['tests_run']==31 and cpu['failures']==cpu['errors']==0
        and cpu['input_hashes']==cpu['input_hashes_after']==freeze['files']
        and cpu['external_input_hashes']==cpu['external_input_hashes_after']==freeze['external_files'],
        'Actual CPU31 source proof differs')
    require(sha(P/'CPU_V108.log')==freeze['CPU_log_sha256'],'Actual passing CPU log differs')
    expected={str(Path(path).relative_to(P)):value for path,value in receipt['source_hashes'].items()}
    require(sha(receipt['source_snapshot'])==receipt['source_snapshot_sha256'],'Actual source tar SHA differs')
    actual_archive=archive_matches(receipt['source_snapshot'],expected)
    prepared=dict(freeze['files']);prepared.update(freeze['retained_attempt_files'])
    prepared.update({str(path):sha(path) for path in (check_v108.FREEZE,check_v108.CPU,P/'CPU_V108.log')})
    prepared_archive=archive_matches(P/'PREPARED_SOURCE_ARCHIVE_V108.tar.gz',
        {str(Path(path).relative_to(P)):value for path,value in prepared.items()})
    pre_path=BASE/'budget-diagnostics/ROOT_V108_PRE_NPU_SOURCE_AND_HOST_AUDIT.json'
    pre=json.loads(pre_path.read_text())
    require(pre['all_checks_passed'] is True and pre['NPU_started'] is False
        and pre['CPU_sha256']==sha(check_v108.CPU) and pre['freeze_sha256']==sha(check_v108.FREEZE)
        and pre['archive_sha256']==prepared_archive['sha256'],'Original ROOT prelaunch proof differs')
    runtime=BASE/'serial-v1/runtime.sh'
    sourced=re.findall(r'^source (/[^ \n]+)',runtime.read_text(),re.MULTILINE)
    expected_shells={path:sha(path) for path in sourced}
    require(len(sourced)==2 and shell_rows(receipt['actual_sourced_runtime_shells_before'])
        ==shell_rows(receipt['actual_sourced_runtime_shells_after'])
        ==shell_rows(pre['actual_sourced_runtime_shells_before'])==expected_shells,
        'Actually sourced runtime shell before/after/source identities differ')
    require(receipt['runtime_aliases_before']==receipt['runtime_aliases_after'],'Runtime alias records changed')
    for row in receipt['runtime_aliases_after']:
        path=Path(row['path'])
        require(path.is_symlink() and path.exists() and str(path.resolve())==row['resolved']
            and str(path.readlink())==row['link'],'Actual retained alias target differs')
    require(runtime_outputs(P)==receipt['runtime_outputs'],'Known strict CANN fusion diagnostic differs')
    paths=[directory/f'rank{rank}-FULL96-scopedmemo.json' for rank in range(4)]
    reports=[json.loads(path.read_text()) for path in paths]
    contract=audit_worker_reports(reports)
    completed=json.loads((directory/'completed.json').read_text())
    require(completed['replies']==[['calibrated',rank,{'rows':0,'timings':30}] for rank in range(4)]
        and completed['replies']==receipt['completed_replies'],'Original four-rank completion replies differ')
    exited=json.loads((directory/'worker-exit.json').read_text())
    require(exited['closed'] is exited['all_exited'] is True,'Actual original workers did not all exit')
    raw_mc2=[]
    for report in reports[1:]:
        for q in report['qualification']:
            for kind,field in (('candidate','MC2_candidate'),('reference','MC2_reference')):
                path=directory/f"rank{report['rank']}-MC2-p{q['prefix']}-{q['mode']}-{kind}.json"
                require(json.loads(path.read_text())==q[field],'Raw MC2 sidecar/report differs')
                raw_mc2.append(artifact(path))
    negative=[]
    for kind in ('memo_hits','Native_discovery','MC2_cold_positive','production_tokens'):
        bad=copy.deepcopy(reports)
        if kind=='memo_hits':
            next(r for r in bad[1]['cost'] if r['mode']=='scopedmemo')['memo_delta']['repeated_checks_reused']=62
        elif kind=='Native_discovery':bad[2]['traces'][0][0]['native_deltas']['aclgraph_capture_attempts']=1
        elif kind=='MC2_cold_positive':bad[1]['qualification'][0]['MC2_candidate']['dispatch']['after']['native_epilogue_success']=0
        else:bad[3]['cost'][0]['tokens'][0]=(bad[3]['cost'][0]['tokens'][0]+1)%151936
        try:audit_worker_reports(bad)
        except (RuntimeError,KeyError,ValueError):negative.append(dict(kind=kind,rejected=True))
        else:raise RuntimeError('Actual-evidence HOST counterexample was incorrectly accepted: '+kind)
    summaries=[]
    for rank in (1,2,3):
        report=reports[rank]
        for prefix in (127,128,255):
            rows={mode:[r for r in report['cost'] if r['prefix']==prefix and r['mode']==mode] for mode in ('original','scopedmemo')}
            medians={mode:statistics.median(r['whole_target_call_complete_ms'] for r in group) for mode,group in rows.items()}
            paired=[next(r['whole_target_call_complete_ms'] for r in rows['scopedmemo'] if r['sample']==s)
                -next(r['whole_target_call_complete_ms'] for r in rows['original'] if r['sample']==s) for s in range(5)]
            summaries.append(dict(rank=rank,prefix=prefix,samples_per_mode=5,median_ms=medians,
                scopedmemo_minus_original_ms=medians['scopedmemo']-medians['original'],
                original_over_scopedmemo_ratio=medians['original']/medians['scopedmemo'],
                paired_scopedmemo_minus_original_ms=paired,
                memo_faster_pairs=sum(value<0 for value in paired),
                timed_scoped_updates=21,timed_original_checker_calls=21,timed_reused_checks=1323))
    memo_totals=[]
    for rank in (1,2,3):
        all_rows=[row for trace in reports[rank]['traces'] for row in trace]+reports[rank]['cost']
        scoped=[row for row in all_rows if row['mode']=='scopedmemo']
        totals={key:sum(row['memo_delta'][key] for row in scoped) for key in ('task_lists','original_checks','repeated_checks_reused')}
        require(totals==dict(task_lists=21,original_checks=21,repeated_checks_reused=1323),'Actual memo update totals differ')
        memo_totals.append(dict(rank=rank,**totals))
    require(not any(name=='torch' or name.startswith(('torch_npu','vllm_ascend.spec_decode.pearl.native_')) for name in sys.modules),
        'HOST actual auditor unexpectedly imported model/Torch/NPU runtime')
    return dict(receipt=artifact(receipt_path),worker_reports=list(map(artifact,paths)),CPU=artifact(check_v108.CPU),
        CPU_log=artifact(P/'CPU_V108.log'),freeze=artifact(check_v108.FREEZE),ROOT_prelaunch=artifact(pre_path),
        actual_source_archive=actual_archive,prepared_source_archive=prepared_archive,
        actual_static_source_files=len(freeze['files']),actual_external_source_files=len(freeze['external_files']),
        actual_sourced_runtime_shells=expected_shells,strict_CANN_fusion_diagnostics=receipt['runtime_outputs'],
        original_child_exit_code=0,original_verifiers_pre_post=[0,0],historical_four_card_idle_before_after=True,
        owned_remaining=[],current_NPU_state_queried=False,worker_exit=artifact(directory/'worker-exit.json'),
        original_elapsed_seconds=receipt['elapsed_seconds'],original_private_bytes=receipt['private_bytes'],
        all_four_final_reports_complete=True,actual_qualification_records=18,actual_whole_target_cost_samples=90,
        actual_zero_discovery_timed_calls=36,actual_new_target_keys_per_target_rank=4,
        original_ALL96_hidden_and_ALL64_query_KV_device_gates_executed=True,
        full_hidden_KV_tensor_values_retained_for_independent_HOST_recomparison=False,
        MC2_raw_sidecars=raw_mc2,MC2_cold_positive_replay_zero_allowance_and_fresh_reference_positive_checked=True,
        whole_target_report_contract=contract,actual_per_rank_median_and_paired_samples=summaries,
        actual_memo_counts_in_two_zero_traces_and_five_samples=memo_totals,HOST_actual_evidence_negative_checks=negative,
        decision='Omit scoped validation memo from V113: the saved HOST checks were actually reused, but paired whole-target latency has no consistent improvement.',
        decision_scope='Three dependent prefix views of16 genuine FastV5 B5 trees, M96/S16/FULL9, five paired samples per prefix/mode; not isolated kernel time or online Goodput.',
        integration_recommended_for_V113=False,statistical_significance_claimed=False,
        allN_qualified=False,whole_workload_qualified=False,formal_Goodput=False,independent_AR_history_qualified=False,
        Native_source_or_runtime_files_modified=False,NPU_initialized_by_auditor=False,Torch_imported_by_auditor=False,
        actual_readiness_identity=ready['identity'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True)
    parser.add_argument('--audit-id',default='');args=parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag),'Invalid actual tag')
    require(not args.audit_id or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.audit_id),'Invalid audit ID')
    suffix='_'+args.audit_id if args.audit_id else ''
    output=BASE/'budget-diagnostics'/f'ROOT_V108_ACTUAL_{args.tag}{suffix}_AUDIT.json'
    require(not output.exists(),'Refusing to overwrite prior actual audit')
    record=dict(schema='ROOT.V108.independent-actual-scopedmemo-audit.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_HOST_python_argv=sys.orig_argv,script=artifact(__file__),status='failed',errors=[])
    if args.audit_id:
        record['preserved_prior_failed_audit']=artifact(BASE/'budget-diagnostics'/f'ROOT_V108_ACTUAL_{args.tag}_AUDIT.json')
        record['preserved_prior_auditor_source']=artifact(Path(__file__).with_name('analyze_v108_actual_scopedmemo_attempt1_retained.py'))
        record['bounded_HOST_schema_correction']='Accept exact path-to-SHA ROOT preaudit dictionary and exact supervisor path/sha256 list; malformed values/duplicate paths still reject. No substantive numerical gate changed.'
    try:record.update(run(args.tag));record['status']='passed'
    except BaseException:record['errors'].append(traceback.format_exc())
    with output.open('x') as stream:stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(path=str(output),sha256=sha(output),status=record['status'],errors=record['errors'],
        integration_recommended_for_V113=record.get('integration_recommended_for_V113'))))
    return 0 if record['status']=='passed' else 2


if __name__=='__main__':raise SystemExit(main())
