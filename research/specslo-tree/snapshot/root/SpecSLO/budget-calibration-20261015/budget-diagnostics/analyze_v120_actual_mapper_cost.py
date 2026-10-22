"""Independent terminal-only HOST audit of actual V120 raw cost evidence."""
from __future__ import annotations
import argparse, ast, copy, hashlib, importlib.abc, importlib, json, math, re, statistics, sys, tarfile, traceback
from datetime import datetime,timezone
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');P=B/'tree-qualified44-cost-stagefix-v120';D=B/'budget-diagnostics'
TAG='existing44-realpartition-stagefix-cost-v1'
CPU_SHA='e6a98f301df7da003201785b6db9c71279ad27647b94890a9c9ab41f9f6d7a66'
FREEZE_SHA='98fa4da92c3c07d4c59e11ceb5b9bd03126ae6d8d8bd17c47795aa44bc66be72'
ARCHIVE_SHA='ac2f05ad598af06ed0418d5084ca938b2d2260532043bf276b06a1cf770e2c1a'
PRE_SHA='6001f3f5051c016175288da71fd0215773ecc086deafd439cdf3dca16c2b50d8'


def need(value,message):
    if not value:raise ValueError('V120 external: '+message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024**2),b''):h.update(part)
    return h.hexdigest()
def desc(path):
    path=Path(path).resolve();return dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
def read(path):
    value=json.loads(Path(path).read_bytes());finite(value);return value
def finite(value):
    if type(value) is dict:
        for v in value.values():finite(v)
    elif type(value) is list:
        for v in value:finite(v)
    elif type(value) is float:need(math.isfinite(value),'nonfinite JSON')
    else:need(value is None or type(value) in (bool,int,str),'unknown JSON type')
def archive(path,expected):
    wanted={str(Path(n).relative_to(P)):w for n,w in expected.items()}
    with tarfile.open(path,'r:gz') as source:
        members=source.getmembers();need(len(members)==len(wanted)==len({m.name for m in members})
            and {m.name for m in members}==set(wanted)
            and all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members),'archive exact regular membership')
        for member in members:need(hashlib.sha256(source.extractfile(member).read()).hexdigest()==wanted[member.name],'archive member bytes')
    return dict(**desc(path),members=len(wanted),all_member_bytes_recomputed=True)


class NoDeviceImports(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST auditor forbids '+fullname)


def provenance(receipt):
    cp=P/'CPU_RESULT_V120.json';fp=P/'PREPARED_SOURCE_HASHES_V120.json';cpu=read(cp);freeze=read(fp)
    need(sha(cp)==CPU_SHA and sha(fp)==FREEZE_SHA and cpu['status']=='passed' and cpu['tests_run']==12
        and cpu['failures']==cpu['errors']==0 and cpu['initialization_attempts']==[]
        and cpu['NPU_initialized'] is cpu['model_initialized'] is False and freeze['CPU_sha256']==CPU_SHA
        and cpu['input_hashes']==cpu['input_hashes_after']==freeze['files']
        and cpu['external_input_hashes']==cpu['external_input_hashes_after']==freeze['external_files'],'actual12 HOST/source identity')
    need(len(freeze['files'])==15 and len(freeze['external_files'])==89 and freeze['retained_attempt_files']=={},'sparse source counts')
    for path,wanted in {**freeze['files'],**freeze['external_files']}.items():need(sha(path)==wanted,'source changed: '+path)
    expected={**freeze['files'],str(cp):CPU_SHA,str(fp):FREEZE_SHA}
    need(receipt['source_hashes']==expected and receipt['external_input_hashes']==freeze['external_files'],'actual supervisor source maps')
    prepared=P/'PREPARED_SOURCE_ARCHIVE_V120.tar.gz';need(sha(prepared)==ARCHIVE_SHA,'prepared archive SHA')
    p=archive(prepared,expected);actual=archive(receipt['source_snapshot'],expected)
    need(actual['sha256']==receipt['source_snapshot_sha256'],'actual tar SHA')
    pre=D/'ROOT_V120_PRE_NPU_SOURCE_AND_HOST_AUDIT.json';need(sha(pre)==PRE_SHA and read(pre)['status']=='passed','ROOT prelaunch audit')
    from check_v120 import source_contract
    contract=source_contract()
    need(contract['exact_timing_stage_delta'] is True and contract['original_numerical_source_AST_unchanged'] is True
        and contract['parent_NPU_qualification_inherited'] is False,'parent exact AST/numerical scope')
    need(receipt['actual_sourced_runtime_shells_before']==receipt['actual_sourced_runtime_shells_after']
        and receipt['runtime_aliases_before']==receipt['runtime_aliases_after'],'actual runtime source/aliases changed')
    for row in receipt['actual_sourced_runtime_shells_before']:need(sha(row['path'])==row['sha256'],'CANN/ATB source shell changed')
    for side in ('pre','post'):
        verifier=receipt[side+'_verifier'];need(verifier['exit_code']==0 and verifier['argv'][:4]==['bash',str(B/'serial-v1/runtime.sh'),'python','-B'],'actual retained verifier')
        need((P/f'{TAG}-{side}-verifier.log').is_file(),'actual verifier log')
    return dict(actual_HOST_tests=12,CPU=desc(cp),freeze=desc(fp),local_sources=15,external_SHA_inputs=89,
        prepared_archive=p,actual_archive=actual,ROOT_prelaunch=desc(pre),exact_parent_source_contract=contract,
        all_local_source_bytes_archived=True,external_sources_SHA_checked_but_not_all_bytes_in_this_local_tar=True,
        runtime_import_event_history_available=False)


def cleanup(receipt,root):
    need(receipt['status']=='complete' and receipt['errors']==[] and receipt['local_execution_complete'] is True
        and receipt['child']['exit_code']==0 and receipt['owned_remaining']==[]
        and receipt['experimental_sources_unchanged'] is receipt['external_sources_unchanged'] is True
        and receipt['physical_cards']==[2,5,6,7],'terminal/owned/source completion')
    for side in ('before','after'):
        state=receipt['devices_'+side];parsed=[(int(c),name) for c,name in re.findall(r'^\|\s*(\d+)\s+(\S+)\s*\|',state['hardware']['output'],re.M) if not name.isdecimal()]
        need(sorted(parsed)==[(p,'910B2') for p in (2,5,6,7)] and state['all_idle'] is state['four_910B2'] is True
            and state['hardware']['exit_code']==0 and state['hardware']['timed_out'] is False
            and set(state['processes'])=={'2','5','6','7'}
            and all(v['exit_code']==0 and v['idle'] is True and re.search(r'No process in device\.|No running processes found',v['output'],re.I) for v in state['processes'].values()),'retained four-card hardware/idle snapshot')
    exited=read(root/'worker-exit.json');need(exited==dict(closed=True,all_exited=True),'all4 workers closed')
    expected=[['calibrated',r,dict(rows=0,timings=0 if r==0 else 61)] for r in range(4)]
    need(read(root/'completed.json')['replies']==receipt['completed_replies']==expected,'original4 completion replies')
    return dict(passed=True,owned_remaining=[],retained_pre_post_four_card_idle=True,worker_exit=desc(root/'worker-exit.json'),completion=desc(root/'completed.json'))


def independently_recompute(reports,scope):
    targets={r['rank']:r for r in reports if r['rank']!=0};cs=scope.cases();outputs=[];tokenchecks=[];prefixchecks=[]
    for case in cs:
        name=case['name'];modes=scope.modes(case);arms={};count=5 if len(modes)==2 else 1
        for mode in modes:
            samples=[]
            for sample in range(count):
                by_rank={rank:next(r for r in report['cost'] if r['case']==name and r['mode']==mode and r['sample']==sample)['whole_target_complete_ms'] for rank,report in targets.items()}
                need(all(type(v) in (float,int) and math.isfinite(v) and v>0 for v in by_rank.values()),'raw actual timing')
                samples.append(dict(sample=sample,TPmax_ms=max(by_rank.values()),by_rank_ms=by_rank))
            m,s,dummy=scope.envelope(case,mode)
            arms[mode]=dict(M=m,S=s,dummy_queries=list(dummy),samples=samples,TPmax_median_ms=statistics.median(v['TPmax_ms'] for v in samples))
        value=dict(case=name,N=case['N'],Q=case['Q'],budgets=case['budgets'],arms=arms,actual_delivered_g=None)
        if len(modes)==2:
            ratios=[arms['new']['samples'][i]['TPmax_ms']/arms['old']['samples'][i]['TPmax_ms'] for i in range(5)]
            value.update(all5_paired_new_over_old=ratios,median_paired_new_over_old=statistics.median(ratios),
                ratio_of_TPmax_medians=arms['new']['TPmax_median_ms']/arms['old']['TPmax_median_ms'],
                relative_reduction_from_medians=1-arms['new']['TPmax_median_ms']/arms['old']['TPmax_median_ms'])
        outputs.append(value)
        for rank,report in targets.items():
            banks={v['bank_id']:v for v in report['bank_evidence']}
            names=[name+'/'+epoch for epoch in ('candidate','independent-reference','zero0','zero1','cost')]
            prefixchecks.append(dict(rank=rank,case=name,all_five_saved128_prefix_hashes_bitidentical=all(banks[n]['prefix_hashes']==banks[names[0]]['prefix_hashes'] for n in names),full_prefix_tensor_bytes_retained=False))
            q={(r['mode'],r['owner_order']):r for r in report['qualification'] if r['case']==name}
            for mode in modes:
                for kind in ('greedy','hidden'):
                    for order in ('canonical','reverse'):
                        row=q[mode,order];records=row[kind+'_phases'];existing=row[kind+'_existing_before']
                        if kind=='greedy':need([r['runtime_validated_before'] for r in records]==([True]*3 if existing else [None,False,True]),'actual Native pre-flag cold/changed/restored sequence')
                        need(records[0]['runtime_validated_after'] is existing and all(r['runtime_validated_after'] is True for r in records[1:]),'actual Native post-flag sequence')
            if len(modes)==2:
                for order in ('canonical','reverse'):
                    old=q['old',order]['greedy_phases'][0]['real_target_query_token_ids'];new=q['new',order]['greedy_phases'][0]['real_target_query_token_ids']
                    differences=[dict(query=i,old=a,new=b) for i,(a,b) in enumerate(zip(old,new,strict=True)) if a!=b]
                    tokenchecks.append(dict(rank=rank,case=name,owner_order=order,real_queries=case['Q'],exact_greedy_token_differences=differences,
                        cross_M_hidden_or_KV_equality_compared=False))
                    if order=='canonical':need(next(v for v in report['cross_M_comparison'] if v['case']==name)['real_greedy_query_tokens_equal'] is (not differences),'stored canonical crossM marker differs')
    return outputs,tokenchecks,prefixchecks


def run():
    need(sys.version_info[:2]==(3,11),'retained3.11 HOST only')
    receipt_path=P/f'receipt-{TAG}.json';receipt=read(receipt_path);root=P/'actual'/TAG
    guard=NoDeviceImports();sys.meta_path.insert(0,guard);sys.path[:0]=[str(P),str(B/'tree-qualified44-cost-v114')]
    try:
        prov=provenance(receipt);clean=cleanup(receipt,root)
        import audit_mapper_v120 as original
        import mapper_scope_v120 as scope
        import mc2_inventory_contract as MC2
        reports=[read(root/f'rank{r}-mapper-cost.json') for r in range(4)]
        need(receipt['report_paths']==[str(root/f'rank{r}-mapper-cost.json') for r in range(4)],'actual report paths/rank order')
        frozen_audit=original.audit_worker_reports(reports)
        startup=MC2.audit(P/'startup-progress')
        need(all(r['MC2_static']==read(P/'startup-progress'/f'MC2-preflight-rank{r["rank"]}.json') for r in reports),'raw startup MC2 equality')
        startup_artifacts=[desc(P/'startup-progress'/f'MC2-preflight-rank{r}.json') for r in range(4)]
        cost,cross,prefix=independently_recompute(reports,scope)
        for row,known in zip(cost,frozen_audit['cost'],strict=True):
            for mode,values in row['arms'].items():need(values['TPmax_median_ms']==known['arms'][mode]['median_ms']
                and [v['TPmax_ms'] for v in values['samples']]==known['arms'][mode]['rankmax_synchronous_call_ms'],'independent critical-TP arithmetic')
        for report in reports:
            for row in [report['initial_memory'],report['final_memory'],*report['qualification_memory']]:
                need(row['blocks']==544 and 60*1024**3<=row['total']<=64*1024**3 and row['free']>=6*1024**3
                    and row['reserved']<=54*1024**3 and row['peak']<=54*1024**3,'actual544/54GiB/6GiB gate')
            expected={k:0 for k in scope.NATIVE_COUNTERS}
            if report['rank']:
                for key in ('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays','aclgraph_target_runtime_validation_calls'):expected[key]=6
            need(report['final_Native']==expected,'minimal exact6key natural Native final inventory/counters')
        negative=[]
        for kind in ('hidden','Nativezero','tokens','MC2','signature'):
            altered=copy.deepcopy(reports)
            if kind=='hidden':altered[1]['qualification'][0]['all64_physical_query_KV']=False
            elif kind=='Nativezero':altered[1]['cost'][0]['Native']['delta'][scope.NATIVE_COUNTERS[0]]=1
            elif kind=='tokens':altered[1]['cost'][0]['real_target_query_token_ids'][0]+=1
            elif kind=='MC2':altered[1]['cost'][0]['MC2']['delta']['fallback']=1
            else:altered[1]['cost'][0]['actual_signature_before']=altered[1]['cost'][0]['actual_signature_after']
            try:original.audit_worker_reports(altered)
            except (ValueError,RuntimeError,KeyError,AssertionError):negative.append(kind)
            else:raise ValueError('Tampered raw evidence accepted: '+kind)
        need('torch' not in sys.modules and 'torch_npu' not in sys.modules,'HOST only no Torch import')
        return dict(status='passed',errors=[],receipt=desc(receipt_path),actual_receipt_elapsed_seconds=receipt['elapsed_seconds'],
            source_provenance=prov,cleanup=clean,all4_reports=[desc(root/f'rank{r}-mapper-cost.json') for r in range(4)],
            original_frozen_HOST_report_validator_reexecuted=True,MC2_startup=startup,MC2_raw_startup_artifacts=startup_artifacts,
            scopes=dict(actual_target_ranks=[1,2,3],sameM_partitions_per_target=26,independent_sameM_references_per_target=26,
                separately_created_prefix_banks_per_target=35,all_physical_hidden_and_all64_query_KV_actual_matchers=True,
                canonical_and_reverse_partitions=True,Native_F_T_F_and_changed_restore_flags=True,two_complete13arm_zero_sweeps=True,
                all21Native_zero_per_zero_prime_and_cost=True,unified_original_seal=True,all64_task_update_delta=[1,64,0],
                timed_pairs=30,FULL165_S38_negative_calls=1,actual_timed_calls_per_target=61,actual_greedy_keys=3,actual_standard_hidden_keys=3),
            costs=cost,cross_M_real_query_tokens=cross,saved_prefix_hash_comparisons=prefix,meaningful_tamper_negatives=negative,
            original_teacher_forced_candidate_inputs=True,genuine_draft_proposals=False,actual_delivered_g=None,
            sameM_full_hidden_KV_tensor_bytes_retained=False,HOST_independent_tensor_numerical_recomparison=False,
            cross_M_hidden_KV_equality_qualified=False,arbitrary_token_or_path_exact_qualification=False,
            local_component_numerical_scope_passed=True,production_mapper_installed_or_admitted=False,
            full463_qualified=False,formal_Goodput=False,actual_online_GP=None,NPU_initialized=False,model_initialized=False,
            actual_audit_torch_imported=False,statistical_scope='5 paired calls per causal case; three TP ranks are synchronous observations, not independent trials; FULL negative has one unchanged call')
    finally:sys.meta_path.remove(guard)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args();start=datetime.now(timezone.utc).isoformat()
    try:value=run()
    except BaseException:value=dict(status='audit_failed',errors=[traceback.format_exc()],NPU_initialized=False,formal_Goodput=False)
    value.update(schema='V120.external-independent-actual-cost-HOST-audit.v1',actual_audit_argv=sys.argv,actual_python=sys.executable,
        runtime_wrapper_used=False,started_UTC=start,finished_UTC=datetime.now(timezone.utc).isoformat(),auditor_source=desc(__file__))
    with Path(args.output).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=value['status'],output=args.output,errors=value['errors'])))
    return int(value['status']!='passed')

if __name__=='__main__':raise SystemExit(main())
