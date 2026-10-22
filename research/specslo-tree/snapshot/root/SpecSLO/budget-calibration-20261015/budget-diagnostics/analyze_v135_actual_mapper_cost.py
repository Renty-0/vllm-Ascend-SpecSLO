"""Independent terminal-only HOST audit of actual V135 raw cost evidence."""
from __future__ import annotations
import argparse, ast, copy, hashlib, importlib.abc, importlib, json, math, re, statistics, sys, tarfile, traceback
from datetime import datetime,timezone
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');P=B/'tree-qualified44-largeB3-cost-v135';D=B/'budget-diagnostics'
TAG='existing44-largeB3-four-partitions-v135-1'
CPU_SHA='a8608ce0e0c505b0a2ecece6a234e25fbfb4c27dfdee5cbc717320e67050b604'
FREEZE_SHA='b7bbe0d52843842f1739e17d81007c9cf39d9d6888609db4a2e49947ae580d4a'
ARCHIVE_SHA='4bef1119048c0a3e6bd481d348288825866ba256d490d0b29af45792cbdd0f6f'
PRE_SHA='4d0dc5f51a8772c69ea5d66ba07e6c79bf8df7cf6ee0b81b5f2548d8e2576588'


def need(value,message):
    if not value:raise ValueError('V135 external: '+message)
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
    cp=P/'CPU_RESULT_V135.json';fp=P/'PREPARED_SOURCE_HASHES_V135.json';cpu=read(cp);freeze=read(fp)
    need(sha(cp)==CPU_SHA and sha(fp)==FREEZE_SHA and cpu['status']=='passed' and cpu['tests_run']==10
        and cpu['failures']==cpu['errors']==0 and cpu['initialization_attempts']==[]
        and cpu['NPU_initialized'] is cpu['model_initialized'] is False and freeze['CPU_sha256']==CPU_SHA
        and cpu['input_hashes']==cpu['input_hashes_after']==freeze['files']
        and cpu['external_input_hashes']==cpu['external_input_hashes_after']==freeze['external_files'],'actual10 HOST/source identity')
    need(len(freeze['files'])==16 and len(freeze['external_files'])==109 and freeze['retained_attempt_files']=={},'sparse source counts')
    for path,wanted in {**freeze['files'],**freeze['external_files']}.items():need(sha(path)==wanted,'source changed: '+path)
    expected={**freeze['files'],str(cp):CPU_SHA,str(fp):FREEZE_SHA}
    need(receipt['source_hashes']==expected and receipt['external_input_hashes']==freeze['external_files'],'actual supervisor source maps')
    prepared=P/'PREPARED_SOURCE_ARCHIVE_V135.tar.gz';need(sha(prepared)==ARCHIVE_SHA,'prepared archive SHA')
    p=archive(prepared,expected);actual=archive(receipt['source_snapshot'],expected)
    need(actual['sha256']==receipt['source_snapshot_sha256'],'actual tar SHA')
    pre=D/'ROOT_V135_PRE_NPU_SOURCE_AND_HOST_AUDIT.json';need(sha(pre)==PRE_SHA and read(pre)['status']=='passed','ROOT prelaunch audit')
    from check_v135 import source_contract
    contract=source_contract()
    need(contract['parent_stagefix_and_numerical_AST_preserved'] is True and contract['actual_module_origin_guard'] is True
        and contract['parent_new_partition_qualification_inherited'] is False,'parent exact AST/numerical scope')
    need(receipt['actual_sourced_runtime_shells_before']==receipt['actual_sourced_runtime_shells_after']
        and receipt['runtime_aliases_before']==receipt['runtime_aliases_after'],'actual runtime source/aliases changed')
    for row in receipt['actual_sourced_runtime_shells_before']:need(sha(row['path'])==row['sha256'],'CANN/ATB source shell changed')
    for side in ('pre','post'):
        verifier=receipt[side+'_verifier'];need(verifier['exit_code']==0 and verifier['argv'][:4]==['bash',str(B/'serial-v1/runtime.sh'),'python','-B'],'actual retained verifier')
        need((P/f'{TAG}-{side}-verifier.log').is_file(),'actual verifier log')
    return dict(actual_HOST_tests=10,CPU=desc(cp),freeze=desc(fp),local_sources=16,external_SHA_inputs=109,
        prepared_archive=p,actual_archive=actual,ROOT_prelaunch=desc(pre),exact_parent_source_contract=contract,
        all_local_source_bytes_archived=True,external_sources_SHA_checked_but_not_all_bytes_in_this_local_tar=True,
        runtime_import_event_history_available=False,actual_worker_helper_origin_snapshots_retained=True)


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
    expected=[['calibrated',r,dict(rows=0,timings=0 if r==0 else 40)] for r in range(4)]
    need(read(root/'completed.json')['replies']==receipt['completed_replies']==expected,'original4 completion replies')
    need(receipt['action']=='qualified44-largeB3-stagefix-cost' and receipt['child']['argv']==['bash',str(P/'launch_mapper_v135.sh'),'--action','qualified44-largeB3-stagefix-cost','--tag',TAG],'actual new action and exact launch argv')
    return dict(passed=True,owned_remaining=[],retained_pre_post_four_card_idle=True,worker_exit=desc(root/'worker-exit.json'),completion=desc(root/'completed.json'))


def owner_queries(case,tokens,order):
    need(type(tokens) is list and len(tokens)==case['Q'] and all(type(t) is int and 0<=t<151936 for t in tokens),'all actual real query tokens')
    owners=list(range(case['N']));owners=owners[::-1] if order=='reverse' else owners;result={};offset=0
    for owner in owners:
        width=case['budgets'][owner]+1;result[owner]=tokens[offset:offset+width];offset+=width
    need(offset==case['Q'] and all(len(v)==4 for v in result.values()),'true allB3 four-query owner partition')
    return result


def token_differences(case,left,right):
    rows=[]
    for owner in range(case['N']):
        for q,(a,b) in enumerate(zip(left[owner],right[owner],strict=True)):
            if a!=b:
                source=case['owners'][owner]
                rows.append(dict(owner=owner,local_query=q,logical_input_position=source['prefix']+q,
                    source_trace_request_index=source['source_trace_request_index'],source_request_id=source['request_id'],
                    input_token_id=([source['root']]+source['teacher_candidate_tokens'])[q],left=a,right=b))
    return rows


def independently_recompute(reports,scope):
    targets={r['rank']:r for r in reports if r['rank']!=0};cs=scope.cases();outputs=[];cross=[];prefix=[];orders=[]
    need([(c['N'],c['Q'],sum(c['budgets'])) for c in cs]==[(n,4*n,3*n) for n in range(27,31)],'four actual N27..30 B3 Q/C fixtures')
    for case in cs:
        name=case['name'];modes=scope.modes(case);arms={}
        for mode in modes:
            samples=[]
            for sample in range(5):
                by_rank={rank:next(r for r in report['cost'] if r['case']==name and r['mode']==mode and r['sample']==sample)['whole_target_complete_ms'] for rank,report in targets.items()}
                need(all(type(v) in (float,int) and math.isfinite(v) and v>0 for v in by_rank.values()),'raw actual timing')
                samples.append(dict(sample=sample,TPmax_ms=max(by_rank.values()),by_rank_ms=by_rank))
            m,segments,dummy=scope.envelope(case,mode)
            need((m,segments)==((165,38) if mode=='old' else (128,32)) and case['Q']+sum(dummy)==m and len(dummy)==segments-case['N'] and all(2<=v<=9 for v in dummy),'actual existing key and legal dummy geometry')
            arms[mode]=dict(M=m,S=segments,dummy_queries=list(dummy),samples=samples,TPmax_median_ms=statistics.median(v['TPmax_ms'] for v in samples))
        ratios=[arms['new']['samples'][i]['TPmax_ms']/arms['old']['samples'][i]['TPmax_ms'] for i in range(5)]
        outputs.append(dict(case=name,N=case['N'],Q=case['Q'],C=sum(case['budgets']),budgets=case['budgets'],arms=arms,actual_delivered_g=None,
            all5_paired_new_over_old=ratios,median_paired_new_over_old=statistics.median(ratios),
            ratio_of_TPmax_medians=arms['new']['TPmax_median_ms']/arms['old']['TPmax_median_ms'],
            relative_reduction_from_medians=1-arms['new']['TPmax_median_ms']/arms['old']['TPmax_median_ms']))
        for rank,report in targets.items():
            banks={v['bank_id']:v for v in report['bank_evidence']};names=[name+'/'+epoch for epoch in ('candidate','independent-reference','zero0','zero1','cost')]
            prefix.append(dict(rank=rank,case=name,saved_bank_count=5,prefix_tensor_hashes_per_bank=128,
                all_five_saved128_prefix_hashes_bitidentical=all(banks[n]['prefix_hashes']==banks[names[0]]['prefix_hashes'] for n in names),
                actual_prefix_lengths=[v['prefix'] for v in case['owners']],source_prompt_digest_closure_rechecked=True,full_prefix_tensor_bytes_retained=False))
            q={(r['mode'],r['owner_order']):r for r in report['qualification'] if r['case']==name}
            for mode in modes:
                for kind in ('greedy','hidden'):
                    for order in ('canonical','reverse'):
                        row=q[mode,order];records=row[kind+'_phases'];existing=row[kind+'_existing_before']
                        if kind=='greedy':need([r['runtime_validated_before'] for r in records]==([True]*3 if existing else [None,False,True]),'actual Native pre-flag cold/changed/restored sequence')
                        need(records[0]['runtime_validated_after'] is existing and all(r['runtime_validated_after'] is True for r in records[1:]),'actual Native post-flag sequence')
                canonical=owner_queries(case,q[mode,'canonical']['greedy_phases'][0]['real_target_query_token_ids'],'canonical')
                reverse=owner_queries(case,q[mode,'reverse']['greedy_phases'][0]['real_target_query_token_ids'],'reverse')
                differences=token_differences(case,canonical,reverse)
                orders.append(dict(rank=rank,case=name,mode=mode,owner_order_remapped=True,real_queries=case['Q'],exact_token_differences=differences,
                    canonical_reverse_exact_equal=not differences,each_order_independent_sameM_reference_gate_retained=True))
            for order in ('canonical','reverse'):
                old=owner_queries(case,q['old',order]['greedy_phases'][0]['real_target_query_token_ids'],order)
                new=owner_queries(case,q['new',order]['greedy_phases'][0]['real_target_query_token_ids'],order)
                differences=token_differences(case,old,new)
                cross.append(dict(rank=rank,case=name,owner_order=order,real_queries=case['Q'],exact_greedy_token_differences=differences,cross_M_hidden_or_KV_equality_compared=False))
                if order=='canonical':need(next(v for v in report['cross_M_comparison'] if v['case']==name)['real_greedy_query_tokens_equal'] is (not differences),'stored canonical crossM marker differs')
    return outputs,cross,prefix,orders


def run():
    need(sys.version_info[:2]==(3,11),'retained3.11 HOST only')
    receipt_path=P/f'receipt-{TAG}.json';receipt=read(receipt_path);root=P/'actual'/TAG
    need(receipt['status']=='complete','terminal complete receipt required before any actual raw audit')
    immutable_before={str(p):sha(p) for p in root.rglob('*') if p.is_file()};immutable_before[str(receipt_path)]=sha(receipt_path)
    guard=NoDeviceImports();sys.meta_path.insert(0,guard);sys.path[:0]=[str(P),str(B/'tree-qualified44-cost-v114')]
    try:
        prov=provenance(receipt);clean=cleanup(receipt,root)
        import audit_mapper_v135 as original
        import mapper_scope_v135 as scope
        import mc2_inventory_contract as MC2
        reports=[read(root/f'rank{r}-mapper-cost.json') for r in range(4)]
        need(receipt['report_paths']==[str(root/f'rank{r}-mapper-cost.json') for r in range(4)],'actual report paths/rank order')
        frozen_audit=original.audit_worker_reports(reports)
        startup=MC2.audit(P/'startup-progress')
        need(all(r['MC2_static']==read(P/'startup-progress'/f'MC2-preflight-rank{r["rank"]}.json') for r in reports),'raw startup MC2 equality')
        startup_artifacts=[desc(P/'startup-progress'/f'MC2-preflight-rank{r}.json') for r in range(4)]
        cost,cross,prefix,orderchecks=independently_recompute(reports,scope)
        for row,known in zip(cost,frozen_audit['cost'],strict=True):
            for mode,values in row['arms'].items():need(values['TPmax_median_ms']==known['arms'][mode]['median_ms']
                and [v['TPmax_ms'] for v in values['samples']]==known['arms'][mode]['rankmax_synchronous_call_ms'],'independent critical-TP arithmetic')
        for report in reports:
            for row in [report['initial_memory'],report['final_memory'],*report['qualification_memory']]:
                need(row['blocks']==544 and 60*1024**3<=row['total']<=64*1024**3 and row['free']>=6*1024**3
                    and row['reserved']<=54*1024**3 and row['peak']<=54*1024**3,'actual544/54GiB/6GiB gate')
            expected={k:0 for k in scope.NATIVE_COUNTERS}
            if report['rank']:
                for key in ('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays','aclgraph_target_runtime_validation_calls'):expected[key]=4
            need(report['final_Native']==expected,'minimal exact4key natural Native final inventory/counters')
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
        need(all(sha(path)==value for path,value in immutable_before.items()),'all original actual data/failed-or-complete receipt remains unchanged')
        need('torch' not in sys.modules and 'torch_npu' not in sys.modules,'HOST only no Torch import')
        return dict(status='passed',errors=[],receipt=desc(receipt_path),actual_receipt_elapsed_seconds=receipt['elapsed_seconds'],
            source_provenance=prov,cleanup=clean,all4_reports=[desc(root/f'rank{r}-mapper-cost.json') for r in range(4)],
            original_frozen_HOST_report_validator_reexecuted=True,MC2_startup=startup,MC2_raw_startup_artifacts=startup_artifacts,
            scopes=dict(actual_target_ranks=[1,2,3],sameM_partitions_per_target=16,independent_sameM_references_per_target=16,
                separately_created_prefix_banks_per_target=20,all_physical_hidden_and_all64_query_KV_actual_matchers=True,
                canonical_and_reverse_partitions=True,Native_F_T_F_and_changed_restore_flags=True,two_complete8arm_zero_sweeps=True,
                all21Native_zero_per_zero_prime_and_cost=True,unified_original_seal=True,all64_task_update_delta=[1,64,0],
                timed_pairs=20,FULL_negative_calls=0,actual_timed_calls_per_target=40,actual_greedy_keys=2,actual_standard_hidden_keys=2),
            costs=cost,cross_M_real_query_tokens=cross,canonical_reverse_owner_query_alignment=orderchecks,saved_prefix_hash_comparisons=prefix,meaningful_tamper_negatives=negative,
            all_cross_M_real_queries_exact_equal=all(not row['exact_greedy_token_differences'] for row in cross),
            all_canonical_reverse_real_queries_exact_equal_after_owner_remap=all(row['canonical_reverse_exact_equal'] for row in orderchecks),
            actual_worker_module_origins=[dict(rank=r['rank'],origins=r['actual_runtime_origins']) for r in reports],
            original_teacher_forced_candidate_inputs=True,genuine_draft_proposals=False,actual_delivered_g=None,
            teacher_prefix_scope=dict(prefix_rule='128 + local_owner%4',real_dataset_prompts=True,teacher_inputs_are_next3_actual_prompt_tokens=True,independently_fresh_PA_prefix_per_bank=True,online_candidate_old_KV_reconstructed=False,matched_oracle_same_M=True,own_AR=False,case_source=desc(P/'ACTUAL_CASE_INPUTS_V135.json')),
            sameM_full_hidden_KV_tensor_bytes_retained=False,HOST_independent_tensor_numerical_recomparison=False,
            cross_M_hidden_KV_equality_qualified=False,arbitrary_token_or_path_exact_qualification=False,
            local_component_numerical_scope_passed=True,production_mapper_installed_or_admitted=False,
            full463_qualified=False,formal_Goodput=False,actual_online_GP=None,NPU_initialized=False,model_initialized=False,
            actual_audit_torch_imported=False,statistical_scope='5 paired calls per each of4 causal cases; three TP ranks are synchronous observations, not independent trials; short source-pinned teacher prefixes only')
    finally:sys.meta_path.remove(guard)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args();start=datetime.now(timezone.utc).isoformat()
    try:value=run()
    except BaseException:value=dict(status='audit_failed',errors=[traceback.format_exc()],NPU_initialized=False,formal_Goodput=False)
    value.update(schema='V135.external-independent-actual-cost-HOST-audit.v1',actual_audit_argv=sys.argv,actual_python=sys.executable,
        runtime_wrapper_used=False,started_UTC=start,finished_UTC=datetime.now(timezone.utc).isoformat(),auditor_source=desc(__file__))
    with Path(args.output).open('x') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(status=value['status'],output=args.output,errors=value['errors'])))
    return int(value['status']!='passed')

if __name__=='__main__':raise SystemExit(main())
