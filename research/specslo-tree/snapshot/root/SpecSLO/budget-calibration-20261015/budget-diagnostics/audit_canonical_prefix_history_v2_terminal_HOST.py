"""SOURCE ONLY until ROOT authorizes one exact terminal receipt SHA.

Standard-library audit of retained diagnostic evidence, never numerical model
re-execution, production qualification, device query, or a performance score.
"""
import argparse,ast,collections,copy,datetime,hashlib,importlib.abc,json,math,re,struct,sys,tarfile
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'canonical-ar-prefix-history-isolation-diagnostic-v2';ACTION=P.name;SCHEMA=ACTION+'.v2'
FREEZE_SHA='2261b25d29867efb7cdd1303915732c44bcf91ca0d1146b5d29d018d7c20b85a'
ROOT_SHA='a75f375018a87545b512c5227d4ccbe622dcd228342988745e7e1d7d4a98f054'
PEER_SHA='213176bfeabece4d06d6ccb4070eb9d60deea8f39aa004ec912bf1678d30aa39'
ARCHIVE_SHA='eb147d3e08eec80d83af2a8c399b179656a93bbccad9d40cfb8cda85a1dcc611'
OLD=D/'audit_canonical_common_prefix_terminal_HOST.py'
OLD_SHA='891b414206ec056aa0eb5f6f4a45e7f8ce1b43b8f25e6d7f8f02b2b1e9992eaa'
FLAGS=dict(production_qualified=False,whole463_AR_qualified=False,numerical_losslessness_proved=False,formal_Goodput=False,adopted=False)
BACKENDS=('torch','torch_npu','vllm','vllm_ascend','transformers','acl','safetensors')
HELPERS=('need','sha','read','desc','digest','function','saved_baseline','input_record','rounded','head_record','graph_state')

class Firewall(importlib.abc.MetaPathFinder):
    attempts=[]
    def find_spec(self,name,path=None,target=None):
        if name.split('.')[0]in BACKENDS:
            self.attempts.append(name);raise ImportError('v2 terminal HOST forbids '+name)

def load_helpers():
    # No import of candidate modules or the old audit's main; exact pure defs.
    if hashlib.sha256(OLD.read_bytes()).hexdigest()!=OLD_SHA:raise ValueError('pinned original pure HOST helpers')
    unit=ast.parse(OLD.read_text());nodes=[copy.deepcopy(n)for n in unit.body if isinstance(n,ast.FunctionDef)and n.name in HELPERS]
    if {n.name for n in nodes}!=set(HELPERS):raise ValueError('complete exact11 original pure HOST helpers')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(OLD)+'::v2-pure-definitions','exec'),globals())

def source_evidence():
    fpath=P/'FROZEN_DIAGNOSTIC.json';need(sha(fpath)==FREEZE_SHA,'new exact ROOT freeze')
    f=read(fpath);rp=Path(f['root_review']['path']);pp=Path(f['peer_review']['path']);rd=Path(f['ready']['path'])
    need(sha(rp)==f['root_review']['sha256']==ROOT_SHA and sha(pp)==f['peer_review']['sha256']==PEER_SHA and sha(rd)==f['ready']['sha256'],'ROOT/peer/ready identity')
    root,ready=read(rp),read(rd);cpu=read(P/'CPU_RESULT.json');prepared=read(P/'PREPARED.json')
    need(root['status']=='passed_source_and_retained_HOST_review'and root['ROOT_source_reviewed']is f['ROOT_source_reviewed']is True and all(f[k]is False for k in FLAGS),'source-only frozen review scope')
    need(len(f['sources']['local'])==10 and len(f['sources']['external'])==69,'actual10/69 source schema')
    need(cpu['source_before']==cpu['source_after']==prepared['sources']==f['sources'] and sha(P/'CPU_RESULT.json')==f['CPU_sha256']==prepared['CPU_sha256'],'current CPU/PREPARED/source before-after')
    need(cpu['status']=='passed'and cpu['tests_run']==4 and cpu['errors']==cpu['failures']==0 and cpu['Torch_imported']is cpu['NPU_used']is cpu['model_constructed']is cpu['old_suites_rerun']is False and cpu['backend_modules_after']==[],'actual new4 HOST schema; not parent REV3 schema')
    flat={**ready['local_sources'],**ready['selected_external'],**ready['retained_HOST_evidence']}
    need(flat==root['exact_verified_pins'] and len(flat)==89 and all(flat[k]==v for d in f['sources'].values()for k,v in d.items()),'ROOT exact89 current pins')
    need(all(sha(k)==v for k,v in flat.items()),'all selected current bytes')
    selected={**flat,str(rp):ROOT_SHA,str(pp):PEER_SHA,str(rd):sha(rd)};archive=Path(f['source_archive']['path'])
    need(len(selected)==f['source_members']==92 and sha(archive)==f['source_archive']['sha256']==ARCHIVE_SHA,'exact92 archive map/binary')
    with tarfile.open(archive,'r:gz')as bundle:
        members=bundle.getmembers();need(len(members)==len({m.name for m in members})==92,'unique92 archive members')
        for member in members:
            key='/'+member.name.lstrip('/')
            need(member.isfile()and key in selected and hashlib.sha256(bundle.extractfile(member).read()).hexdigest()==selected[key],'regular exact archive member '+member.name)
    return dict(freeze=desc(fpath),ROOT_review=desc(rp),peer=desc(pp),ready=desc(rd),CPU=desc(P/'CPU_RESULT.json'),archive=dict(**desc(archive),members=92),selected_SHA_checked=89,old_suites_or_verifier_rerun=False)

def terminal_evidence(r):
    need(r['schema']=='canonical-ar-prefix-history-isolation-diagnostic.receipt.v2'and r['physical_cards']==[2,5,6,7]and r['execution_lock']==str(B/'.ROOT_NPU_EXECUTION.lock'),'actual v2 receipt/lock/cards')
    need(r['sources']==read(P/'FROZEN_DIAGNOSTIC.json')['sources']and r['sources_unchanged']is True and r['owned_remaining']==[],'source prepost and owned cleanup')
    baseline={k:saved_baseline(r[k])for k in ('pre_verifier','post_verifier')}
    for key in ('devices_before','devices_after'):
        v=r[key];need(v['all_idle']is v['four_910B2']is True and set(v['processes'])=={'2','5','6','7'}and sorted(map(tuple,v['parsed_hardware']))==[(2,'910B2'),(5,'910B2'),(6,'910B2'),(7,'910B2')],'saved actual4 hardware/idle')
        h=v['hardware'];need(h['exit_code']==0 and h['timed_out']is False and sorted((int(i),n)for i,n in re.findall(r'^\|\s*(\d+)\s+(\S+)\s*\|',h['output'],re.M)if not n.isdecimal())==[(2,'910B2'),(5,'910B2'),(6,'910B2'),(7,'910B2')],'independent saved hardware parsing')
        for proc in v['processes'].values():need(proc['exit_code']==0 and proc['timed_out']is False and proc['idle']is True and re.search(r'No process in device\.|No running processes found',proc['output'],re.I)is not None,'saved idle process evidence')
    need(all(r[k]is False for k in FLAGS),'unscored unqualified flags')
    need(type(r.get('termination_signals'))is list,'actual REV3 signal receipt field')
    if r['status']=='complete':need(r['termination_signals']==[],'a termination request cannot be called successful completion')
    need(type(r['private_bytes'])is int and 0<=r['private_bytes']<=128*1024**2 and type(r['elapsed_seconds'])in(int,float)and math.isfinite(r['elapsed_seconds'])and r['elapsed_seconds']>0,'actual polling storage/elapsed schema')
    return dict(baseline=baseline,owned_remaining=[],all4_idle=True,source_prepost=True,termination_signals=r['termination_signals'],declared_guardian_limits=dict(seconds=3600,private_bytes=128*1024**2),cleanup_in_terminal_elapsed=True,terminal_elapsed_not_forced_below3600=True,exact_private_peak_not_retained=True)

def byte_record(row,*,shape=None):
    need(type(row)is dict and row['dtype']=='torch.bfloat16'and type(row['shape'])is list and all(type(v)is int and v>0 for v in row['shape'])and re.fullmatch('[0-9a-f]{64}',row['sha256'])is not None,'retained original BF16 bytes/shape/SHA schema')
    if shape is not None:need(row['shape']==shape,'actual byte record shape')

def past_record(value,case):
    root=len(case['common_prefix_tokens'])-1
    need(value['root_position']==root and value['logical_positions_per_owner']==[0,root]and value['end_exclusive']is True and value['owners']==list(range(16))and value['slot_count']==16*root and value['root_and_future_slots_included']is value['reference_KV_copied']is False,'all16 actual past excludes root/future')
    need(type(value['allocation_id'])is int and value['allocation_id']>0 and all(re.fullmatch('[0-9a-f]{64}',value[k])is not None for k in ('slots_sha256','block_tables_sha256')),'local allocator/table/slot fingerprint schema')
    arrays=value['arrays'];need([(v['layer'],v['kind'])for v in arrays]==[(i,k)for i in range(64)for k in ('key_cache','value_cache')],'all64 actual target K/V=128')
    for row in arrays:byte_record(row);need(row['shape'][0]==16*root,'each protected KV covers every owner past slot')

def top2(tokens,ids,scores):
    need(len(tokens)==len(ids)==len(scores),'actual chosen roots/top2 arrays')
    for token,index,values in zip(tokens,ids,scores):
        need(type(token)is int and 0<=token<151936 and len(index)==len(values)==2 and len(set(index))==2 and all(type(i)is int and 0<=i<151936 for i in index),'actual distinct top2 token vocabulary')
        need(all(type(v)in(int,float)and math.isfinite(v)for v in values)and values[0]>=values[1],'finite ordered logits')
        if values[0]>values[1]:need(token==index[0],'unique maximum agrees with argmax')

def pair_record(row,tokens,positions):
    need(row['same_computed_hidden']is True and row['precision_or_weights_changed']is row['cross_shape_equality_required']is False and (row['model_physical_M'],row['head_full_physical_M'],row['head_owner0_input_M'])==(80,80,1),'same computed hidden; original full80 before selection versus row1')
    need(row['hidden_dtype']==row['head_weight_dtype']==row['logits_dtype']=='torch.bfloat16'and row['restricted_vocabulary']==151936 and type(row['head_pad_multiple'])is int and row['head_pad_multiple']>0,'original head dtype/vocabulary/pad metadata')
    input_record(row['actual_input'],tokens,positions,fia=True)
    byte_record(row['root0_hidden']);need(row['root0_hidden']['shape'][0]==1,'actual root hidden row byte schema')
    for key in ('full80_root0_logits','row1_root0_logits'):byte_record(row[key],shape=[151936])
    need(len(row['full80_root_tokens'])==16,'all16 full80 root observations')
    top2(row['full80_root_tokens'],row['full80_root_top2_ids'],row['full80_root_top2_logits'])
    top2([row['row1_root0_token']],[row['row1_root0_top2_ids']],[row['row1_root0_top2_logits']])
    indices=row['root0_different_vocabulary_indices'];delta=row['root0_max_abs_logit_difference'];exact=row['root0_restricted_logits_exact']
    need(type(exact)is bool and type(indices)is list and indices==sorted(set(indices))and all(type(i)is int and 0<=i<151936 for i in indices)and type(delta)in(int,float)and math.isfinite(delta)and delta>=0,'retained exact difference index/max schema')
    need(exact is (not indices)and (delta==0)is exact,'numeric exact flag, differing indices and max consistent; equality itself is not required')
    byte_equal=row['full80_root0_logits']==row['row1_root0_logits']
    if not exact:need(not byte_equal,'different finite BF16 values cannot have identical retained raw byte hashes')
    # Numeric equality may still have different raw signed-zero encodings.
    # Preserve byte equality independently; do not require its converse.
    a=dict(zip(row['full80_root_top2_ids'][0],row['full80_root_top2_logits'][0]));b=dict(zip(row['row1_root0_top2_ids'],row['row1_root0_top2_logits']))
    for i in a.keys()&b.keys():
        need(delta>=abs(a[i]-b[i]),'retained shared top2 scores bound the reported full-vector max')
        if a[i]!=b[i]:need(i in indices,'observed differing score index is in full difference list')
    return dict(root0_full80_token=row['full80_root_tokens'][0],root0_row1_token=row['row1_root0_token'],root0_tokens_equal=row['full80_root_tokens'][0]==row['row1_root0_token'],retained_logits_exact=exact,retained_byte_hash_equal=byte_equal,retained_differing_vocabulary_count=len(indices),retained_max_abs_logit_difference=delta,raw_logits_bytes_recomputed=False)

def report_case(report,case,rank,actual):
    need(report['schema']==SCHEMA and report['rank']==rank and report['fixture_sha256']==sha(P/'FIXTURE.json')and report['request_id']==case['request_id']and report['request_index']==case['request_index'],'exact rank/case/source')
    function(P/'ar_margin_actor.py','audit_case_report',dict(require=need,FLAGS=FLAGS))(report,case)
    need(report['same_allocation_controlled_routes']is report['reconstructed_cohort_not_historical_call']is True and report['independent_fresh_route_qualification']is report['reference_KV_copied']is False,'actual controlled-history scope')
    histories=report['own_histories'];calls=0
    for owner,row in enumerate(histories):
        need(read(actual/f'rank{rank}-{case["case_id"]}-owner{owner}-history.json')==row,'saved owner history equals case report')
        need(row['outputs_sha256']==digest(row['outputs'])and row['cache_ready_through_position_exclusive']==len(case['common_prefix_tokens'])-1 and len(row['calls'])==case['step'],'actual owner outputs/past extent')
        for step,call in enumerate(row['calls']):
            inp=case['prompt_tokens']if step==0 else[row['outputs'][step-1]]
            positions=list(range(len(inp)))if step==0 else[len(case['prompt_tokens'])+step-1]
            need(call==dict(step=step,input_sha256=digest(inp),positions=[positions[0],positions[-1]+1],actual_prediction=row['outputs'][step],expected_reference_token=case['reference_outputs'][step]),'independently reconstructed actual own recurrence and only post-prediction reference assertion')
        calls+=row['actual_model_calls']
    expected=0 if rank==0 else 16*case['step']
    need(all(type(report[k])is int for k in ('required_own_model_calls','required_own_single_row_head_calls','required_shared_history_steps'))and calls==report['required_own_model_calls']==report['required_own_single_row_head_calls']==expected and report['required_shared_history_steps']==16*case['step'],'per-rank own model/head requests; draft0')
    result=dict(case_id=case['case_id'],rank=rank,own_model_and_single_head_calls=expected,shared_history_steps=16*case['step'],own_reference_prefix_reproduced=True,independent_host_recurrence_recomputed=True,original_pure_case_audit_reexecuted=True)
    root=len(case['common_prefix_tokens'])-1;positions=list(range(root,root+5))*16;inputs=[case['common_prefix_tokens'][-1],0,0,0,0]*16
    capacity=report['query_capacity'];need(capacity==dict(owners=16,root=root,reserved_through_exclusive=root+5,cache_not_rebuilt=True,reference_KV_copied=False),'actual reserve before protected fingerprint')
    if rank==0:
        need(report['past_before']is report['past_final']is None and all(r['past_after']is r['snapshot']is None for r in report['controlled_PA1']),'draft PA coordination only')
        for row in report['controlled_eager_FIA80']+report['controlled_natural_hidden80']:need(row['heads']is row['past_after']is row['before']is row['after']is None,'draft FIA coordination only')
        need(report['controlled_natural_greedy80']==[None,None,None],'draft greedy coordination only');return result
    protected=report['past_before'];past_record(protected,case)
    for owner,row in enumerate(report['controlled_PA1']):
        need(row['owner']==owner and row['past_after']==protected,'all16 controlled PA preserve original protected past')
        head_record(row['snapshot'],[case['common_prefix_tokens'][-1]],[root],fia=False)
    pair_summaries=[]
    for kind,phases in (('controlled_eager_FIA80',['eager']),('controlled_natural_hidden80',['first','changed','restored'])):
        rows=report[kind];need([r['phase']for r in rows]==phases,'real eager/graph phase order')
        for row in rows:
            tokens=list(inputs)
            if row['phase']=='changed':tokens[0]=(tokens[0]+1)%151936
            need(row['past_after']==protected,'every hidden route preserves all128 protected hashes')
            pair_summaries.append(dict(route=kind,phase=row['phase'],**pair_record(row['heads'],tokens,positions)))
            if kind=='controlled_natural_hidden80':graph_state(row['before'],after=False);graph_state(row['after'],after=True)
            else:need(row['before']is row['after']is None,'direct eager does not invent Native graph state')
    greedy=report['controlled_natural_greedy80'];need([r['phase']for r in greedy]==['first','changed','restored'],'real natural greedy phase order')
    for row in greedy:
        tokens=list(inputs)
        if row['phase']=='changed':tokens[0]=(tokens[0]+1)%151936
        need(len(row['physical_tokens'])==80 and all(type(t)is int and 0<=t<151936 for t in row['physical_tokens'])and row['past_after']==protected,'all80 actual greedy tokens/protected past')
        input_record(row['actual_input'],tokens,positions,fia=True);graph_state(row['actual'],after=True)
    hidden=report['controlled_natural_hidden80']
    result.update(past128_original_dtype_fingerprints_structurally_checked=True,all_routes_preserve_same_local_past=True,pair_observations=pair_summaries,
      hidden_FTF_modes=[v['after']['execution']['mode']for v in hidden],greedy_FTF_modes=[v['actual']['execution']['mode']for v in greedy],
      hidden_first_restored_roots_equal=hidden[0]['heads']['full80_root_tokens']==hidden[2]['heads']['full80_root_tokens'],greedy_first_restored_all80_equal=greedy[0]['physical_tokens']==greedy[2]['physical_tokens'])
    return result

def completion(actual,cases):
    done=read(actual/'completed.json');exited=read(actual/'worker-exit.json')
    need(exited['closed']is exited['all_exited']is True and all(exited[k]is False for k in FLAGS),'all workers closed/exited')
    need(done['status']=='unscored-diagnostic-complete'and done['case_ids']==[c['case_id']for c in cases]and all(done[k]is False for k in FLAGS),'actual unscored diagnostic completion')
    need(done['all16_histories_are_genuine_own_AR']is done['controlled_routes_are_not_independent_fresh_qualification']is done['inner_receive3600_is_not_guaranteed_by_outer_guardian3600']is True and done['required_own_model_calls_per_target_rank']==9200 and done['configuration_delta']=={'worker_timeout_seconds':[1800,3600]}and done['API_receive_timeout_seconds']==3600,'original configuration single operational delta and actual9200 scope')
    replies=done['replies'];need(sorted(v[1]for v in replies)==list(range(4))and all(v[0]=='calibrated'for v in replies),'actual all4 command completions')
    summaries=[]
    for rank in range(4):
        row=read(actual/f'rank{rank}-margin-summary.json');summaries.append(row)
        need(row['schema']==SCHEMA and row['rank']==rank and row['action']==ACTION and row['fixture_sha256']==sha(P/'FIXTURE.json')and row['enforce_eager']is row['precompile_decode_graphs']is row['precompile_serial_draft_graphs']is False,'original whole-worker summary/config')
        need(row['before']['aclgraph_entries']==row['before']['aclgraph_target_entries']==0 and all(row[k]is False for k in ('candidate_logits_measured','candidate_KV_measured','numerical_qualification','online_qualified','cost_or_Goodput')),'no inherited graph inventory/production qualification')
        need(row['cases']==[dict(request_index=c['request_index'],status='diagnostic_complete')for c in cases]and next(v[2]for v in replies if v[1]==rank)==row,'actual four-case order and saved-vs-IPC summaries')
    return dict(completed=desc(actual/'completed.json'),worker_exit=desc(actual/'worker-exit.json'),summaries=summaries,config_delta=done['configuration_delta'],per_target_rank_own_model_and_head_requests=9200,draft_model_and_head_requests=0)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--terminal-authorized',action='store_true',required=True);parser.add_argument('--expected-receipt-sha256',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    if args.output.parent.resolve()!=D or args.output.exists():raise ValueError('fresh privateD output only')
    if re.fullmatch('[0-9a-f]{64}',args.expected_receipt_sha256)is None:raise ValueError('ROOT exact terminal receipt SHA required')
    result=dict(schema='canonical-prefix-history-v2.independent-terminal-HOST.v1',errors=[],audited_stages=[],expected_receipt_sha256=args.expected_receipt_sha256,**FLAGS)
    stage='ROOT-terminal-authorization';load_helpers();sys.meta_path.insert(0,Firewall())
    try:
        receipt_path=P/'receipt.json';need(sha(receipt_path)==args.expected_receipt_sha256,'exact ROOT authorized receipt; no future or running read')
        receipt=read(receipt_path);need(receipt['status']in('complete','failed'),'running receipt forbidden')
        result.update(receipt=desc(receipt_path),job_status=receipt['status'],script=desc(__file__));result['audited_stages'].append(stage)
        stage='frozen-source-archive-CPU';result['provenance']=source_evidence();result['audited_stages'].append(stage)
        stage='terminal-cleanup-baseline'
        try:result['terminal']=terminal_evidence(receipt);result['audited_stages'].append(stage)
        except Exception as error:result['errors'].append(dict(stage=stage,type=type(error).__name__,message=str(error)))
        if receipt['status']!='complete'or receipt['errors']!=[]or receipt.get('child_exit_code')!=0 or receipt.get('diagnostic_execution_complete')is not True:
            result['errors'].append(dict(stage='actual-job-status',type='FailedOrIncompleteJob',message='Retain whole-job failure even if individual cases completed',actual_errors=receipt['errors']))
        actual=P/'actual';stage='private-actual-regular-inventory';paths=sorted(actual.rglob('*'))
        need(actual.is_dir()and not actual.is_symlink()and not any(p.is_symlink()for p in paths),'private actual regular artifacts')
        result['actual_files']={str(p):desc(p)for p in paths if p.is_file()};result['audited_stages'].append(stage)
        fixture=read(P/'FIXTURE.json');cases=function(P/'contract.py','validate_fixture',dict(require=need,digest=digest,SCHEMA=SCHEMA))(fixture)
        stage='completion-and-original-worker-schema'
        try:result['completion']=completion(actual,cases);result['audited_stages'].append(stage)
        except Exception as error:result['errors'].append(dict(stage=stage,type=type(error).__name__,message=str(error)))
        result['case_audits']=[];totals=dict.fromkeys(range(4),0);shared=dict.fromkeys(range(4),0)
        for case in cases:
            stage='case/'+case['case_id'];reports=[]
            try:
                for rank in range(4):
                    stage='case/'+case['case_id']+'/rank'+str(rank)
                    report=read(actual/f'rank{rank}-{case["case_id"]}.json');reports.append(report)
                    value=report_case(report,case,rank,actual);result['case_audits'].append(value);totals[rank]+=value['own_model_and_single_head_calls'];shared[rank]+=value['shared_history_steps']
                stage='case/'+case['case_id']+'/TP-gathered-observations'
                observations=function(P/'ar_margin_actor.py','tp_observations',{})
                views=[observations(r)for r in reports[1:]];need(views[0]==views[1]==views[2],'only gathered heads/tokens must agree acrossTP; no private hidden/KV hash equality')
                need(all([v['outputs']for v in r['own_histories']]==[v['outputs']for v in reports[0]['own_histories']]for r in reports),'all4 actual shared own prediction schedules')
                result['audited_stages'].append(stage)
            except Exception as error:result['errors'].append(dict(stage=stage,type=type(error).__name__,message=str(error)))
        result['recorded_own_request_totals_by_rank']=totals;result['recorded_shared_history_steps_by_rank']=shared
        if len(result['case_audits'])==16:
            need(totals=={0:0,1:9200,2:9200,3:9200}and shared==dict.fromkeys(range(4),9200),'9200 each target, zero draft; do not collapse ranks')
        else:result['errors'].append(dict(stage='all-cases-completeness',type='IncompleteCases',message='Fewer than16 independently audited rank/case reports'))
        result['saved_partial_or_last_error_files']=[str(p)for p in paths if p.is_file()and ('partial-error'in p.name or 'last-error'in p.name)]
        result['runtime_error_evidence']=[]
        for name in result['saved_partial_or_last_error_files']:
            error=read(name);partial=error.get('partial',{});own=partial.get('own_histories',[])
            last=own[-1]if own else {};calls=last.get('calls',[])
            result['runtime_error_evidence'].append(dict(artifact=desc(name),error=error.get('error'),case_id=partial.get('case_id'),last_retained_owner=last.get('owner'),last_retained_call=calls[-1]if calls else None,chronological_first_across_ranks_not_inferred=True))
        result['diagnostic_log']=desc(P/'model-diagnostic.log')
    except Exception as error:result['errors'].append(dict(stage=stage,type=type(error).__name__,message=str(error)))
    if Firewall.attempts or any(n.split('.')[0]in BACKENDS for n in sys.modules):result['errors'].append(dict(stage='HOST-backend-firewall',type='ForbiddenBackend',message=str(Firewall.attempts)))
    result['first_failure_stage']=result['errors'][0]['stage']if result['errors']else None
    result['status']='passed_unscored_terminal_evidence_audit'if not result['errors']else'failed_or_incomplete_terminal_evidence_audit'
    result['limitations']=[
      'Retained actual runtime gates only; no hidden/logit/KV tensor is recomputed by this HOST tool. Past/byte hashes are checked structurally and for preservation, not regenerated from raw tensors.',
      'Root0 full-vocabulary logits are retained as original BF16 byte hashes, exact difference indices and max delta, not raw full-vocabulary arrays. Only top2 arithmetic and consistency bounds can be independently recomputed. Full80 and row1 may differ without failing this diagnostic.',
      'Each9200 count is external own-AR model/head requests on EACH target rank, not aggregate target calls or layer ops. Draft participates in9200 shared steps and has zero such model/head requests. ControlledPA/graph and internal capture/validation forwards are additional.',
      'All16 owners build their own past from actual predictions before the controlled routes reuse that allocation. This is history isolation, not independent fresh same-M qualification, a historical packed candidate call, whole463 AR equivalence or production permission.',
      'Only gathered token/head observations compare across TP. Private hidden/KV shard fingerprints are compared within each rank; no cross-rank KV equality is required.',
      'F/T/F token differences, changed output, graph fallback or head-shape disagreement remain observations. No low-precision cause, losslessness, Goodput or formal performance conclusion is asserted.',
      'Original guardian3600s starts before constructor/preflight and cleanup may extend terminal elapsed. Saved polling bytes are not exact private peak. Source declarations are not a promise of runtime fit.',
      'A failed job or partial/missing case stays failed/incomplete even when retained earlier cases are complete. ROOT exact terminal receipt authorization is mandatory before reading actual outputs.'
    ];result['UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with args.output.open('x')as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(path=str(args.output),sha256=sha(args.output),status=result['status'],first_failure_stage=result['first_failure_stage'])))
    return int(bool(result['errors']))

if __name__=='__main__':raise SystemExit(main())
