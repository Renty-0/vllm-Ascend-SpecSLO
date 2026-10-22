"""Independent retained P151 terminal audit; stdlib HOST only, no execution.

Reads exact frozen selected sources/archive and actual terminal artifacts. It
recomputes raw acceptance/zero arithmetic, not device hidden/KV tensor math.
"""
import argparse,ast,datetime,gzip,hashlib,json,re,tarfile,time,traceback
from collections import Counter
from pathlib import Path

B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-normal4-parent3-shared4-eager-v151';D=B/'budget-diagnostics'
TAG='normal4-parent3-shared4-v151-local-v1';ACTION='normal4-parent3-child3-shared4-local'
RECEIPT_SHA='0ba407d9ccd1b73560ad27f075dfd52d4e673111f5169e37084ef29e242fcbb2'
CPU_SHA='a552c3be4a431334e66c2b8a1a717cd8eb1e962e25a23119749896c661d7f457'
FREEZE_SHA='5ff880a1dfa426d4f887529169e91f4f4d89e4cac5b5ea01ab7679719f5a095f'
ARCHIVE_SHA='c59c642310082ca509c8d75da637c5be0b08249b785923ee616ca2543bd03181'
ROOT_REVIEW_SHA='8a1df1a90ededefd1c66b107d85b967cb78bce2944ab107cf1c5412ad1023fa0'
FIELDS=('aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays','aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks',*(f'aclgraph_{r}_{f}' for r in ('generic','draft','target') for f in ('eager_fallback_calls','disabled_entry_calls','unclassified_calls','runtime_validation_calls','runtime_validation_failures')))
NORMAL_B=(4,)*19+(3,)+(2,)*5+(1,)*7

def need(value,message):
 if value is not True:raise ValueError('V151 independent: '+message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
 def reject(v):raise ValueError('nonfinite JSON '+v)
 return json.loads(Path(p).read_bytes(),parse_constant=reject)
def desc(p):
 p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def merge(*maps):
 out={}
 for m in maps:
  for p,h in m.items():need(p not in out or out[p]==h,'noncontradictory source maps');out[p]=h
 return out

def audit_sources(receipt):
 cp=P/'CPU_RESULT151.json';fp=P/'FROZEN_SOURCE151.json';ap=P/'PREPARED_SOURCE_ARCHIVE_V151.tar.gz'
 need(sha(cp)==CPU_SHA and sha(fp)==FREEZE_SHA and sha(ap)==ARCHIVE_SHA,'exact CPU/freeze/archive')
 c,f=read(cp),read(fp)
 need(c['status']=='passed' and c['tests_run']==5 and c['failures']==c['errors']==0 and c['Torch_imported'] is c['NPU_initialized'] is c['model_initialized'] is False,'original own actual HOST proof')
 need(c['input_hashes']==c['input_hashes_after']==f['files'] and c['external_input_hashes']==c['external_input_hashes_after']==f['external_files'] and f['CPU_sha256']==CPU_SHA and f['ROOT_source_reviewed'] is True and len(f['files'])==20 and len(f['external_files'])==146,'original own exact pre/post selected source')
 inputs=merge(f['files'],f['external_files'],f['retained_HOST_evidence'],{str(cp):CPU_SHA},f['execution_guard_inputs'],f['peer_reviews'])
 root=f['root_review'];need(root['sha256']==ROOT_REVIEW_SHA and sha(root['path'])==ROOT_REVIEW_SHA,'ROOT review identity')
 reviewed=read(root['path']);need(reviewed['status']=='passed_source_review' and reviewed['reviewed_inputs']==inputs,'exact ROOT review flat source binding')
 all_inputs=merge(inputs,{root['path']:root['sha256'],str(fp):FREEZE_SHA})
 for p,h in all_inputs.items():
  path=Path(p);need(path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve()==path and sha(path)==h,'current regular selected source '+p)
 with tarfile.open(ap,'r:gz') as tar:
  members=tar.getmembers();need(len(members)==len(all_inputs)==183 and {m.name for m in members}=={p.lstrip('/') for p in all_inputs},'exact selected archive183 set')
  for m in members:need(m.isfile() and hashlib.sha256(tar.extractfile(m).read()).hexdigest()==all_inputs['/'+m.name],'archive actual member bytes '+m.name)
 r=receipt['readiness'];need(r['CPU_sha256']==CPU_SHA and r['freeze_sha256']==FREEZE_SHA and r['root_review']==root and r['peers']==f['peer_reviews'] and r['archive']['sha256']==ARCHIVE_SHA and r['archive']['members']==183,'actual receipt exact source readiness')
 return dict(local=20,external=146,deduplicated_archive_members=183,current_selected_bytes_verified=True,archive_member_bytes_verified=True,CPU=desc(cp),freeze=desc(fp),archive=desc(ap),ROOT_review=root,complete_environment_import_closure_claim=False),f

def verifier(row):
 p=Path(row['path']);need(sha(p)==row['sha256'] and row['exit_code']==0,'actual baseline verifier log identity/exit')
 values=[]
 for line in p.read_text().splitlines():
  try:x=json.loads(line)
  except json.JSONDecodeError:continue
  if type(x)is dict and 'checked_frozen_files' in x:values.append(x)
 need(len(values)==1 and values[0]==row['actual_result'],'actual baseline verifier result matches raw log')
 x=values[0];need(x['status']=='pass' and x['errors']==[] and (x['checked_frozen_files'],x['checked_external_files'],x['checked_packages'])==(10496,15,212),'actual original full verifier10496/15/212 PASS')
 return dict(log=desc(p),result=x)

def idle(row):
 need(row['all_idle'] is row['four_910B2'] is True and row['parsed_hardware']==[[n,'910B2'] for n in (2,5,6,7)],'actual original four910B2 idle schema')
 h=row['hardware'];need(h['exit_code']==0 and h['timed_out'] is False and all('No running processes found in NPU '+str(n) in h['output'] for n in (2,5,6,7)),'retained actual hardware no-process output')
 for n in (2,5,6,7):
  p=row['processes'][str(n)];need(p['idle'] is True and p['exit_code']==0 and p['timed_out'] is False and re.fullmatch(r'\s*No process in device\.\s*',p['output']) is not None,'retained actual per-card no-process output')
 return dict(physical_cards=[2,5,6,7],retained_original_commands_exit0=True,all_idle=True,current_device_queried_by_auditor=False)

def finish_source(frozen):
 native=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_engine.py')
 need(frozen['external_files'][str(native)]==sha(native),'original Native source is in frozen selection')
 tree=ast.parse(native.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='NativePearlEngine')
 fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_generate_spec_rhythm_tree_decode')
 assigned=lambda n,name:isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
 full=next(n for n in ast.walk(fn) if assigned(n,'fully_accepted'));expression=ast.unparse(full.value)
 need(expression=='accepted_count >= self._tree_primary_path_length(tree_plan)','original full acceptance threshold unchanged')
 eager=next(n for n in ast.walk(fn) if assigned(n,'eager_valid') and isinstance(n.value,ast.BoolOp))
 expr=ast.unparse(eager.value)
 for key in ('eager_valid','eager_parent_proposal_id','dependency_length > 0','observed_path == dependency_tokens','observed_frontier == int(eager_frontier)'):need(key in expr,'original eager validity conjunct '+key)
 setup=(P/'protocol151.py').read_text();need('path==(0,1,2)' in setup and "child_plan.prefix_len==plan.prefix_len+4" in setup,'actual original parent3 dependency source')
 return dict(native_source=desc(native),fully_accepted_expression=expression,parent_primary_path=[0,1,2],required_parent_accept_count=3,eager_valid_expression=expr,reason_no_promotion='Every actual parent accepted fewer than3 nodes, so original full-parent gate is false before further dependency/frontier guards. No extra reuse gate is needed to explain invalidation.',frontier_dependency_match_recomputed_from_raw=False,frontier_limitation='The retained report does not include full staged frontier/dependency rows; they cannot be independently reconstructed here. The already-false accepted_count>=3 gate suffices to explain all actual negatives.')

def raw_report(path):
 need(path.is_file() and not path.is_symlink() and path.stat().st_size<=32*1024**2,'bounded regular rank report')
 with gzip.open(path,'rb') as s:raw=s.read(64*1024**2+1)
 need(len(raw)<=64*1024**2,'bounded expanded rank report')
 return json.loads(raw,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite raw '+v)))

def audit_traces(reports):
 zero_rows=[];phases={};fingerprints={};stages=[]
 for rank,r in enumerate(reports):
  need(r['rank']==rank and r['role']==('draft' if rank==0 else 'target') and r['action']==ACTION and r['errors']==[] and r['cache_released'] is True and r['Native_flags_assigned'] is r['formal_Goodput'] is False,'actual role terminal report')
  x=r['result'];result=x['result'];need(result['status']=='passed_subset' and result['natural_positive_and_real_child_consume'] is False and result['actual_natural_invalidation_and_nextdraft'] is True and result['full_local_scope_qualified'] is result['parent4_qualified'] is result['ordinary_deferE_production_gate_opened'] is result['adopted'] is result['formal_Goodput'] is False,'actual subset/natural-negative scope')
  graph=result['actual_graph_states'];need(graph['generic']==[] and all(v['runtime_validated'] is True for k in ('draft','target') for v in graph[k]),'actual final Native entries naturally validated')
  need(len(x['exploration'])==2 and list(map(len,x['complete_zero']))==[2,2],'two scopes/two complete zero passes')
  events=r['boundary_events'];order=[]
  for i,e in enumerate(events):
   need(e['sequence']==i and sorted(v['rank'] for v in e['ranks'])==list(range(4)) and all(v['error'] is None and v['sequence']==i and v['stage']==e['stage'] for v in e['ranks']),'common all4 actual stage transcript')
   order.append((i,e['stage']))
  stages.append(order)
  for iteration,trace in enumerate([x['exploration'],*x['complete_zero']]):
   for repair,row in enumerate(trace):
    need(row['case']['repair']==repair and row['generation']['actual_shared_steps']==4 and row['generation']['real_raw_nodes']==512 and row['generation']['selected_total']==192,'genuine shared4 canonical8 work')
    d=row['draft_numerical'];need(d['all28_written_query_KV'] is d['exact_full_physical_encoded'] is d['independent_fresh_history'] is True and d['candidate_KV_used_as_reference'] is False,'retained own device draft numerical obligations')
    p=row['protocol'];need(p['outcome']=='observed_target' and p['synthetic_verdict'] is p['parent4_qualified'] is p['adopted'] is p['Goodput_measured'] is False and p['promoted_owners']==[] and p['invalidated_owners']==list(range(32,64)) and p['actual_promoted_child_consumed'] is False and p['actual_natural_invalidation'] is True,'actual zero promotion and natural invalidation')
    need([q['label'] for q in p['phases']]==['parent3','normal-variable4'],'no promoted child phase or consume')
    need(p['current_work_exchange']['selected_candidates']==192 and p['current_work_exchange']['budgets']==list(NORMAL_B)+[3]*32 and p['current_work_exchange']['typed_budget']['normal_candidates']==p['current_work_exchange']['typed_budget']['future_eager_candidates']==96,'actual separate typed work pools')
    normalized=[]
    for phase,owners,budgets in zip(p['phases'],(list(range(32,64)),list(range(32))),([3]*32,list(NORMAL_B)),strict=True):
     need(phase['owners']==owners and len(phase['accepted'])==len(phase['outputs'])==len(phase['tickets'])==32,'actual partition rows')
     per_owner=[]
     for owner,budget,accepted,output,ticket in zip(owners,budgets,phase['accepted'],phase['outputs'],phase['tickets'],strict=True):
      need(all(type(v)is int for v in accepted+output),'integer actual verifier arrays')
      nodes=[v for v in accepted if v>=0];values=[v for v in output if v>=0];count=len(nodes)
      need(nodes==list(range(count)) and count<=budget and len(values)==count+1 and ticket['owner']==owner and ticket['gamma']==budget and ticket['accepted_nodes']==nodes and ticket['emitted_tokens']==values and ticket['promoted_proposal_id'] is None,'raw spine counts and exact original ticket record')
      if phase['label']=='parent3':need(count<3,'every actual parent fails full parent3 threshold')
      per_owner.append(dict(owner=owner,selected_B=budget,accepted_count=count,fully_accepted=count>=budget,accepted_nodes=nodes,emitted_count=len(values)))
     hist=dict(sorted(Counter(v['accepted_count'] for v in per_owner).items()))
     if phase['label']=='parent3':need(hist=={0:8,1:8,2:16} and phase['counters']['spec_rhythm_tree_eager_invalidated']==32 and phase['counters']['spec_rhythm_tree_eager_dependency_rejections']==32,'actual parent rejection histogram/counters')
     if rank:
      t=phase['target'];need(t['shape']['N']==32 and t['shape']['C']==96 and t['shape']['real_Q']==128 and t['allphysical_hidden'] is t['all128query_KV'] is t['independent_complete_history'] is True and phase['committed_KV']['all128_committed_KV'] is True,'retained original legal target and independent commit KV proof')
     normalized.append(dict(label=phase['label'],per_owner=per_owner,accepted_histogram=hist))
    key=str(repair)
    if key not in phases:phases[key]=normalized
    need(phases[key]==normalized,'per-owner acceptance stable across all ranks/exploration/zeros')
    fingerprints[(rank,iteration,repair)]=normalized
    n=p['next_draft']
    if rank==0:
     need(n['passed'] is n['own_old_shared4_history_rebuilt'] is n['all28_written_KV'] is n['known_prefix_protected'] is n['real_invalidation_or_promotion_followed_by_nextdraft'] is True and n['candidate_KV_used_as_reference'] is n['Goodput_measured'] is False,'actual own-history invalidation nextdraft numerical record')
     need([w['owners'] for w in n['normal_waves']]==[list(range(32)),list(range(32,64))] and all(w['actual_shared_steps']==3 and w['allphysical_hidden_encoded'] is w['runtime_validated'] is True and w['validated_real_row_count']>=32 for w in n['normal_waves']) and len(n['actual_missing_suffix_lengths'])==64 and all(type(v)is int and 0<=v<=1 for v in n['actual_missing_suffix_lengths']),'true next3 all64 owners and genuine untruncated missing suffix')
    else:need(n is None,'only actual draft rank executes nextdraft math')
    if iteration:
     a,b,stored=row['Native_before'],row['Native_after'],row['Native_delta'];need(set(a)==set(b)==set(stored)==set(FIELDS),'original exact21 fields')
     delta={k:b[k]-a[k] for k in FIELDS}
     need(all(type(a[k])is type(b[k])is type(stored[k])is int and 0<=a[k]<=b[k] and delta[k]==stored[k]==0 for k in FIELDS),'independent raw21 zero differences')
     zero_rows.append(dict(rank=rank,repair=repair,zero_iteration=iteration,raw_before=a,raw_after=b,independently_subtracted=delta))
 need(all(v==stages[0] for v in stages) and len(zero_rows)==16,'all4 ordering and exact16 zero rows')
 return dict(parent_full_accept_threshold=3,per_repair_case_acceptance=phases,replicated_observations=len(fingerprints),all_rank_and_repeat_acceptance_identical=True,natural_parent_invalidations_per_trace=32,natural_promotions=0,actual_promoted_child_consumptions=0,natural_invalid_nextdraft_passed=True,all4_common_stages=len(stages[0]),raw_zero_rows=zero_rows,independent_Native_zero_subtractions=16*21)

def audit(expected):
 rp=P/f'receipt-{TAG}.json';need(expected==RECEIPT_SHA and sha(rp)==RECEIPT_SHA,'exact requested retained receipt');r=read(rp)
 need(r['schema']=='V151.root-owned-local-component-supervisor.v1' and r['status']=='complete' and r['errors']==[] and r['actual_execution_complete'] is r['sources_unchanged'] is True and r['owned_remaining']==[] and r['child']['exit_code']==0 and r['elapsed_seconds']<1800 and r['actual_local_coverage_status']=='passed_subset','actual completed supervisor/source/owned cleanup')
 need(r['adopted'] is r['formal_Goodput'] is r['full_workload_qualified'] is r['production_deferE_gate_opened'] is r['parent4_qualified'] is False,'no production adoption')
 sources,frozen=audit_sources(r);pre=verifier(r['pre_verifier']);post=verifier(r['post_verifier']);before=idle(r['devices_before']);after=idle(r['devices_after'])
 directory=P/'actual'/TAG;reports=[raw_report(directory/f'rank{i}-component.json.gz') for i in range(4)]
 complete=read(directory/'completed.json');exited=read(directory/'worker-exit.json')
 need(exited==dict(closed=True,all_exited=True) and complete['action']==ACTION and complete['replies']==[['calibrated',i,dict(cases=2,complete_zeros=2,timings=0)] for i in range(4)] and complete['audit']==r['actual_component_audit'] and complete['audit']['status']=='passed_subset','actual all4 completion/exit and subset schema')
 traces=audit_traces(reports);need(r['independent_native_zero_arithmetic']==dict(counter_fields=21,complete_trace_rows=16,raw_deltas_recomputed=336,all_zero=True),'supervisor336 agrees with independent raw subtraction')
 return dict(schema='HOST.V151.actual-terminal-subset-independent-audit.v1',status='passed_subset',audit_passed=True,receipt=desc(rp),sources=sources,pre_verifier=pre,post_verifier=post,retained_devices_before=before,retained_devices_after=after,actual4workers_exited=True,owned_cleanup_complete=True,sources_unchanged_verified=True,original_finish_thresholds=finish_source(frozen),acceptance_and_zero_evidence=traces,raw_inputs=[desc(directory/name) for name in [*(f'rank{i}-component.json.gz' for i in range(4)),'completed.json','worker-exit.json','prerequisites.json']],P152_prerequisite=dict(status='passed_subset',ROOT_actual_conclusion_reviewed=False,qualification_inherited=False,scope='actual negative/invalidation+nextdraft subset only; ROOT review required before binding'),natural_positive_or_child_consumption_qualified=False,full_component_scope_qualified=False,parent4_qualified=False,production_adopted=False,production_deferE_gate_opened=False,formal_Goodput=False,dynamic_prefix_cost_measured=False,hidden_KV_recomputed_by_HOST=False,limitations=['The original two provider fixtures naturally accept only0/1/2 parent nodes; no parent accepts all3. This is a fixture coverage limit, not evidence that an additional gate rejected an otherwise fully accepted parent.','Natural invalidation and actual nextdraft were exercised. Natural promotion, positive dependency/frontier match and real promoted-child consume were not exercised and are not qualified.','Hidden/KV numerical equality and finite flags are actual frozen-runtime reports. This HOST auditor checks provenance/schema/arithmetic; it does not recompute tensors.','No online production scheduler, deferE gate, overlap, dynamic prefix timing, full463, parent4, adoption or Goodput result follows from this local sequential subset.','Current device state was not queried; idle/cleanup statements refer to retained original pre/post execution evidence.'])

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--expected-receipt-sha256',required=True);parser.add_argument('--output',required=True);a=parser.parse_args();p=Path(a.output).resolve();need(p.parent==D and not p.exists() and not p.is_symlink(),'fresh D-only audit output')
 start=time.perf_counter();row=dict(status='failed',audit_passed=False)
 try:row=audit(a.expected_receipt_sha256)
 except BaseException:row.update(error=traceback.format_exc())
 row.update(auditor=desc(__file__),UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,NPU_used=False,Torch_imported=False,model_or_constructor_called=False,CPU_or_parent_suite_rerun=False,candidate_or_frozen_source_modified=False)
 with p.open('x') as stream:json.dump(row,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(status=row['status'],audit_passed=row['audit_passed'],output=desc(p),elapsed_seconds=row['elapsed_seconds'])))
 if not row['audit_passed']:raise SystemExit(1)
if __name__=='__main__':main()
