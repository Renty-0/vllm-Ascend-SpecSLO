"""Prepared independent terminal190 audit. Never read actual until ROOT authorizes.

--source-check-only reads only frozen source inputs. The terminal mode requires
an explicit tag and receipt SHA supplied after ROOT's terminal authorization.
No model, device, backend, global verifier, or old suite is invoked.
"""
import argparse,ast,collections,copy,datetime,gzip,hashlib,importlib.abc,json,math,re,statistics,sys,tarfile,types
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-vacancy-cohort-prefill-cap4-active32-full-v190';BASE=B/'tree-high-normal-cap4-native-families-v148'
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
EXPECTED_FREEZE='622094f5426f205f995787a785f4bc11b026838d5dfb01deead0b42bf4fe9ffc'
EXPECTED_CPU='c518b8a0ee4d325bab7baf1b4672ddec5b895548982962f44324cd24866c6fe0'
EXPECTED_ARCHIVE='d17d4b7fd6770cd0bbf3f4bc9f45951db615d200e614b75b67bc07ba5ec35331'
ACTION='vacancy-cohort-prefill-cap4-active32-full463'
def need(ok,label):
 if not ok:raise ValueError(label)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def desc(p):p=Path(p);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def read(p):
 p=Path(p);data=p.read_bytes()
 if p.suffix=='.gz':data=gzip.decompress(data)
 return json.loads(data,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
class Firewall(importlib.abc.MetaPathFinder):
 attempts=[]
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors','acl'):
   self.attempts.append(name);raise ImportError('Independent190 HOST forbids '+name)
def extract(path,name,ns):
 node=copy.deepcopy(next(n for n in ast.walk(ast.parse(Path(path).read_text()))if isinstance(n,ast.FunctionDef)and n.name==name));node.decorator_list=[]
 unit=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
 exec(compile(ast.fix_missing_locations(unit),str(path)+'::independent-HOST','exec'),ns);return ns[name]
def source_binding(*,check_archive=False):
 need(sha(P/'FROZEN_SOURCE190.json')==EXPECTED_FREEZE and sha(P/'CPU_RESULT190.json')==EXPECTED_CPU,'exact prepared190 freeze/CPU')
 f=read(P/'FROZEN_SOURCE190.json');pins={}
 for name in('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):
  for path,h in f[name].items():need(path not in pins or pins[path]==h,'consistent source pin');pins[path]=h
 pins.update({str(P/'CPU_RESULT190.json'):EXPECTED_CPU,str(P/'FROZEN_SOURCE190.json'):EXPECTED_FREEZE,f['root_review']['path']:f['root_review']['sha256']})
 need(len(pins)==271 and all(sha(p)==h for p,h in pins.items()),'all current frozen271 source bytes')
 source_archive=P/'PREPARED_SOURCE_ARCHIVE190.tar.gz';need(sha(source_archive)==EXPECTED_ARCHIVE,'prepared271 source archive SHA')
 if check_archive:
  with tarfile.open(source_archive,'r:gz')as t:
   members=t.getmembers();need(len(members)==len(pins)==len({m.name for m in members}),'exact unique source tar inventory')
   for m in members:need(m.isfile()and '/'+m.name in pins and hashlib.sha256(t.extractfile(m).read()).hexdigest()==pins['/'+m.name],'regular exact source member '+m.name)
 return dict(freeze=desc(P/'FROZEN_SOURCE190.json'),CPU=desc(P/'CPU_RESULT190.json'),archive=dict(**desc(source_archive),members=271,all_regular_member_bytes_checked=check_archive),selected_current_bytes_exact=True),f
class Value:
 """Explicit HOST list carrier for original integer verifier, not model tensors."""
 def __init__(self,v):self.values=v
 @property
 def ndim(self):return 1
 def numel(self):return len(self.values)
 def detach(self):return self
 def cpu(self):return self
 def tolist(self):return copy.deepcopy(self.values)
 def __getitem__(self,k):
  if isinstance(k,tuple):return Value([r[k[1]]for r in self.values[k[0]]])
  return Value(self.values[k])
def original_verifier():
 parents=extract(NATIVE/'tree.py','_spine_first_topology',{})(2,4)[0]
 node=copy.deepcopy(next(n for n in ast.parse((NATIVE/'tree.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='pack_selected_tree_plan'));node.decorator_list=[]
 end=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Assign)and any(isinstance(z,ast.Name)and z.id=='device'for z in n.targets));node.body=node.body[:end]+[ast.Return(value=ast.Name(id='new_parents',ctx=ast.Load()))]
 carrier=types.SimpleNamespace(Tensor=Value,tensor=lambda v,**kw:Value(v),long='long');ns=dict(torch=carrier)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])),'original-packed-ancestry-prefix190','exec'),ns)
 verify=extract(NATIVE/'native_engine.py','_verify_tree_outputs_host',dict(torch=carrier,TreeVerificationOutput=lambda a,b:types.SimpleNamespace(token_ids=a,accepted_node_indices=b)))
 def run(rows,selected,heads):
  plans=[types.SimpleNamespace(candidate_budget=len(ids),parent_indices=Value(ns['pack_selected_tree_plan'](types.SimpleNamespace(parent_indices=Value(parents),candidate_budget=8),ids)))for ids in selected]
  q=sum(len(ids)+1 for ids in selected);out=verify(rows,plans,Value(heads[:q]),4,'explicit-HOST-values')
  return [[v for v in row if v>=0]for row in out.token_ids.values],out.accepted_node_indices.values
 return run
def retained_baseline_and_outer(r,A):
 need(r['status']=='complete'and r['child']['exit_code']==0 and r['errors']==[]and r['sources_unchanged']is True and r['owned_remaining']==[],'complete clean unchanged terminal, not a partial/failed run')
 need(r['readiness']['freeze_sha256']==EXPECTED_FREEZE and r['readiness']['CPU_sha256']==EXPECTED_CPU,'actual process frozen source/CPU binding')
 need(r['readiness']['archive']['sha256']==EXPECTED_ARCHIVE and r['readiness']['archive']['members']==271,'actual process prepared source tar')
 need(r['execution_lock']['nonblocking_exclusive']is True and r['elapsed_seconds']<7200 and r['limits']==dict(child_execution_seconds=7200,actual_tag_bytes=256*1024**2,aggregate_private_bytes=512*1024**2),'selflock and bounded limits')
 for key in('pre_verifier','post_verifier'):
  v=r[key];actual=read(v['path']);need(sha(v['path'])==v['sha256']and v['exit_code']==0 and actual==v['actual_result']and actual['status']=='pass'and actual['errors']==[]and(actual['checked_frozen_files'],actual['checked_external_files'],actual['checked_packages'])==(10496,15,212),'retained original10496/15/212 '+key)
 for key in('devices_before','devices_after'):
  v=r[key];need(v['all_idle']is v['four_910B2']is True and set(v['processes'])=={'2','5','6','7'},'retained all4 idle '+key)
  need(all(x['exit_code']==0 and x['idle']is True and x['output'].strip()=='No process in device.'for x in v['processes'].values()),'saved idle outputs '+key)
 need(read(A/'worker-exit.json')==dict(closed=True,all_exited=True)and not list(A.glob('rank*-last-error.json')),'all workers closed and no errors')
def pct(values):
 values=sorted(values);x=(len(values)-1)*.9;lo,hi=math.floor(x),math.ceil(x)
 return dict(median_ms=statistics.median(values),P90_ms=values[lo]+(values[hi]-values[lo])*(x-lo))
def independent_clock_metrics(trace,retained):
 source=D/'analyze_completed_tree_full463_cost.py';ns=dict(math=math,statistics=statistics,CLASSES=('tight','normal','loose'));extract(source,'need',ns);metrics=extract(source,'clocks',ns)(trace)
 first={}
 for e in trace['stream_events']:
  i=e['request_index']
  if i not in first:need('first_token_wall_seconds160'in e,'original actual first-token wall');first[i]=e['first_token_wall_seconds160']
  else:need('first_token_wall_seconds160'not in e,'one actual first-token callback')
 need(set(first)==set(range(463))and all(row['arrival_ts']==trace['absolute_arrivals'][i]for i,row in enumerate(trace['generate_results'])),'all463 original arrivals/first callbacks')
 ttft=[1000*(first[i]-row['arrival_ts'])for i,row in enumerate(trace['generate_results'])];need(all(math.isfinite(v)and v>=0 for v in ttft),'finite nonnegative TTFT')
 need(all(retained[k]==v for k,v in metrics.items()),'independent Native charged/event wall/raw/fullwall arithmetic')
 TTFT=dict(samples=463,**pct(ttft),by_class={name:pct([ttft[i]for i,row in enumerate(trace['generate_results'])if row['slo_class']==name])for name in('tight','normal','loose')})
 need(retained['TTFT160']['median_ms']==TTFT['median_ms']and retained['TTFT160']['P90_ms']==TTFT['P90_ms'],'independent callback TTFT arithmetic')
 return dict(metrics,TTFT_independent=TTFT)
def additional_component_integer_audit(A):
 verify=original_verifier();counts=dict(B4_raw_cycles=0,B4_original_verifier_calls=0,B4_owner_verdicts=0,B4_zero_scalars=0,M80_original_verifier_calls=0,M80_owner_verdicts=0,M80_zero_scalars=0,new_multiowner_zero_scalars=0)
 base=A/'variable-B4-own-full-process'
 for rank in range(4):
  report=read(base/f'rank{rank}-component.json.gz')
  for tr in report['traces']:
   if tr['pass_index']:need(not any(tr['Native']['delta'].values()),'base complete protocol zero');counts['B4_zero_scalars']+=len(tr['Native']['delta'])
   for cycle in range(4):
    row=read(base/f"rank{rank}-pass{tr['pass_index']}-case{tr['case']}-cycle{cycle}.json.gz");counts['B4_raw_cycles']+=1
    need(row['commit']['actual_logical_tokens']==[v['token_ids']+new for v,new in zip(row['source']['contexts'],row['commit']['committed_tokens'],strict=True)],'B4 genuine recurrent history')
    if rank:
     tokens,paths=verify(row['rows'],row['selected'],row['target']['greedy'][2]['full_physical_tokens']);need(tokens==row['commit']['committed_tokens']and paths==row['commit']['accepted_paths'],'original B4 ancestry/verdict from actual saved head')
     counts['B4_original_verifier_calls']+=1;counts['B4_owner_verdicts']+=len(row['rows'])
 own=A/'causal80-own-current-process';cross=[]
 for rank in range(4):
  report=read(own/f'rank{rank}-component.json.gz')
  for tr in report['traces']:
   if tr['pass_index']:need(not any(tr['Native']['delta'].values()),'M80 complete actual protocol zero');counts['M80_zero_scalars']+=len(tr['Native']['delta'])
  if rank:
   for p in range(3):
    for cycle in range(4):
     row=read(own/f'rank{rank}-pass{p}-cycle{cycle}.json.gz');cm=row['crossM']
     for which in('old','new'):
      tokens,paths=verify(row['rows'],row['selected'],cm[which+'_real80']);need(tokens==[[v for v in line if v>=0]for line in cm[which+'_verdict_tokens']]and paths==cm[which+'_accepted'],'retained all80 original Native verifier '+which)
      if which=='new':need(tokens==row['commit']['committed_tokens']and paths==row['commit']['accepted_paths'],'new80 actual commit exact')
      counts['M80_original_verifier_calls']+=1;counts['M80_owner_verdicts']+=16
     cross.append(dict(rank=rank,pass_index=p,cycle=cycle,head_equal=cm['head_equal'],verdict_equal=cm['verdict_equal'],accepted_equal=cm['accepted_equal'],different_head_positions=cm['different_head_positions']))
 extra=A/'multiowner-eager-prefill-own-current-process'
 for rank in range(4):
  for p in range(3):
   for n in(2,4):
    row=read(extra/f'rank{rank}-N{n}-pass{p}.json.gz');need(row['N']==n and row['pass_index']==p and row['publication']['original_callback']is True,'actual retained multiowner case')
    if p:need(not any(row['Native']['delta'].values()),'new component both full zero sweeps');counts['new_multiowner_zero_scalars']+=len(row['Native']['delta'])
    need(row['next_draft']['selected']==[list(range(4))]*n and all(len(v)==4 for v in row['next_draft']['rows']),'published root feeds real next B4')
    if rank==0:
     proof=row['next_draft']['draft_proof'];need(proof['actual_draft_model_stages']==4 and proof['raw_encoded_shape']==[n,4,2]and proof['raw_pool_nodes']==8*n and proof['exact_full8_encoded']is True,'next4 full8/stage4 actual same-M device marker')
 need(counts['B4_raw_cycles']==96 and counts['B4_original_verifier_calls']==72 and counts['B4_owner_verdicts']==2304 and counts['B4_zero_scalars']==336,'all original B4 domain')
 need(counts['M80_original_verifier_calls']==72 and counts['M80_owner_verdicts']==1152 and counts['M80_zero_scalars']==168 and counts['new_multiowner_zero_scalars']==336,'current80/newprefill complete zeros and original integer verifier scope')
 return dict(counts=counts,M80_crossM_comparisons=cross,full_hidden_KV_recomputed_on_HOST=False)
def prefill_and_admission(trace,envelope):
 from trace_admission190 import audit_record
 row=envelope['admission190'];saved=read(row['path']);need(desc(row['path'])['sha256']==row['sha256']and Path(row['path']).stat().st_size==row['bytes']and saved==row['record'],'actual admission saved bytes/record')
 replay=audit_record(saved);events=[e for e in saved['events']if e['kind']=='decision'];release=[e for e in events if e['release']];deferred=[e for e in events if not e['release']]
 calls=trace['standalone_prefill_calls'];actual_hist=dict(sorted(collections.Counter(str(len(c['sequence_ids']))for c in calls).items()));release_hist=dict(sorted(collections.Counter(str(len(e['ready_ids']))for e in release).items()))
 need(actual_hist==release_hist==dict(sorted(replay['cohort_size_histogram'].items())),'actual prefill cohort histogram equals independent policy releases')
 owners=[i for call in calls for i in call['sequence_ids']];released=[i for e in release for i in e['ready_ids']]
 need(sorted(owners)==sorted(released)==list(range(463))and len(owners)==len(set(owners)),'each463 true prefill and release exactlyonce')
 for e in events:need(e['original_arrivals']==[trace['absolute_arrivals'][i]for i in e['ready_ids']]and all(a<=e['leader_now']for a in e['original_arrivals']),'unchanged real arrival predicate in actual policy inputs')
 c=trace['decode_counters'];need(c['spec_rhythm_slo_aware_admission_enabled']==c['spec_rhythm_target_slo_first_enabled']==0 and c['spec_rhythm_prefill_coalesce_enabled']==1,'actual policy0/0 enabled4/150')
 need(c['spec_rhythm_prefill_coalesce_deferred_polls']==len(deferred)and c['spec_rhythm_prefill_batches']==len(calls)and c['spec_rhythm_prefill_requests']==463,'actual deferred/batch/request count arithmetic')
 need(c['spec_rhythm_prefill_coalesce_size_releases']==sum(e['reason']=='size'for e in release)and c['spec_rhythm_prefill_coalesce_timeout_releases']==sum(e['reason']=='timeout'for e in release),'actual size and timeout release counters')
 need(not trace['logical_events']and not trace['begin_inputs'],'no mixed transaction or precomputed prefix')
 return dict(actual_prefill_calls=len(calls),actual_cohort_size_histogram=actual_hist,actual_prefill_prompt_queries=sum(x['uncached_queries']for x in calls),actual_deferred_polls=len(deferred),actual_release_reasons=dict(collections.Counter(e['reason']for e in release)),scalar_policy_replay=replay,all463_prefilled_once=True,minimum4_is_not_maximum4=True,maximum_actual_cohort=max(map(int,actual_hist)),actual_prefill_HOST_seconds=c['spec_rhythm_online_refill_ms']/1000,HOST_seconds_not_isolated_device_time=True,extra_arrival_or_TPOT_clock_added=False)
def completed173_comparison(metrics,geometry,traces,fast):
 path=D/'V173_INDEPENDENT_TERMINAL_FULL463_AUDIT.json';old=read(path)
 need(old['status']=='passed_terminal_component_and_full463_audit','completed173 independent terminal audit')
 manifest=read(old['provenance']['raw_manifest']['path']);q=Path(old['candidate'])/'actual'/old['tag']/'windows/window32/full463/rank1-full463-actual-trace.json.gz'
 need(str(q)in manifest and sha(q)==manifest[str(q)]['sha256']and q.stat().st_size==manifest[str(q)]['bytes'],'completed173 exact retained full leader trace')
 raw=read(q);old_calls=raw['standalone_prefill_calls'];hist=dict(sorted(collections.Counter(str(len(c['sequence_ids']))for c in old_calls).items()))
 previous=dict(metrics=old['full463_metrics'],prefill_calls=len(old_calls),cohort_size_histogram=hist,prefill_prompt_queries=sum(c['uncached_queries']for c in old_calls),prefill_HOST_seconds=raw['decode_counters']['spec_rhythm_online_refill_ms']/1000,deferred_polls=raw['decode_counters']['spec_rhythm_prefill_coalesce_deferred_polls'],new80_calls=old['geometry']['new80_calls'],new80_fraction=old['geometry']['new80_fraction'],target_calls=old['geometry']['target_calls'],verified_owner_rounds=old['geometry']['verified_owner_rounds'],physical_queries=old['geometry']['physical_Q'],real_queries=old['geometry']['real_Q'],draft_model_submissions=sum(old['actual_FastV5_delta'].get(k,0)for k in('chain_calls','repair_calls','eager_frontier_calls','root_seed_calls')),FastV5=old['actual_FastV5_delta'],HOST_phase_seconds=old['HOST_phase_seconds'])
 current=dict(metrics=metrics,prefill=geometry['standalone_prefill'],new80_calls=geometry['new80_calls'],new80_fraction=geometry['new80_fraction'],target_calls=geometry['target_calls'],verified_owner_rounds=geometry['verified_owner_rounds'],physical_queries=geometry['physical_queries'],real_queries=geometry['real_queries'],draft_model_submissions=geometry['draft_model_submissions'],FastV5=fast,HOST_phase_seconds={str(i):t['phase_seconds']for i,t in enumerate(traces)})
 return dict(completed173_audit=desc(path),completed173_raw_leader=desc(q),baseline=previous,current=current,observed_deltas=dict(full_generate_seconds=metrics['host_generate_wall_seconds']-old['full463_metrics']['host_generate_wall_seconds'],raw_tokens_per_second=metrics['raw_tokens_per_second']-old['full463_metrics']['raw_tokens_per_second'],Native_paper_Goodput=metrics['Native_paper']['Goodput']-old['full463_metrics']['Native_paper']['Goodput'],prefill_HOST_seconds=geometry['standalone_prefill']['actual_prefill_HOST_seconds']-previous['prefill_HOST_seconds'],prefill_calls=geometry['standalone_prefill']['actual_prefill_calls']-previous['prefill_calls'],target_HOST_seconds=traces[1]['phase_seconds']['target']-old['HOST_phase_seconds']['1']['target'],draft_model_submissions=geometry['draft_model_submissions']-previous['draft_model_submissions'],M80_call_fraction=geometry['new80_fraction']-previous['new80_fraction']),causal_attribution_established=False,HOST_phase_durations_must_not_be_added_as_device_time=True,actual_cohorts_prefixes_target_budget_overlap_and_numerical_shape_can_all_differ=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--source-check-only',action='store_true');parser.add_argument('--tag');parser.add_argument('--expected-receipt-sha256');parser.add_argument('--output',type=Path);a=parser.parse_args()
 need(sys.version_info[:2]==(3,11),'retained Py311');sys.meta_path.insert(0,Firewall())
 provenance,f=source_binding(check_archive=not a.source_check_only)
 if a.source_check_only:
  print(json.dumps(dict(status='prepared_source_only',candidate=str(P),source=provenance,actual190_read=False,Torch_imported=False,NPU_used=False,model_constructed=False)));return
 need(a.tag is not None and re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',a.tag)and a.expected_receipt_sha256 is not None and re.fullmatch('[0-9a-f]{64}',a.expected_receipt_sha256),'ROOT terminal authorization supplies explicit tag+receipt SHA')
 need(a.output is not None and not a.output.exists(),'fresh output path');A=P/'actual'/a.tag;rp=P/f'receipt-{a.tag}.json';need(sha(rp)==a.expected_receipt_sha256,'ROOT exact terminal receipt SHA');r=read(rp);retained_baseline_and_outer(r,A)
 sys.path.insert(0,str(P));import audit190
 actual=audit190.audit(A,(32,));completed=read(A/'completed.json');need(actual['passed']is True and actual==completed['audit']==r['actual_full_audit'],'reopened actual HOST audit equals completed and final guard')
 need(completed['action']==ACTION and completed['replies']==[['calibrated',i,dict(windows=[32],full463=True)]for i in range(4)],'four original successful replies')
 reports=[read(A/f'rank{i}-local-diagnostic.json')for i in range(4)];envelopes=[x['windows']['traces'][-1]['original_trace']for x in reports];traces=[read(x['artifact']['path'])for x in envelopes]
 need(all(x['role']=='full463'and x['sealed_before']is x['sealed_after']is x['cache_released']is True for x in traces),'actual sealed released full463 all4')
 need(all(x['admission_window160']==32 and x['configured_capacity']==64 and 0<x['active_peak']<=32 for x in traces),'actual window32/configured64')
 extra=additional_component_integer_audit(A);t=traces[1];draft=traces[0];retained=reports[1]['windows']['per_rank_full_metrics'][0]['output'];metrics=independent_clock_metrics(t,retained)
 admission=[prefill_and_admission(trace,envelope)for trace,envelope in zip(traces,envelopes,strict=True)];need(all(x==admission[0]for x in admission),'all4 actual cohort/arrival/replay/counters')
 calls=t['normal_target_calls'];N=sum(x['N']for x in calls);C=sum(sum(x['budgets'])for x in calls);physical=sum(x['physical_M']for x in calls);real=sum(x['real_query_count']for x in calls);c=t['decode_counters'];need(C==c['spec_rhythm_verified_tokens']and physical==c['spec_rhythm_target_query_tokens']and physical-real==c['spec_rhythm_target_graph_padding_queries'],'actual target call geometry/counter arithmetic')
 work=reports[0]['windows']['traces'][-1]['actual_split_work_counts']['delta'];fast=work['FastV5'];dc=draft['decode_counters'];need(dc['spec_rhythm_draft_model_calls']==sum(fast.get(k,0)for k in('chain_calls','repair_calls','eager_frontier_calls','root_seed_calls')),'actual draft submission categories')
 selected80=[x for x in calls if (x['N'],x['real_query_count'],x['family'])==(16,80,'causal')]
 need(all((x['physical_M'],x['physical_S'])==(80,16)for x in selected80),'all actual new80 routes')
 geometry=dict(new80_calls=len(selected80),new80_fraction=len(selected80)/len(calls),target_calls=len(calls),verified_owner_rounds=N,verified_candidates=C,physical_queries=physical,real_queries=real,padding_queries=physical-real,N_hist=dict(collections.Counter(str(x['N'])for x in calls)),shapes=[dict(M=m,S=s,family=f,calls=n)for(m,s,f),n in sorted(collections.Counter((x['physical_M'],x['physical_S'],x['family'])for x in calls).items())],standalone_prefill=admission[0],draft_model_submissions=dc['spec_rhythm_draft_model_calls'],eager_promoted=dc['spec_rhythm_tree_eager_promoted'],eager_invalidated=dc['spec_rhythm_tree_eager_invalidated'])
 comparison=completed173_comparison(metrics,geometry,traces,fast)
 manifest={str(p):dict(bytes=p.stat().st_size,sha256=sha(p))for p in sorted(A.rglob('*'))if p.is_file()};mp=a.output.with_name(a.output.stem+'_RAW_MANIFEST.json');need(not mp.exists(),'fresh raw manifest');mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 need(not Firewall.attempts and not any(n.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend')for n in sys.modules),'no backend imports')
 out=dict(schema='V190.independent-terminal-prefill-component-and-full463-audit.v1',status='passed_terminal_component_and_full463_audit',candidate=str(P),tag=a.tag,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),receipt=desc(rp),script=desc(__file__),provenance=dict(provenance,raw_manifest=dict(**desc(mp),files=len(manifest),bytes_total=sum(x['bytes']for x in manifest.values())),original_prepost_10496_15_212_passed=True,owned_remaining=[],saved_prepost_all4idle=True,worker_exit=desc(A/'worker-exit.json'),device_requeried=False),original_unmodified_current_audit=dict(source=desc(P/'audit190.py'),result=actual,equals_runtime_and_guard=True),additional_component_integer_audit=extra,full463_metrics=metrics,geometry=geometry,actual_prefill_and_admission=admission[0],completed173_comparison=comparison,actual_split_policy_delta=work['policy'],actual_FastV5_delta=fast,HOST_phase_seconds={str(i):x['phase_seconds']for i,x in enumerate(traces)},early_target_HOST_seconds={k:v/1e9 for k,v in c.items()if k.endswith('_ns')and'early_target'in k},limitations=['Actual current model matcher gates are retained markers/source/head evidence; full hidden/KV tensor values are not reconstructed on HOST.','Current multiowner N2/4 numerical representatives are finite, not every cohort/prompt tuple up to32.','Native charged TPOT, event-wall TPOT, callback TTFT and HOST phases are separate; HOST durations overlap and are not added as isolated device work.','A150ms vacancy polling episode is not a strict TTFT bound; original arrivals and actual completion/clock audit remain primary.','No whole-workload independent AR equality, production adoption or optimization-goal achievement is asserted.'],Torch_imported=False,NPU_used=False,model_constructed=False,old_suites_rerun=False,formal_Goodput=False,adopted=False,goal_achieved=False)
 with a.output.open('x')as stream:json.dump(out,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(output=desc(a.output),status=out['status'],wall=metrics['host_generate_wall_seconds'],raw=metrics['raw_tokens_per_second'],Goodput=metrics['Native_paper']['Goodput'],actual_prefill_calls=admission[0]['actual_prefill_calls'],cohort_histogram=admission[0]['actual_cohort_size_histogram'],deferred_polls=admission[0]['actual_deferred_polls'])))
if __name__=='__main__':main()
