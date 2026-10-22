"""Independent completed176 HOST audit; no model, device, or old suite run."""
import argparse,ast,collections,copy,gzip,hashlib,importlib.abc,json,math,statistics,sys,tarfile,types
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-high-normal-cap3-active32-full-v176';TAG='high-normal-cap3-v176-full463-v1';A=P/'actual'/TAG
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
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
   self.attempts.append(name);raise ImportError('Independent176 HOST '+name)
def extracted(path,name,ns):
 n=copy.deepcopy(next(n for n in ast.walk(ast.parse(Path(path).read_text()))if isinstance(n,ast.FunctionDef)and n.name==name));n.decorator_list=[]
 unit=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),n],type_ignores=[])
 exec(compile(ast.fix_missing_locations(unit),str(path)+'::independent-HOST','exec'),ns);return ns[name]
class Value:
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
def pct(values):
 a=sorted(values);x=(len(a)-1)*.9;lo=math.floor(x);hi=math.ceil(x)
 return dict(median_ms=statistics.median(a),P90_ms=a[lo]+(a[hi]-a[lo])*(x-lo))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--expected-receipt-sha256',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 need(not args.output.exists(),'fresh independent output');rp=P/f'receipt-{TAG}.json';need(sha(rp)==args.expected_receipt_sha256,'exact caller terminal receipt')
 r=read(rp);f=read(P/'FROZEN_SOURCE176.json');need(r['status']=='complete'and r['child']['exit_code']==0 and r['errors']==[]and r['sources_unchanged']is True and r['owned_remaining']==[],'successful unchanged clean terminal')
 need(r['readiness']['freeze_sha256']==sha(P/'FROZEN_SOURCE176.json')=='5e2bfc3ae1ce9bb9d0553f2ed2ec3b3457a9f6799b1820cc4d800e18df8de319'and r['readiness']['CPU_sha256']==sha(P/'CPU_RESULT176.json'),'source freeze/CPU')
 pins={}
 for k in('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):pins.update(f[k])
 pins.update({str(P/'CPU_RESULT176.json'):f['CPU_sha256'],str(P/'FROZEN_SOURCE176.json'):sha(P/'FROZEN_SOURCE176.json'),f['root_review']['path']:f['root_review']['sha256']})
 need(all(sha(p)==h for p,h in pins.items()),'all exact current selected source bytes')
 ar=r['readiness']['archive'];need(sha(ar['path'])==ar['sha256'],'archive SHA')
 with tarfile.open(ar['path'],'r:gz')as tar:
  members=tar.getmembers();need(len(members)==len(pins)==ar['members']==162 and len({m.name for m in members})==162,'exact162member archive')
  for m in members:need(m.isfile()and '/'+m.name in pins and hashlib.sha256(tar.extractfile(m).read()).hexdigest()==pins['/'+m.name],'regular archive exact bytes')
 for k in('pre_verifier','post_verifier'):
  v=r[k];q=read(v['path']);need(sha(v['path'])==v['sha256']and v['exit_code']==0 and q==v['actual_result']and q['status']=='pass'and q['errors']==[]and(q['checked_frozen_files'],q['checked_external_files'],q['checked_packages'])==(10496,15,212),'retained original baseline pre/post')
 for k in('devices_before','devices_after'):
  v=r[k];need(v['all_idle']is v['four_910B2']is True and set(v['processes'])=={'2','5','6','7'},'retained four idle')
  need(all(x['exit_code']==0 and x['idle']is True and x['output'].strip()=='No process in device.'for x in v['processes'].values()),'retained actual saved idle commands')
 need(read(A/'worker-exit.json')==dict(closed=True,all_exited=True)and not list(A.glob('rank*-last-error.json')),'all workers closed/no errors')
 manifest={str(p):dict(bytes=p.stat().st_size,sha256=sha(p))for p in sorted(A.rglob('*'))if p.is_file()}
 mp=args.output.with_name(args.output.stem+'_RAW_MANIFEST.json');need(not mp.exists(),'fresh raw manifest');mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 sys.meta_path.insert(0,Firewall());sys.path.insert(0,str(P))
 import audit176
 actual=audit176.audit(A,(32,));completed=read(A/'completed.json')
 need(actual['passed']is True and completed['audit']==actual==r['actual_full_audit'],'unmodified actual original176 audit independently matches completed and supervisor')
 need(completed['action']=='high-normal-cap3-active32-full463'and completed['replies']==[['calibrated',i,dict(windows=[32],full463=True)]for i in range(4)],'exact current action/all4 replies')
 reports=[read(A/f'rank{i}-local-diagnostic.json')for i in range(4)]
 traces=[read(A/f'windows/window32/full463/rank{i}-full463-actual-trace.json.gz')for i in range(4)];t=traces[1];draft=traces[0]
 need(all(x['sealed_before']is x['sealed_after']is x['cache_released']is True for x in traces),'all4 sealed released full')
 # Exact original topology packing plus unmodified Native host verifier on actual retained target heads.
 parents=extracted(NATIVE/'tree.py','_spine_first_topology',{})(2,4)[0]
 node=copy.deepcopy(next(n for n in ast.parse((NATIVE/'tree.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='pack_selected_tree_plan'));node.decorator_list=[]
 stop=next(i for i,n in enumerate(node.body)if isinstance(n,ast.Assign)and any(isinstance(z,ast.Name)and z.id=='device'for z in n.targets));node.body=node.body[:stop]+[ast.Return(value=ast.Name(id='new_parents',ctx=ast.Load()))]
 ht=types.SimpleNamespace(Tensor=Value,tensor=lambda x,**kw:Value(x),long='long');pns=dict(torch=ht)
 exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])),'original-packing-prefix-only','exec'),pns)
 vns=dict(torch=ht,TreeVerificationOutput=lambda a,b:types.SimpleNamespace(token_ids=a,accepted_node_indices=b));verify=extracted(NATIVE/'native_engine.py','_verify_tree_outputs_host',vns)
 C=A/'variable-B4-own-full-process';verdicts=owners=cycles=zeros=0
 component_reports=[read(C/f'rank{i}-component.json.gz')for i in range(4)]
 for rank,report in enumerate(component_reports):
  for tr in report['traces']:
   if tr['pass_index']:zeros+=len(tr['Native']['delta']);need(not any(tr['Native']['delta'].values()),'component actual complete21zero')
   for cycle in range(4):
    row=read(C/f"rank{rank}-pass{tr['pass_index']}-case{tr['case']}-cycle{cycle}.json.gz");cycles+=1
    need(row['commit']['actual_logical_tokens']==[old['token_ids']+new for old,new in zip(row['source']['contexts'],row['commit']['committed_tokens'],strict=True)],'actual logical commit independent arithmetic')
    if not rank:continue
    plans=[types.SimpleNamespace(candidate_budget=len(ids),parent_indices=Value(pns['pack_selected_tree_plan'](types.SimpleNamespace(parent_indices=Value(parents),candidate_budget=8),ids)))for ids in row['selected']]
    Q=sum(len(ids)+1 for ids in row['selected']);heads=row['target']['greedy'][2]['full_physical_tokens'];out=verify(row['rows'],plans,Value(heads[:Q]),4,'explicit-HOST-values')
    emitted=[[v for v in line if v>=0]for line in out.token_ids.values]
    need(emitted==row['commit']['committed_tokens']and out.accepted_node_indices.values==row['commit']['accepted_paths'],'actual original selected ancestry verdict')
    verdicts+=1;owners+=len(plans)
 need((cycles,verdicts,owners,zeros)==(96,72,2304,336),'all4 component raw/verdict/zero domain')
 # Fresh clock reconstruction from actual463 rows and463 first callbacks.
 cp=D/'analyze_completed_tree_full463_cost.py';ns=dict(math=math,statistics=statistics,CLASSES=('tight','normal','loose'))
 extracted(cp,'need',ns);metrics=extracted(cp,'clocks',ns)(t)
 first={}
 for event in t['stream_events']:
  i=event['request_index']
  if i not in first:need('first_token_wall_seconds160'in event,'actual first wall');first[i]=event['first_token_wall_seconds160']
  else:need('first_token_wall_seconds160'not in event,'one first callback sample')
 need(set(first)==set(range(463)),'all463 first callbacks')
 need(all(row['arrival_ts']==t['absolute_arrivals'][i]for i,row in enumerate(t['generate_results'])),'unchanged rebased actual arrivals')
 ttft=[(first[i]-row['arrival_ts'])*1000 for i,row in enumerate(t['generate_results'])];need(all(math.isfinite(v)and v>=0 for v in ttft),'finite nonnegative TTFT')
 metrics['TTFT_independent']=dict(samples=463,**pct(ttft),by_class={cl:pct([ttft[i]for i,row in enumerate(t['generate_results'])if row['slo_class']==cl])for cl in('tight','normal','loose')})
 retained=reports[1]['windows']['per_rank_full_metrics'][0]['output']
 need(all(retained[k]==v for k,v in metrics.items()if k!='TTFT_independent'),'retained metrics equals independent same-clock reconstruction')
 need(retained['TTFT160']['median_ms']==metrics['TTFT_independent']['median_ms']and retained['TTFT160']['P90_ms']==metrics['TTFT_independent']['P90_ms'],'independent TTFT')
 calls=t['normal_target_calls'];N=sum(c['N']for c in calls);Cnodes=sum(sum(c['budgets'])for c in calls);physical=sum(c['physical_M']for c in calls);real=sum(c['real_query_count']for c in calls)
 need(Cnodes==t['decode_counters']['spec_rhythm_verified_tokens']and physical==t['decode_counters']['spec_rhythm_target_query_tokens']and physical-real==t['decode_counters']['spec_rhythm_target_graph_padding_queries'],'actual geometry/counter arithmetic')
 coverage=[]
 for lo,hi,newM in [(9,16,64),(17,20,80),(21,24,96)]:
  selected=[c for c in calls if lo<=c['N']<=hi];uniform=[c for c in selected if all(b==3 for b in c['budgets'])]
  hist=collections.Counter((c['N'],c['physical_M'],c['physical_S'])for c in uniform)
  coverage.append(dict(N_min=lo,N_max=hi,all_calls=len(selected),uniform_B3_calls=len(uniform),uniform_physical=sum(c['physical_M']for c in uniform),uniform_real=sum(c['real_query_count']for c in uniform),uniform_dummy=sum(c['physical_M']-c['real_query_count']for c in uniform),actual_shapes=[dict(N=n,M=m,S=s,calls=v)for(n,m,s),v in sorted(hist.items())],hypothetical_same_list_M=newM,hypothetical_physical=len(uniform)*newM,fixed_list_volume_only_not_latency_or_new_trajectory=True))
 env=reports[0]['windows']['traces'][-1];work=env['actual_split_work_counts']['delta'];p=work['policy'];fast=work['FastV5'];dc=draft['decode_counters'];proposal=env['actual_proposal_counts']['delta']
 eager=dict(generated_rows=p['actual_draft_eager_rows'],promoted_rows=dc['spec_rhythm_tree_eager_promoted'],invalidated_rows=dc['spec_rhythm_tree_eager_invalidated'],selected_future_nodes=p['split_draft_final_eager_candidates'],normal_rows=p['actual_draft_normal_rows'],normal_nodes=p['split_draft_final_normal_candidates'],draft_work_rows=p['actual_draft_work_rows'],selected_total_draft_nodes=dc['spec_rhythm_allocated_draft_tokens'],consumed_rows_from_target_minus_normal=N-p['actual_draft_normal_rows'],consumed_nodes_from_target_minus_normal=Cnodes-p['split_draft_final_normal_candidates'])
 need(eager['generated_rows']==eager['promoted_rows']+eager['invalidated_rows']==fast['eager_frontier_real_queries']and eager['selected_future_nodes']==3*eager['generated_rows'],'future generator conservation')
 need(eager['consumed_rows_from_target_minus_normal']==eager['promoted_rows']and eager['consumed_nodes_from_target_minus_normal']==3*eager['promoted_rows'],'source-bound aggregate actual consume conservation')
 need(eager['selected_total_draft_nodes']==eager['normal_nodes']+eager['selected_future_nodes']==dc['spec_rhythm_tree_nodes']and N==sum(proposal['actual_target_B'+str(b)]for b in range(1,6)),'whole draft/proposal work conservation')
 eager['promoted_fraction']=eager['promoted_rows']/eager['generated_rows'];eager['consumption_evidence']='Source-bound aggregate conservation, not a serialized individual parent-child ticket join.'
 phases={str(i):x['phase_seconds']for i,x in enumerate(traces)}
 geometry=dict(target_calls=len(calls),mean_N=N/len(calls),verified_owner_rounds=N,verified_candidate_nodes=Cnodes,physical_Q=physical,real_Q=real,padding_Q=physical-real,padding_fraction=(physical-real)/physical,effective_outputs_excluding_initial_per_owner_round=118065/N,N_hist=dict(sorted(collections.Counter(c['N']for c in calls).items())),budget_owner_rounds={str(b):proposal['actual_target_B'+str(b)]for b in range(1,6)},actual_shapes=[dict(M=m,S=s,family=f,calls=n)for(m,s,f),n in sorted(collections.Counter((c['physical_M'],c['physical_S'],c['family'])for c in calls).items())],uniform_B3_coverage=coverage,mixed_calls=0,standalone_calls=len(t['standalone_prefill_calls']),standalone_prompt_Q=sum(c['uncached_queries']for c in t['standalone_prefill_calls']),standalone_HOST_seconds=t['decode_counters']['spec_rhythm_online_refill_ms']/1000)
 need(geometry['standalone_prompt_Q']==203850 and geometry['mixed_calls']==len(t['logical_events'])==0,'all prompts standalone')
 prior=read(D/'V166_VS_FIXED_COMPLETED_BOTTLENECK_AND_MATCHED65_ANALYSIS.json');p148=read(D/'ROOT_V148_COMPLETED_FULL463_COST_SUMMARY.json')
 comparison=dict(P166_full32=prior['full32'],P148_full64=dict(wall_seconds=p148['clocks']['host_generate_wall_seconds'],raw_tokens_per_second=p148['clocks']['raw_tokens_per_second'],Goodput=p148['clocks']['Native_paper']['Goodput'],attained=p148['clocks']['Native_paper']['attained'],target_calls=p148['geometry_full']['target_rounds'],mean_N=p148['geometry_full']['total_verified_owner_rounds']/p148['geometry_full']['target_rounds'],target_HOST_seconds=p148['phase_seconds_by_rank'][1]['phase_seconds']['target'],draft_HOST_seconds=p148['phase_seconds_by_rank'][0]['phase_seconds']['draft'],standalone_HOST_seconds=p148['standalone_refill_seconds']),fixed_gamma4=prior['fixed_gamma4'])
 need(not Firewall.attempts and not any(n.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend')for n in sys.modules),'no backend imports')
 result=dict(schema='V176.independent-terminal-component-and-full463-audit.v1',status='passed_terminal_component_and_full463_audit',candidate=str(P),tag=TAG,job_status='complete',receipt=desc(rp),script=desc(__file__),errors=[],provenance=dict(freeze=desc(P/'FROZEN_SOURCE176.json'),CPU=desc(P/'CPU_RESULT176.json'),all_selected_current_SHA=len(pins),all_selected_current_bytes_exact=True,archive=dict(**ar,all162_regular_members_exact=True),raw_manifest=dict(**desc(mp),files=len(manifest),bytes_total=sum(x['bytes']for x in manifest.values())),source_receipt_unchanged=True,baseline_prepost='retained10496/15/212 PASS; not rerun',saved_prepost_all4idle=True,owned_remaining=[],worker_exit=desc(A/'worker-exit.json'),old_dependency_archives_not_reexpanded=True),original_unmodified_audit=dict(source=desc(P/'audit176.py'),passed=True,equals_runtime_completed_and_supervisor=True,result=actual),additional_component_arithmetic=dict(raw_cycles=cycles,original_Native_verdict_reexecutions=verdicts,owner_verdicts=owners,actual_component_zero21_subtractions=zeros,actual_full_and_twozero_subtractions=actual['native_zero_subtractions'],hidden_or_KV_tensors_recomputed=False),full463_metrics=metrics,geometry=geometry,eager=eager,actual_draft_policy_delta=p,actual_FastV5_delta=fast,actual_proposal_delta=proposal,HOST_phase_seconds=phases,actual_early_target_HOST_seconds={k:v/1e9 for k,v in t['decode_counters'].items()if k.endswith('_ns')and 'early_target'in k},draft_model_submissions=dc['spec_rhythm_draft_model_calls'],comparison_inputs=[desc(D/'V166_VS_FIXED_COMPLETED_BOTTLENECK_AND_MATCHED65_ANALYSIS.json'),desc(D/'ROOT_V148_COMPLETED_FULL463_COST_SUMMARY.json')],completed_run_comparison=comparison,performance_inferences=[
'Eager is genuinely active:24817 generated,10471 promoted/aggregate consumed,14346 invalidated. Absence of eager is not the explanation for this result.',
'Mean targetN16.43 exceedsP16615.21 but selected lower-depth B3 reduces effective delivered-per-owner-round; target calls2615 exceedP1662544 despite ready-home merging.',
'Target padding remains severe:2004/2615 calls use128/S38,133401/306452 physical queries arepadding.2176 uniformB3 calls lie in the three proposed smaller bucket domains; fixed-list volume savings are102384 queries if those exact calls used64/80/96. This is no latency or trajectory prediction.',
'Selected draft candidates173137 exceedP166154445 and draft model/graph submissions7382 exceed5062; rank0 HOSTdraft131.74s exceeds95.55s. Extra repair/frontier work with57.8% future invalidation is a concrete optimization domain.',
'Rank1 HOSTtarget133.43s is close toP166131.83s; standalone prefill49.95s is close to50.62s. There is no observed isolated target or prefill gain to offset added full cost.',
'Next smaller-bucket experiment needs current-engine same-M allphysical hidden/queryKV/head plus realverdict/commit/next and retained cross-M behavior checks. Do not copy old timing or infer faster kernels from query-volume reduction.'
],limitations=[
'The complete actual job and retained component/protocol/output/clock/source checks passed, but full-workload independent AR equality/formal adoption and user goal are not thereby established.',
'Actual fullhidden/KV finite/matcher gates were checked from retained originaldevice evidence; tensors were not recomputed onHOST.',
'NativeTPOT original charged decode clock, event-wall TPOT and first-callback TTFT are kept separate; missing fields are not filled with0.',
'HOST scopes overlap acrossranks/CPU/device waits. Do not sum draft+target or attribute their changes as additive device costs. Per-call/family device costs were not retained.',
'P166/P148/fixed comparisons share actual workload/token budgets but activewindow, prefixbatching and selected-budget histories differ; no isolated causal speedups follow. Fixedgamma4 linearbonusFalse uses four queries perrow while treeB3 includesroot+3.',
'Aggregate consume conservation assumes source-bound exactchild3, full output completeness/noabort/noprune and all work published; individual ticket joins and perchild cost were not retained.',
'No current173 or other newrun raw/device/IPC was accessed, no suites or baseline verifier rerun, no frozen source changed.'
],HOST_imports=dict(forbidden_attempts=Firewall.attempts),Torch_imported=False,NPU_used=False,formal_Goodput=False,adopted=False)
 with args.output.open('x')as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(**desc(args.output),status=result['status'],wall=metrics['host_generate_wall_seconds'],raw=metrics['raw_tokens_per_second'],Goodput=metrics['Native_paper']['Goodput'],attained=metrics['Native_paper']['attained'])))
if __name__=='__main__':main()
