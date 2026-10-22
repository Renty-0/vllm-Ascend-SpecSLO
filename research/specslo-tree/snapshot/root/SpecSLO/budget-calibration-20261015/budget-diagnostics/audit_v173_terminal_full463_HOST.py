"""Independent completed173 audit; stdlib only, no model/device/test execution."""
import argparse,ast,collections,copy,gzip,hashlib,importlib.abc,json,math,statistics,sys,tarfile,types
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-causal80-real16-bankfix-full-v173';TAG='causal80-bankfix-v173-full463-v1-overlay177';A=P/'actual'/TAG
G=D/'v177-root-safe-storage-overlay'
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
def need(ok,label):
 if not ok:raise ValueError(label)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def desc(p):p=Path(p);return dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size)
def read(p):
 p=Path(p);raw=p.read_bytes()
 if p.suffix=='.gz':raw=gzip.decompress(raw)
 return json.loads(raw,parse_constant=lambda x:(_ for _ in ()).throw(ValueError(x)))
def archive(path,pins):
 with tarfile.open(path,'r:gz')as tar:
  members=tar.getmembers();need(len(members)==len(pins)and len({v.name for v in members})==len(members),'exact unique archive inventory')
  for m in members:need(m.isfile()and '/'+m.name in pins and hashlib.sha256(tar.extractfile(m).read()).hexdigest()==pins['/'+m.name],'regular exact archive member '+m.name)
 return dict(**desc(path),members=len(pins),all_regular_member_bytes_exact=True)
class Firewall(importlib.abc.MetaPathFinder):
 attempts=[]
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors','acl'):
   self.attempts.append(name);raise ImportError('Independent173 HOST '+name)
def extracted(path,name,ns):
 n=copy.deepcopy(next(v for v in ast.walk(ast.parse(Path(path).read_text()))if isinstance(v,ast.FunctionDef)and v.name==name));n.decorator_list=[]
 tree=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),n],type_ignores=[])
 exec(compile(ast.fix_missing_locations(tree),str(path)+'::independent-HOST','exec'),ns);return ns[name]
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
 parser=argparse.ArgumentParser();parser.add_argument('--expected-receipt-sha256',required=True);parser.add_argument('--output',required=True,type=Path);args=parser.parse_args()
 need(not args.output.exists(),'fresh independent output');rp=P/f'receipt-{TAG}.json';need(sha(rp)==args.expected_receipt_sha256,'caller exact terminal receipt')
 r=read(rp);f=read(P/'FROZEN_SOURCE173.json');need(r['status']=='complete'and r['child']['exit_code']==0 and r['errors']==[]and r['sources_unchanged']is True and r['owned_remaining']==[]and r['actual_execution_complete']is True,'complete unchanged owned-clean terminal')
 need(r['readiness']['freeze_sha256']==sha(P/'FROZEN_SOURCE173.json')and r['readiness']['CPU_sha256']==sha(P/'CPU_RESULT173.json'),'actual source/CPU binding')
 pins={}
 for k in('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):pins.update(f[k])
 pins.update({str(P/'CPU_RESULT173.json'):f['CPU_sha256'],str(P/'FROZEN_SOURCE173.json'):sha(P/'FROZEN_SOURCE173.json'),f['root_review']['path']:f['root_review']['sha256']})
 need(all(sha(p)==h for p,h in pins.items()),'current exact selected sources');need(len(pins)==r['readiness']['archive']['members']==158,'actual158selected')
 ar=r['readiness']['archive'];need(sha(ar['path'])==ar['sha256'],'original archive SHA');source_archive=archive(ar['path'],pins)
 ovp=G/f'receipt-overlay-173-{TAG}.json';ov=read(ovp);of=read(G/'FROZEN_OVERLAY177.json');opins=dict(of['inputs']);opins[str(G/'FROZEN_OVERLAY177.json')]=sha(G/'FROZEN_OVERLAY177.json')
 need(ov['exit_code']==0 and ov['errors']==[]and ov['original_status']=='complete'and ov['original_receipt_sha256']==sha(rp)and ov['runtime_or_model_source_changed']is ov['outer_lock_added']is False,'separate storage-only overlay successful and exact original receipt')
 need(all(sha(p)==h for p,h in opins.items()),'overlay exact source closure');oarchive=archive(G/'PREPARED_OVERLAY177.tar.gz',opins)
 need(ov['source_before']==ov['source_after']==dict(freeze_sha256=sha(G/'FROZEN_OVERLAY177.json'),archive_sha256=sha(G/'PREPARED_OVERLAY177.tar.gz'),inputs=len(opins)),'overlay prepost current exact20inputs')
 for k in('pre_verifier','post_verifier'):
  v=r[k];q=read(v['path']);need(sha(v['path'])==v['sha256']and v['exit_code']==0 and q==v['actual_result']and q['status']=='pass'and q['errors']==[]and(q['checked_frozen_files'],q['checked_external_files'],q['checked_packages'])==(10496,15,212),'retained original10496/15/212 '+k)
 for k in('devices_before','devices_after'):
  v=r[k];need(v['all_idle']is v['four_910B2']is True and set(v['processes'])=={'2','5','6','7'},'saved4idle '+k)
  need(all(z['exit_code']==0 and z['idle']is True and z['output'].strip()=='No process in device.'for z in v['processes'].values()),'actual retained idle outputs')
 need(r['execution_lock']['nonblocking_exclusive']is True and r['limits']==dict(child_execution_seconds=7200,actual_tag_bytes=256*1024**2,aggregate_private_bytes=512*1024**2)and r['elapsed_seconds']<7200,'original selflock/actual limits')
 need(read(A/'worker-exit.json')==dict(closed=True,all_exited=True)and not list(A.glob('rank*-last-error.json')),'closed all4/no errors')
 manifest={str(p):dict(bytes=p.stat().st_size,sha256=sha(p))for p in sorted(A.rglob('*'))if p.is_file()};mp=args.output.with_name(args.output.stem+'_RAW_MANIFEST.json');need(not mp.exists(),'fresh raw manifest')
 mp.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
 sys.meta_path.insert(0,Firewall());sys.path.insert(0,str(P))
 import audit173
 actual=audit173.audit(A,(32,));completed=read(A/'completed.json')
 need(actual['passed']is True and actual==completed['audit']==r['actual_full_audit'],'original unmodified173 actual HOST auditor equals runtime and supervisor')
 need(completed['action']=='causal80-active32-full463'and completed['replies']==[['calibrated',i,dict(windows=[32],full463=True)]for i in range(4)],'actual4success replies')
 reports=[read(A/f'rank{i}-local-diagnostic.json')for i in range(4)];traces=[read(A/f'windows/window32/full463/rank{i}-full463-actual-trace.json.gz')for i in range(4)];t=traces[1];draft=traces[0]
 need(all(x['sealed_before']is x['sealed_after']is x['cache_released']is True for x in traces),'all4 sealed released full')
 ns=dict(math=math,statistics=statistics,CLASSES=('tight','normal','loose'));cp=D/'analyze_completed_tree_full463_cost.py';extracted(cp,'need',ns);metrics=extracted(cp,'clocks',ns)(t)
 first={}
 for event in t['stream_events']:
  i=event['request_index']
  if i not in first:need('first_token_wall_seconds160'in event,'real first callback');first[i]=event['first_token_wall_seconds160']
  else:need('first_token_wall_seconds160'not in event,'single first callback')
 need(set(first)==set(range(463))and all(row['arrival_ts']==t['absolute_arrivals'][i]for i,row in enumerate(t['generate_results'])),'all463 original actual arrivals and callbacks')
 ttft=[(first[i]-row['arrival_ts'])*1000 for i,row in enumerate(t['generate_results'])];need(all(math.isfinite(v)and v>=0 for v in ttft),'actual finite TTFT')
 retained=reports[1]['windows']['per_rank_full_metrics'][0]['output'];need(all(retained[k]==v for k,v in metrics.items()),'independent original clock reconstruction equals retained full metrics')
 tt=dict(samples=463,**pct(ttft),by_class={cl:pct([ttft[i]for i,row in enumerate(t['generate_results'])if row['slo_class']==cl])for cl in('tight','normal','loose')})
 need(retained['TTFT160']['median_ms']==tt['median_ms']and retained['TTFT160']['P90_ms']==tt['P90_ms'],'retained actual TTFT exact');metrics['TTFT_independent']=tt
 # Original Native verifier, independently reexecuted on every retained new80
 # target result. Value is an explicit HOST list carrier, not a tensor oracle.
 ht=types.SimpleNamespace(Tensor=Value,tensor=lambda v,**kw:Value(v),long='long');vns=dict(torch=ht,TreeVerificationOutput=lambda a,b:types.SimpleNamespace(token_ids=a,accepted_node_indices=b));verify=extracted(NATIVE/'native_engine.py','_verify_tree_outputs_host',vns)
 comparisons=[];verdicts=owner_verdicts=0
 for rank in(1,2,3):
  for p in range(3):
   for c in range(4):
    q=read(A/f'causal80-own-current-process/rank{rank}-pass{p}-cycle{c}.json.gz');cm=q['crossM'];plans=[types.SimpleNamespace(candidate_budget=4,parent_indices=Value([-1,0,1,2]))for _ in range(16)]
    for which in('old','new'):
     output=verify(q['rows'],plans,Value(cm[which+'_real80']),4,'explicit-HOST-values');tokens=[[v for v in row if v>=0]for row in output.token_ids.values]
     need(output.accepted_node_indices.values==cm[which+'_accepted']and tokens==[[v for v in row if v>=0]for row in cm[which+'_verdict_tokens']],'actual originalNative host verifier all80 '+which);verdicts+=1;owner_verdicts+=16
    positions=[i for i,(a,b)in enumerate(zip(cm['old_real80'],cm['new_real80'],strict=True))if a!=b]
    comparisons.append(dict(rank=rank,pass_index=p,cycle=c,different_head_positions=positions,old_real80=cm['old_real80'],new_real80=cm['new_real80'],verdict_equal=cm['verdict_equal'],accepted_equal=cm['accepted_equal']))
 need(verdicts==72 and owner_verdicts==1152,'all36 actual same-source old/new verifier comparisons')
 calls=t['normal_target_calls'];N=sum(c['N']for c in calls);nodes=sum(sum(c['budgets'])for c in calls);physical=sum(c['physical_M']for c in calls);real=sum(c['real_query_count']for c in calls);dc=draft['decode_counters'];env=reports[0]['windows']['traces'][-1];work=env['actual_split_work_counts']['delta'];pol=work['policy'];fast=work['FastV5'];proposal=env['actual_proposal_counts']['delta']
 need(nodes==t['decode_counters']['spec_rhythm_verified_tokens']and physical==t['decode_counters']['spec_rhythm_target_query_tokens']and physical-real==t['decode_counters']['spec_rhythm_target_graph_padding_queries'],'actual geometry counter arithmetic')
 selected80=[c for c in calls if(c['N'],c['real_query_count'],c['family'])==(16,80,'causal')];need(all((c['physical_M'],c['physical_S'],c['physical_query_partition'])==(80,16,[b+1 for b in c['budgets']])for c in selected80),'all actual current80 dispatches')
 geometry=dict(target_calls=len(calls),verified_owner_rounds=N,verified_candidate_nodes=nodes,mean_N=N/len(calls),physical_Q=physical,real_Q=real,padding_Q=physical-real,padding_fraction=(physical-real)/physical,new80_calls=len(selected80),new80_fraction=len(selected80)/len(calls),new80_uniform_B4_calls=sum(c['budgets']==[4]*16 for c in selected80),new80_physical_Q=80*len(selected80),N_hist=dict(sorted(collections.Counter(c['N']for c in calls).items())),actual_shapes=[dict(M=m,S=s,family=f,calls=count)for(m,s,f),count in sorted(collections.Counter((c['physical_M'],c['physical_S'],c['family'])for c in calls).items())],budget_owner_rounds={str(b):proposal['actual_target_B'+str(b)]for b in range(1,6)},standalone_calls=len(t['standalone_prefill_calls']),standalone_prompt_Q=sum(c['uncached_queries']for c in t['standalone_prefill_calls']),standalone_HOST_seconds=t['decode_counters']['spec_rhythm_online_refill_ms']/1000,mixed_calls=len(t['logical_events']))
 need(geometry['standalone_prompt_Q']==203850 and geometry['mixed_calls']==0,'whole prompts remain standalone')
 eager=dict(normal_rows=pol['actual_draft_normal_rows'],normal_nodes=pol['split_draft_final_normal_candidates'],generated_rows=pol['actual_draft_eager_rows'],selected_future_nodes=pol['split_draft_final_eager_candidates'],promoted=dc['spec_rhythm_tree_eager_promoted'],invalidated=dc['spec_rhythm_tree_eager_invalidated'],selected_draft_nodes=dc['spec_rhythm_allocated_draft_tokens'],model_submissions=dc['spec_rhythm_draft_model_calls'],target_rows_minus_normal=N-pol['actual_draft_normal_rows'],target_nodes_minus_normal=nodes-pol['split_draft_final_normal_candidates'])
 need(eager['generated_rows']==eager['selected_future_nodes']==eager['promoted']==eager['invalidated']==0 and eager['selected_draft_nodes']==eager['normal_nodes']==nodes and eager['normal_rows']==N,'actual E0 and normal publication conservation')
 geometry['effective_outputs_excluding_initial_per_owner_round']=sum(len(row['completion_token_ids'])-1 for row in t['generate_results'])/N
 old=read(D/'V176_INDEPENDENT_TERMINAL_FULL463_AUDIT.json');prior=read(D/'V166_VS_FIXED_COMPLETED_BOTTLENECK_AND_MATCHED65_ANALYSIS.json')
 comparison=dict(P176=dict(full463_metrics=old['full463_metrics'],geometry=old['geometry'],eager=old['eager'],HOST_phase_seconds=old['HOST_phase_seconds'],draft_model_submissions=old['draft_model_submissions']),P166=prior['full32'],fixed_gamma4=prior['fixed_gamma4'])
 need(not Firewall.attempts and not any(n.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend')for n in sys.modules),'no backend imports')
 result=dict(schema='V173.independent-terminal-component-and-full463-audit.v1',status='passed_terminal_component_and_full463_audit',candidate=str(P),tag=TAG,job_status='complete',receipt=desc(rp),script=desc(__file__),errors=[],provenance=dict(freeze=desc(P/'FROZEN_SOURCE173.json'),CPU=desc(P/'CPU_RESULT173.json'),all_selected_current_SHA=len(pins),all_selected_current_bytes_exact=True,archive=source_archive,overlay=dict(receipt=desc(ovp),freeze=desc(G/'FROZEN_OVERLAY177.json'),archive=oarchive,inputs=len(opins),source_before_equals_after=True,storage_only_no_model_or_outerlock_change=True),raw_manifest=dict(**desc(mp),files=len(manifest),bytes_total=sum(v['bytes']for v in manifest.values())),baseline_prepost='retained10496/15/212 PASS, not rerun',saved_all4idle=True,owned_remaining=[],worker_exit=desc(A/'worker-exit.json'),source_unchanged=True),original_unmodified_audit=dict(source=desc(P/'audit173.py'),passed=True,equals_runtime_completed_and_supervisor=True,result=actual),additional_original_verifier=dict(source=desc(NATIVE/'native_engine.py'),actual_old_new_verifier_calls=verdicts,owner_verdicts=owner_verdicts,crossM_all36_target_rank_records=comparisons,all_real80_heads_equal=all(not v['different_head_positions']for v in comparisons),all_verdicts_equal=all(v['verdict_equal']and v['accepted_equal']for v in comparisons),hidden_KV_recomputed_on_HOST=False),full463_metrics=metrics,geometry=geometry,eager=eager,actual_draft_policy_delta=pol,actual_FastV5_delta=fast,actual_proposal_delta=proposal,HOST_phase_seconds={str(i):v['phase_seconds']for i,v in enumerate(traces)},actual_early_target_HOST_seconds={k:v/1e9 for k,v in t['decode_counters'].items()if k.endswith('_ns')and 'early_target'in k},completed_run_comparison=comparison,comparison_inputs=[desc(D/'V176_INDEPENDENT_TERMINAL_FULL463_AUDIT.json'),desc(D/'V166_VS_FIXED_COMPLETED_BOTTLENECK_AND_MATCHED65_ANALYSIS.json')],limitations=['Actual current same-M allphysical hidden/head/all64queryKV and original verdict/compaction/next/twozero gates passed; HOST recomputes retained integer arithmetic and source evidence, not model tensors.','Cross-M all80 heads/verdict equality is proven for12 bounded same-source component comparisons across all3 target ranks, not arbitrary prefixes or full-workload independent AR equality.','Full463 original output/stream/source/arrival/clock/TTFT and all21counter gates passed; this does not authorize production adoption or declare the optimization goal fulfilled.','Full GP/raw/wall and aggregate HOST phases are whole-run observations. Compared P166/P176 have differing prefix/budget/protocol trajectories; phases include CPU/device waits and overlap and cannot be added as isolated device costs.','No paired local target cost measured; query-volume reduction alone does not establish latency causality. No current180 raw/device/IPC access, suites/global verifier rerun or frozen-source edit.'],Torch_imported=False,NPU_used=False,formal_Goodput=False,adopted=False)
 with args.output.open('x')as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(**desc(args.output),status=result['status'],wall=metrics['host_generate_wall_seconds'],raw=metrics['raw_tokens_per_second'],Goodput=metrics['Native_paper']['Goodput'],attained=metrics['Native_paper']['attained'],geometry=geometry,eager=eager)))
if __name__=='__main__':main()
