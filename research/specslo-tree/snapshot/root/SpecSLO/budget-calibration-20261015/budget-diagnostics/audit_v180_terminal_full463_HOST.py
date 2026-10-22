"""Independent terminal V180 source/raw audit; no model, device or old suite."""
import argparse,collections,copy,gzip,hashlib,importlib.util,json,math,statistics,sys,tarfile,types
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-uniform-b3-three-bucket-full-v180';TAG='uniform-b3-three-bucket-v180-full463-v1';A=P/'actual'/TAG
HELPER=D/'audit_v176_terminal_full463_HOST.py'
spec=importlib.util.spec_from_file_location('_completed176_HOST_utilities_only',HELPER)
h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
need,sha,desc,read,extracted,Value,pct=h.need,h.sha,h.desc,h.read,h.extracted,h.Value,h.pct
NATIVE=h.NATIVE

def original_verifier():
 parents=extracted(NATIVE/'tree.py','_spine_first_topology',{})(2,4)[0]
 node=copy.deepcopy(next(n for n in h.ast.parse((NATIVE/'tree.py').read_text()).body if isinstance(n,h.ast.FunctionDef)and n.name=='pack_selected_tree_plan'));node.decorator_list=[]
 stop=next(i for i,n in enumerate(node.body)if isinstance(n,h.ast.Assign)and any(isinstance(z,h.ast.Name)and z.id=='device'for z in n.targets));node.body=node.body[:stop]+[h.ast.Return(value=h.ast.Name(id='new_parents',ctx=h.ast.Load()))]
 ht=types.SimpleNamespace(Tensor=Value,tensor=lambda x,**kw:Value(x),long='long');pns=dict(torch=ht)
 exec(compile(h.ast.fix_missing_locations(h.ast.Module(body=[h.ast.ImportFrom(module='__future__',names=[h.ast.alias(name='annotations')],level=0),node],type_ignores=[])),'original-packing-prefix-only-V180','exec'),pns)
 vns=dict(torch=ht,TreeVerificationOutput=lambda a,b:types.SimpleNamespace(token_ids=a,accepted_node_indices=b))
 verify=extracted(NATIVE/'native_engine.py','_verify_tree_outputs_host',vns)
 def run(row,heads):
  plans=[types.SimpleNamespace(candidate_budget=len(ids),parent_indices=Value(pns['pack_selected_tree_plan'](types.SimpleNamespace(parent_indices=Value(parents),candidate_budget=8),ids)))for ids in row['selected']]
  q=sum(len(ids)+1 for ids in row['selected']);out=verify(row['rows'],plans,Value(heads[:q]),4,'explicit-HOST-values')
  need([[v for v in line if v>=0]for line in out.token_ids.values]==row['commit']['committed_tokens']and out.accepted_node_indices.values==row['commit']['accepted_paths'],'actual original Native ancestry/verdict from retained physical heads')
  return len(plans)
 return run

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--expected-receipt-sha256',required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
 need(sys.version_info[:2]==(3,11),'retained Py311');need(not args.output.exists(),'fresh output')
 rp=P/f'receipt-{TAG}.json';need(sha(rp)==args.expected_receipt_sha256,'caller terminal receipt SHA')
 r=read(rp);f=read(P/'FROZEN_SOURCE180.json')
 need(r['status']=='complete'and r['child']['exit_code']==0 and r['errors']==[]and r['sources_unchanged']is True and r['owned_remaining']==[],'successful unchanged clean terminal')
 need(r['readiness']['freeze_sha256']==sha(P/'FROZEN_SOURCE180.json')=='c07267efc475ede2d0983bf6ffac37c3c6d10f8c8c8648e1b4b018f34b4702e3'and r['readiness']['CPU_sha256']==sha(P/'CPU_RESULT180.json'),'freeze/CPU binding')
 pins={}
 for k in('files','external_files','retained_HOST_evidence','execution_guard_inputs','peer_reviews'):pins.update(f[k])
 pins.update({str(P/'CPU_RESULT180.json'):f['CPU_sha256'],str(P/'FROZEN_SOURCE180.json'):sha(P/'FROZEN_SOURCE180.json'),f['root_review']['path']:f['root_review']['sha256']})
 need(all(sha(p)==v for p,v in pins.items()),'all current selected source bytes')
 ar=r['readiness']['archive'];need(sha(ar['path'])==ar['sha256']=='3d0b2117530b14dbf8418a2b0cdc842e2de586622406a5f46f63576b2d1900a0','prepared source archive SHA')
 with tarfile.open(ar['path'],'r:gz')as tar:
  members=tar.getmembers();need(len(members)==len(pins)==ar['members']==246 and len({m.name for m in members})==246,'exact246 source members')
  for m in members:need(m.isfile()and '/'+m.name in pins and hashlib.sha256(tar.extractfile(m).read()).hexdigest()==pins['/'+m.name],'every regular archived member exact')
 for k in('pre_verifier','post_verifier'):
  v=r[k];q=read(v['path']);need(sha(v['path'])==v['sha256']and v['exit_code']==0 and q==v['actual_result']and q['status']=='pass'and q['errors']==[]and(q['checked_frozen_files'],q['checked_external_files'],q['checked_packages'])==(10496,15,212),'retained original pre/post verifier')
 for k in('devices_before','devices_after'):
  v=r[k];need(v['all_idle']is v['four_910B2']is True and set(v['processes'])=={'2','5','6','7'},'saved four idle cards')
  need(all(x['exit_code']==0 and x['idle']is True and x['output'].strip()=='No process in device.'for x in v['processes'].values()),'retained original idle command output')
 need(read(A/'worker-exit.json')==dict(closed=True,all_exited=True)and not list(A.glob('rank*-last-error.json')),'worker closure/no failure file')
 raw_manifest={str(p):dict(bytes=p.stat().st_size,sha256=sha(p))for p in sorted(A.rglob('*'))if p.is_file()}
 mp=args.output.with_name(args.output.stem+'_RAW_MANIFEST.json');need(not mp.exists(),'fresh raw manifest');mp.write_text(json.dumps(raw_manifest,indent=2,sort_keys=True)+'\n')
 sys.meta_path.insert(0,h.Firewall());sys.path.insert(0,str(P))
 import audit180
 actual=audit180.audit(A,(32,));completed=read(A/'completed.json')
 need(actual['passed']is True and completed['audit']==actual==r['actual_full_audit'],'unmodified audit180 HOST equals runtime completed and final guard')
 need(completed['action']=='uniform-b3-three-bucket-full463'and completed['replies']==[['calibrated',i,dict(windows=[32],full463=True)]for i in range(4)],'all4 action/replies')
 reports=[read(A/f'rank{i}-local-diagnostic.json')for i in range(4)]
 traces=[read(A/f'windows/window32/full463/rank{i}-full463-actual-trace.json.gz')for i in range(4)];t=traces[1];draft=traces[0]
 need(all(x['sealed_before']is x['sealed_after']is x['cache_released']is True for x in traces),'sealed released actual full')
 verify=original_verifier();base_counts=dict(raw_cycles=0,original_verifier_calls=0,owner_verdicts=0,zero_scalars=0)
 folder=A/'variable-B4-own-full-process'
 for rank in range(4):
  report=read(folder/f'rank{rank}-component.json.gz')
  for tr in report['traces']:
   if tr['pass_index']:base_counts['zero_scalars']+=len(tr['Native']['delta']);need(not any(tr['Native']['delta'].values()),'B4 complete actual21zero')
   for cycle in range(4):
    row=read(folder/f"rank{rank}-pass{tr['pass_index']}-case{tr['case']}-cycle{cycle}.json.gz");base_counts['raw_cycles']+=1
    need(row['commit']['actual_logical_tokens']==[old['token_ids']+new for old,new in zip(row['source']['contexts'],row['commit']['committed_tokens'],strict=True)],'B4 genuine logical history')
    if rank:
     base_counts['owner_verdicts']+=verify(row,row['target']['greedy'][2]['full_physical_tokens']);base_counts['original_verifier_calls']+=1
 need(base_counts==dict(raw_cycles=96,original_verifier_calls=72,owner_verdicts=2304,zero_scalars=336),'base actual domain')
 G=A/'uniformB3-three-bucket-own-current-process';extrema=(9,16,17,20,21,24);own_counts=dict(protocol_cycles_per_rank=0,original_verifier_calls=0,owner_verdicts=0,crossM_different_heads=0,crossM_changed_owner_verdicts=0,zero_scalars=0,next_files=0)
 cost=[]
 for n in range(9,25):
  for p in range(3)if n in extrema else range(1):
   cycles=4 if n in extrema else 3
   for cycle in range(cycles):
    rows=[read(G/f'rank{rank}-N{n}-pass{p}-cycle{cycle}.json.gz')for rank in range(4)]
    heads=[row['target']['greedy'][2]['full_physical_tokens']for row in rows[1:]]
    need(all(v==heads[0]for v in heads),'actual target TP replicas full physical heads')
    for rank in range(1,4):own_counts['owner_verdicts']+=verify(rows[rank],heads[rank-1]);own_counts['original_verifier_calls']+=1
    row=rows[1];own_counts['protocol_cycles_per_rank']+=1;own_counts['crossM_different_heads']+=len(row['crossM']['different_head_positions'])
    own_counts['crossM_changed_owner_verdicts']+=sum(a!=b for a,b in zip(row['crossM']['old_verdict_tokens'],row['crossM']['new_verdict_tokens'],strict=True))
    if row['local_cost']is not None:
     for rank in range(1,4):
      value=rows[rank]['local_cost'];cost.append(dict(N=n,rank=rank,median_nanoseconds=value['median_nanoseconds'],new_over_old=value['new_over_old'],scope=value['scope'],samples=value['samples']))
   for rank in range(4):
    nxt=read(G/f'rank{rank}-N{n}-pass{p}-next.json.gz');need(nxt['source']['contexts'],'actual next history');own_counts['next_files']+=1
 for rank in range(4):
  cr=read(G/f'rank{rank}-component.json.gz')
  for tr in cr['traces']:
   if tr['pass_index']:need(not any(tr['Native']['delta'].values()),'new component both complete zero');own_counts['zero_scalars']+=len(tr['Native']['delta'])
 need(own_counts['protocol_cycles_per_rank']==102 and own_counts['original_verifier_calls']==306 and own_counts['zero_scalars']==1008 and own_counts['next_files']==112,'new actual102cycles/28next/all4zero domain')
 cp=D/'analyze_completed_tree_full463_cost.py';ns=dict(math=math,statistics=statistics,CLASSES=('tight','normal','loose'));extracted(cp,'need',ns);metrics=extracted(cp,'clocks',ns)(t)
 first={}
 for event in t['stream_events']:
  i=event['request_index']
  if i not in first:need('first_token_wall_seconds160'in event,'actual first wall');first[i]=event['first_token_wall_seconds160']
  else:need('first_token_wall_seconds160'not in event,'one first callback')
 need(set(first)==set(range(463)),'all463 first callbacks')
 need(all(row['arrival_ts']==t['absolute_arrivals'][i]for i,row in enumerate(t['generate_results'])),'original rebased arrivals')
 ttft=[(first[i]-row['arrival_ts'])*1000 for i,row in enumerate(t['generate_results'])];need(all(math.isfinite(v)and v>=0 for v in ttft),'finite actual TTFT')
 metrics['TTFT_independent']=dict(samples=463,**pct(ttft),by_class={cl:pct([ttft[i]for i,row in enumerate(t['generate_results'])if row['slo_class']==cl])for cl in('tight','normal','loose')})
 retained=reports[1]['windows']['per_rank_full_metrics'][0]['output'];need(all(retained[k]==v for k,v in metrics.items()if k!='TTFT_independent'),'independent clock reconstruction')
 need(retained['TTFT160']['median_ms']==metrics['TTFT_independent']['median_ms']and retained['TTFT160']['P90_ms']==metrics['TTFT_independent']['P90_ms'],'independent callback TTFT')
 calls=t['normal_target_calls'];N=sum(c['N']for c in calls);C=sum(sum(c['budgets'])for c in calls);physical=sum(c['physical_M']for c in calls);real=sum(c['real_query_count']for c in calls)
 dc=draft['decode_counters'];need(C==t['decode_counters']['spec_rhythm_verified_tokens']and physical==t['decode_counters']['spec_rhythm_target_query_tokens']and physical-real==t['decode_counters']['spec_rhythm_target_graph_padding_queries'],'actual call geometry/counters')
 env=reports[0]['windows']['traces'][-1];work=env['actual_split_work_counts']['delta'];policy=work['policy'];fast=work['FastV5'];proposal=env['actual_proposal_counts']['delta']
 eager=dict(generated_rows=policy['actual_draft_eager_rows'],promoted_rows=dc['spec_rhythm_tree_eager_promoted'],invalidated_rows=dc['spec_rhythm_tree_eager_invalidated'],selected_future_nodes=policy['split_draft_final_eager_candidates'],normal_rows=policy['actual_draft_normal_rows'],normal_nodes=policy['split_draft_final_normal_candidates'],draft_work_rows=policy['actual_draft_work_rows'],selected_total_draft_nodes=dc['spec_rhythm_allocated_draft_tokens'],aggregate_consumed_rows=N-policy['actual_draft_normal_rows'],aggregate_consumed_nodes=C-policy['split_draft_final_normal_candidates'])
 need(eager['generated_rows']==eager['promoted_rows']+eager['invalidated_rows']==fast['eager_frontier_real_queries']and eager['selected_future_nodes']==3*eager['generated_rows'],'actual E generation conservation')
 need(eager['aggregate_consumed_rows']==eager['promoted_rows']and eager['aggregate_consumed_nodes']==3*eager['promoted_rows'],'source-bound aggregate E consume')
 need(eager['selected_total_draft_nodes']==eager['normal_nodes']+eager['selected_future_nodes']==dc['spec_rhythm_tree_nodes']and N==sum(proposal['actual_target_B'+str(b)]for b in range(1,6)),'published work conservation')
 eager['promoted_fraction']=eager['promoted_rows']/eager['generated_rows']
 need(dc['spec_rhythm_draft_model_calls']==sum(fast[k]for k in('chain_calls','repair_calls','eager_frontier_calls','root_seed_calls')),'real draft submission categories sum')
 from geometry180 import extent
 compact=[c for c in calls if extent(c['N'])is not None and c['family']=='causal'and c['budgets']==[3]*c['N']]
 need(all((c['physical_M'],c['physical_S'])==extent(c['N'])for c in compact),'production actually uses current compact keys')
 geometry=dict(target_calls=len(calls),mean_N=N/len(calls),verified_owner_rounds=N,verified_candidate_nodes=C,physical_Q=physical,real_Q=real,padding_Q=physical-real,padding_fraction=(physical-real)/physical,effective_outputs_excluding_initial_per_owner_round=(118528-463)/N,compact_calls=len(compact),compact_fraction=len(compact)/len(calls),N_hist=dict(sorted(collections.Counter(c['N']for c in calls).items())),budget_owner_rounds={str(b):proposal['actual_target_B'+str(b)]for b in range(1,6)},actual_shapes=[dict(M=m,S=s,family=family,calls=count)for(m,s,family),count in sorted(collections.Counter((c['physical_M'],c['physical_S'],c['family'])for c in calls).items())],standalone_calls=len(t['standalone_prefill_calls']),standalone_prompt_Q=sum(c['uncached_queries']for c in t['standalone_prefill_calls']),standalone_HOST_seconds=t['decode_counters']['spec_rhythm_online_refill_ms']/1000)
 need(geometry['standalone_prompt_Q']==203850 and len(t['logical_events'])==0,'all original prompts genuine standalone')
 previous={k:read(D/f'V{k}_INDEPENDENT_TERMINAL_FULL463_AUDIT.json')for k in(176,173)}
 need(all(v['status']=='passed_terminal_component_and_full463_audit'for v in previous.values()),'prior completed independent audits')
 need(not h.Firewall.attempts and not any(n.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend')for n in sys.modules),'no backend import')
 result=dict(schema='V180.independent-terminal-component-and-full463-audit.v1',status='passed_terminal_component_and_full463_audit',candidate=str(P),tag=TAG,job_status='complete',receipt=desc(rp),script=desc(__file__),HOST_utilities_source=desc(HELPER),errors=[],provenance=dict(freeze=desc(P/'FROZEN_SOURCE180.json'),CPU=desc(P/'CPU_RESULT180.json'),all_selected_current_SHA=len(pins),all_selected_current_bytes_exact=True,archive=dict(**ar,all246_regular_members_exact=True),raw_manifest=dict(**desc(mp),files=len(raw_manifest),bytes_total=sum(v['bytes']for v in raw_manifest.values())),source_receipt_unchanged=True,baseline_prepost='retained10496/15/212 PASS; not rerun',saved_prepost_all4idle=True,owned_remaining=[],worker_exit=desc(A/'worker-exit.json')),original_unmodified_audit=dict(source=desc(P/'audit180.py'),passed=True,equals_runtime_completed_and_supervisor=True,result=actual),additional_component_arithmetic=dict(original_B4=base_counts,compact_uniformB3=own_counts,actual_full_and_twozero_subtractions=actual['native_zero_subtractions'],logicalNs=list(range(9,25)),six_extrema_three_fresh_passes=list(extrema),interior_ten_one_fresh_history_three_successive_commits=True,raw_math_files=524,hidden_or_KV_tensors_recomputed=False),full463_metrics=metrics,geometry=geometry,eager=eager,actual_draft_policy_delta=policy,actual_FastV5_delta=fast,actual_proposal_delta=proposal,HOST_phase_seconds={str(i):x['phase_seconds']for i,x in enumerate(traces)},actual_early_target_HOST_seconds={k:v/1e9 for k,v in t['decode_counters'].items()if k.endswith('_ns')and'early_target'in k},draft_model_submissions=dc['spec_rhythm_draft_model_calls'],paired_local_target_cost=cost,comparisons={str(k):dict(metrics=v['full463_metrics'],geometry=v['geometry'],eager=v['eager'],HOST_phase_seconds=v['HOST_phase_seconds'],draft_model_submissions=v.get('draft_model_submissions'),FastV5=v['actual_FastV5_delta'],early_target=v['actual_early_target_HOST_seconds'],input=desc(D/f'V{k}_INDEPENDENT_TERMINAL_FULL463_AUDIT.json'))for k,v in previous.items()},limitations=['Actual current numerical matcher gates and complete physical hidden/128 target KV arrays were checked on the device during the component; saved markers/source/raw head and metadata are audited here. Full tensor values are not retained and cannot be recomputed on HOST.','The geometry component has six extrema with three fresh repeated histories and ten interior Ns with one fresh history of three successive genuine commits;histories_per_N=3 in the inherited summary is not a claim of three separately initialized interior histories.','Native charged TPOT,event-wall TPOT and callback TTFT remain separate. HOST phases include overlapping CPU/device work and synchronization;their changes are not additive device speedups.','Old/new local target costs are fixed-prefix whole-call micro timings,not full pipeline or Goodput predictions. Whole workloads differ in selected candidates,prefix histories and overlap feedback.','Eager consumed counts are source-bound aggregate conservation,not independently retained per-parent-ticket joins.','No NPU/device command,model,old suite or baseline verifier was rerun. No full-workload AR equality/adoption is inferred.'],HOST_imports=dict(forbidden_attempts=h.Firewall.attempts),Torch_imported=False,NPU_used=False,formal_Goodput=False,adopted=False)
 with args.output.open('x')as stream:json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
 print(json.dumps(dict(**desc(args.output),status=result['status'],wall=metrics['host_generate_wall_seconds'],raw=metrics['raw_tokens_per_second'],Goodput=metrics['Native_paper']['Goodput'],attained=metrics['Native_paper']['attained'])))

if __name__=='__main__':main()
