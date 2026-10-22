"""Terminal-only independent HOST V117 evidence audit; no Torch/model/NPU import."""
import argparse
import copy
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import traceback
from analyze_v110_actual_cpu_context import archive_matches,shell_rows

B=Path('/root/SpecSLO/budget-calibration-20261015');P=B/'tree-linear-mixed-causal-v117'
D=B/'budget-diagnostics'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(v,m):
 if not v:raise RuntimeError(m)
def artifact(p):return dict(path=str(p),sha256=sha(p),bytes=Path(p).stat().st_size)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(gzip.decompress(Path(p).read_bytes()) if str(p).endswith('.gz') else Path(p).read_bytes())

def expected(case,source,reverse=False,changed=False):
 n=case.N;segments=[]
 for i,b in enumerate(case.budgets):
  t=source['trees'][i]['token_ids'];p=min(127+i%4,len(t)-6)
  parents=[-1,0,1,2,-1] if case.branch and i==0 else [-1,*range(b-1)]
  logical=[p]
  for parent in parents:logical.append(p+1 if parent==-1 else logical[parent+1]+1)
  segments.append(dict(owner=i,tokens=t[p:p+b+1],logical=logical,physical=list(range(p,p+b+1)),parents=parents))
 prompt=source['prompt']['token_ids'];start=case.cursor;end=start+case.P
 segments.append(dict(owner=n,tokens=prompt[start:end],logical=list(range(start,end)),physical=list(range(start,end))))
 for j,count in enumerate(case.scratch):segments.append(dict(owner=n+j+1,tokens=[0]*count,logical=list(range(count)),physical=list(range(count))))
 if reverse:segments=list(reversed(segments[:n]))+segments[n:]
 ids=[x for s in segments for x in s['tokens']]
 if changed:ids[0]=(ids[0]+1)%151936
 return dict(ids=ids,positions=[x for s in segments for x in s['logical']],physical=[x for s in segments for x in s['physical']],
  owners=[s['owner'] for s in segments],partition=[len(s['tokens']) for s in segments],segments=segments)

def verify_actual_input(row,case,mode,source,bank,reverse=False,changed=False):
 value=row['model_input'];e=expected(case,source,reverse,changed)
 need(value['ids']==e['ids'] and value['positions']==e['positions'] and value['sequence_ids']==e['owners'],'actual input/RoPE/owner bytes differ')
 need(value['actual_seq_lengths_q']==[sum(e['partition'][:i+1]) for i in range(case.S)]
  and value['sequence_lens']==[max(s['physical'])+1 for s in e['segments']],'actual query/KV literal partition differs')
 lengths=case.partition;chunks=[];cursor=0
 for count in lengths:chunks.append(bank['query_slots'][cursor:cursor+count]);cursor+=count
 if reverse:chunks=list(reversed(chunks[:case.N]))+chunks[case.N:]
 need(value['slot_mapping']==[x for part in chunks for x in part],'actual slot order differs from actual bank query slots')
 need(len(set(value['slot_mapping']))==197 and all(type(x) is int and 0<=x<544*128 and x%128==pos%128 for x,pos in zip(value['slot_mapping'],e['physical'],strict=True)),'physical slots invalid/aliased')
 if mode=='FULL197':
  need(value['tree_attention'] is value['use_fused_infer_attention'] is True and value['tree_mask_shape']==[case.S,1,128,4096] and value['slot_dtype']=='torch.int32','actual FULL ABI differs')
 else:
  need(value['tree_attention'] is False and value['tree_mask_shape'] is None and value['use_fused_infer_attention'] is True
   and value['causal_mask_shape']==[2048,2048] and value['slot_dtype']=='torch.int64','actual causal ABI differs')
 need(row['actual_production_head_M']==case.Q+int(case.final) and len(row['normal_query_token_ids'])==case.Q
  and len(row['final_prompt_token_ids'])==int(case.final) and row['intermediate_prompt_or_dummy_tokens_published'] is False,'real production selected-head extent')
 # Reconstruct the original host verifier independently from the actual selected tokens.
 q=row['normal_query_token_ids'];depth=max(case.budgets) if not case.branch else 4
 emitted=[];accepted=[];offset=0
 for seg in e['segments'][:case.N]:
  b=len(seg['parents']);targets=q[offset:offset+b+1];offset+=b+1
  out=[-1]*(depth+1);path=[-1]*depth;parent=-1;prediction_index=0;active=True
  for step in range(depth):
   if not active or prediction_index>=len(targets):break
   prediction=targets[prediction_index];out[step]=prediction
   chosen=next((i for i,(p,t) in enumerate(zip(seg['parents'],seg['tokens'][1:],strict=True)) if p==parent and t==prediction),-1)
   if chosen<0:active=False;break
   path[step]=chosen;parent=chosen;prediction_index=chosen+1
  if active:out[depth]=targets[min(prediction_index,len(targets)-1)]
  emitted.append(out);accepted.append(path)
 need(emitted==row['actual_verdict_token_ids'] and accepted==row['actual_verdict_accepted_node_indices'],'independent original verifier reconstruction differs')
 if 'auxiliary_all197_physical_token_ids' in row:
  need(len(row['auxiliary_all197_physical_token_ids'])==197 and row['auxiliary_head_M']==197
   and row['auxiliary_all197_is_production_head'] is False,'auxiliary physical tokens mislabeled as production')


def tensor_gate(record,count,shape=None):
 need(record['tensor_count']==count and len(record['rows'])==count and record['actual_device_comparison_executed'] is True
  and record['full_vectors_retained'] is False,'actual gate coverage/retention boundary')
 for i,row in enumerate(record['rows']):
  need(row['index']==i and row['dtype']=='torch.bfloat16' and all(type(v) is int and v>0 for v in row['shape'])
   and (shape is None or row['shape']==shape),'actual gate tensor shape/dtype')
  need(all(type(row[k]) in (int,float) and math.isfinite(row[k]) and row[k]>=0 for k in ('max_abs','mean_abs'))
   and row['max_abs']>=row['mean_abs'] and type(row['bit_equal']) is bool,'actual error statistics')


def cross_summary(row,case,source):
 left,right=(row['real_outputs'][m] for m in ('FULL197','causal197'))
 queries=[];cursor=0
 for owner,b in enumerate(case.budgets):
  for q in range(b+1):
   if left['normal_query_token_ids'][cursor+q]!=right['normal_query_token_ids'][cursor+q]:
    queries.append(dict(owner=owner,query_in_owner=q,packed_query=cursor+q,FULL197=left['normal_query_token_ids'][cursor+q],causal197=right['normal_query_token_ids'][cursor+q]))
  cursor+=b+1
 verdict=[]
 for owner,(a,b) in enumerate(zip(left['actual_verdict_token_ids'],right['actual_verdict_token_ids'],strict=True)):
  for step,(x,y) in enumerate(zip(a,b,strict=True)):
   if x!=y:verdict.append(dict(owner=owner,request_id=source['trees'][owner]['request_id'],emitted_index=step,FULL197=x,causal197=y))
 stats=lambda gate:dict(Native_match=gate['Native_match'],tensor_count=gate['tensor_count'],max_abs=max(r['max_abs'] for r in gate['rows']),mean_abs_mean=statistics.mean(r['mean_abs'] for r in gate['rows']),bit_equal_tensors=sum(r['bit_equal'] for r in gate['rows']))
 return dict(case=case.name,component_numerically_qualified=row['component_numerically_qualified'],
  all197hidden=stats(row['all197hidden']),all64queryKV=stats(row['all64queryKV']),
  real_normal_query_prediction_differences=len(queries),normal_query_differences=queries,
  actual_verifier_emitted_token_differences=len(verdict),verdict_different_owners=len({v['owner'] for v in verdict}),verdict_differences=verdict,
  accepted_node_indices_equal=left['actual_verdict_accepted_node_indices']==right['actual_verdict_accepted_node_indices'],
  final_prompt_tokens={m:row['real_outputs'][m]['final_prompt_token_ids'] for m in ('FULL197','causal197')},
  auxiliary_head197_physical_argmax_differences=sum(a!=b for a,b in zip(left['auxiliary_all197_physical_token_ids'],right['auxiliary_all197_physical_token_ids'],strict=True)),
  dummy_or_partial_prompt_argmax_not_labeled_user_output=True,local_fixture_verdict_not_actual_online_publication=True)


def run(tag):
 sys.path.insert(0,str(P))
 from check_v117 import require_ready,FREEZE,CPU
 from linear_mixed_contract_v117 import cases,branch_case,fixture,MODES,NATIVE_COUNTERS
 from audit_linear_mixed_v117 import audit_reports
 from source_snapshot_v117 import runtime_outputs
 receipt_path=P/f'receipt-{tag}.json';receipt=read(receipt_path)
 need(receipt['status'] in ('complete','failed') and 'devices_after' in receipt and receipt.get('child',{}).get('exit_code') is not None,'original run has not reached terminal cleanup')
 ready=require_ready();freeze=read(FREEZE);cpu=read(CPU)
 need(receipt['readiness']==json.loads(json.dumps(ready)) and receipt['external_input_hashes']==freeze['external_files'],'actual source/CPU epoch differs')
 need(receipt['experimental_sources_unchanged'] is receipt['external_sources_unchanged'] is True,'original source closure failed')
 need(cpu['tests_run']==18 and cpu['failures']==cpu['errors']==0 and cpu['input_hashes']==cpu['input_hashes_after']==freeze['files']
  and cpu['external_input_hashes']==cpu['external_input_hashes_after']==freeze['external_files'],'actual18 HOST source identity')
 prepath=D/'ROOT_V117_PRE_NPU_SOURCE_AND_HOST_AUDIT.json';pre=read(prepath)
 need(sha(prepath)=='435bb28d9f71cdf94269a57d848eee0f0a0ba39811c20574bb74a04a38b4d54e' and pre['status']=='passed'
  and pre['CPU_sha256']==sha(CPU) and pre['freeze_sha256']==sha(FREEZE),'ROOT independent prelaunch proof')
 need(all(sha(p)==h for p,h in receipt['source_hashes'].items()),'actual archive source map differs')
 actual_tar=archive_matches(receipt['source_snapshot'],{str(Path(p).relative_to(P)):h for p,h in receipt['source_hashes'].items()})
 need(actual_tar['sha256']==receipt['source_snapshot_sha256'],'actual archive SHA')
 prepared={str(Path(p).relative_to(P)):h for p,h in freeze['files'].items()}
 for path in (CPU,FREEZE,P/'CPU_V117.log',*sorted((P/'host-attempts').glob('*'))):prepared[str(path.relative_to(P))]=sha(path)
 prepared_tar=archive_matches(P/'PREPARED_SOURCE_ARCHIVE_V117.tar.gz',prepared)
 need(prepared_tar['sha256']==pre['archive_sha256'] and prepared_tar['members']==28,'prepared archive actual bytes')
 runtime=B/'serial-v1/runtime.sh';sourced=re.findall(r'^source (/[^ \n]+)',runtime.read_text(),re.MULTILINE);shells={p:sha(p) for p in sourced}
 need(len(sourced)==2 and shell_rows(receipt['actual_sourced_runtime_shells_before'])==shell_rows(receipt['actual_sourced_runtime_shells_after'])==shells
  and pre['runtime_shells']=={str(runtime):sha(runtime),**shells},'actual runtime and sourced shell before/after SHA')
 need(receipt['runtime_aliases_before']==receipt['runtime_aliases_after'],'runtime aliases changed')
 for side in ('pre','post'):
  v=receipt[side+'_verifier'];need(v['exit_code']==0 and sha(v['log'])==v['log_sha256'] and Path(v['log']).stat().st_size==v['log_bytes'],'original verifier/log')
 for side in ('before','after'):
  s=receipt['devices_'+side];need(s['all_idle'] is s['four_910B2'] is True and {r[0] for r in s['parsed_hardware']}=={2,5,6,7}
   and all(v['exit_code']==0 and v['idle'] is True for v in s['processes'].values()),'historical four-card hardware/idle cleanup evidence')
 need(receipt['owned_remaining']==[] and runtime_outputs(P)==receipt['runtime_outputs'],'owned cleanup/known fusion record')
 directory=P/'actual'/tag
 paths=[directory/f'rank{r}-linear-mixed197.json.gz' for r in range(4)]
 records=[artifact(p) for p in paths if p.exists()]
 errors=[read(p) for p in sorted(directory.glob('rank*-last-error.json'))]
 common=dict(receipt=artifact(receipt_path),original_status=receipt['status'],original_errors=receipt['errors'],
  actual_archive=actual_tar,prepared_archive=prepared_tar,CPU=artifact(CPU),freeze=artifact(FREEZE),ROOT_prelaunch=artifact(prepath),
  actual_sourced_shells=shells,original_last_errors=errors,available_worker_artifacts=records,
  source_and_terminal_cleanup_checked=True,NPU_started_by_auditor=False,full_workload_or_AR_or_GP_qualified=False)
 if receipt['status']=='failed':
  return dict(status='original_run_failed',**common,numerical_full_component_qualification=False,
   available_partial_counts=[dict(rank=e['rank'],qualification=len((e.get('partial_actor') or {}).get('qualification',[])),
    references=len((e.get('partial_actor') or {}).get('references',[])),cost=len((e.get('partial_actor') or {}).get('cost',[]))) for e in errors],
   failure_does_not_erase_available_actual_partial_evidence=True)
 need(receipt['errors']==[] and receipt['child']['exit_code']==0 and len(records)==4,'original complete/4reports')
 reports=[read(p) for p in paths];contract=audit_reports(reports)
 completed=read(directory/'completed.json');exited=read(directory/'worker-exit.json')
 need(completed['actual_component_audit']==contract and completed['replies']==receipt['completed_replies']
  ==[['calibrated',r,{'rows':0,'timings':0}] for r in range(4)] and exited['closed'] is exited['all_exited'] is True,'actual original completion/exit')
 source=fixture();bycase={c.name:c for c in (*cases(),branch_case())};negative=[]
 for report in reports[1:]:
  a=report['result'];banks={b['bank_id']:b for b in a['bank_evidence']}
  for q in a['qualification']:
   c=bycase[q['case']];bank=banks[c.name+'/candidate']
   for phase in q['phases']:verify_actual_input(phase['output'],c,q['mode'],source,bank,q['reverse'],phase['phase']=='changed')
   tensor_gate(q['FTF_all197hidden'],1,[197,5120]);tensor_gate(q['FTF_all64queryKV'],128,[197,3,128])
  for q in a['references']:
   c=bycase[q['case']];verify_actual_input(q['output'],c,q['mode'],source,banks[c.name+'/independent-reference'],q['reverse'])
   tensor_gate(q['all197hidden'],1,[197,5120]);tensor_gate(q['all64queryKV'],128,[197,3,128])
  c=branch_case();branch=a['branch_fallback']
  for phase in branch['qualification']['phases']:verify_actual_input(phase['output'],c,'FULL197',source,banks[c.name+'/candidate'],False,phase['phase']=='changed')
  verify_actual_input(branch['independent_reference']['output'],c,'FULL197',source,banks[c.name+'/independent-reference'])
  events={e['stage']:e for e in report['boundary_events']}
  for row in [*a['cost'],*a['zero_traces'][0],*a['zero_traces'][1]]:
   label='V117/'+row['role']+'/'+row['case']+'/'+row['mode']+'/'+str(row['sample'])
   prelude=events[label+'/all4-prelude-ready'];measure=events[label+'/all4-forward-timer-only']
   need(measure['sequence']==prelude['sequence']+1,'actual all4 prelude not immediately before measurement')
  for cross in a['cross_backend']:
   tensor_gate(cross['all197hidden'],1,[197,5120]);tensor_gate(cross['all64queryKV'],128,[197,3,128])
   left,right=(cross['real_outputs'][m] for m in MODES)
   expected_exact={key:left[key]==right[key] for key in cross['exact_output_comparisons']}
   need(expected_exact==cross['exact_output_comparisons'],'cross actual output comparison changed')
   real=all(expected_exact[k] for k in ('actual_verdict_token_ids','actual_verdict_accepted_node_indices','final_prompt_token_ids'))
   need(cross['actual_real_verdict_and_final_prompt_equal']==real and cross['component_numerically_qualified']==bool(real and cross['all197hidden']['Native_match'] and cross['all64queryKV']['Native_match']),'component numerical qualification overclaimed')
 # Actual-data negatives, not synthetic device success fixtures.
 for kind in ('task_skip','TP_production_tokens','MC2_route','cold_flag'):
  bad=copy.deepcopy(reports)
  if kind=='task_skip':bad[1]['result']['cost'][0]['task_updates']['delta']['task_update_skip_replays']=1
  elif kind=='TP_production_tokens':bad[2]['result']['cost'][0]['output']['normal_query_token_ids'][0]^=1
  elif kind=='MC2_route':bad[3]['construction']['actual_MC2_preflight']['route_details'][0]['enabled']=False
  else:bad[1]['result']['qualification'][0]['phases'][0]['runtime_validated_after']=True
  try:audit_reports(bad)
  except (RuntimeError,KeyError,TypeError,ValueError):negative.append(dict(kind=kind,rejected=True))
  else:raise RuntimeError('actual evidence negative accepted '+kind)
 costs=[]
 for c in cases():
  samples={m:[max(next(x['whole_target_complete_ms'] for x in r['result']['cost'] if x['case']==c.name and x['mode']==m and x['sample']==i) for r in reports[1:]) for i in range(5)] for m in MODES}
  med={m:statistics.median(v) for m,v in samples.items()}
  costs.append(dict(case=c.name,method='median of5 paired max(target TP ranks) whole-component host elapsed samples',
   samples_ms=samples,median_ms=med,FULL_over_causal_ratio=med['FULL197']/med['causal197'],component_ms_scope='strictmetadata+hidden+selectedhead+devicecompletion; common-stage prelude excluded'))
 return dict(status='passed',**common,actual_component_contract=contract,actual_data_negative_checks=negative,
  complete_model_input_and_HOST_verifier_reconstruction=True,allhidden_allKV_device_gates_checked=True,
  HOST_full_tensor_numerical_recomputation_possible=False,full_hidden_KV_vectors_not_retained=True,
  cross_backend_findings={str(r['rank']):[cross_summary(row,bycase[row['case']],source) for row in r['result']['cross_backend']] for r in reports[1:]},
  adoption_recommended=bool(contract['cross_backend_component_numerically_qualified']),
  interpretation='FULL and causal are distinct numerical backend paths; real verdict differences block adoption. No unique root cause, AR advantage, full workload or Goodput claim.',
  actual_whole_component_cost=costs,whole_workload_gain_proved=False,formal_Goodput=False)


def main():
 parser=argparse.ArgumentParser();parser.add_argument('--tag',required=True);parser.add_argument('--output')
 args=parser.parse_args();need(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag),'private tag')
 out=Path(args.output) if args.output else D/f'ROOT_V117_ACTUAL_{args.tag}_AUDIT.json'
 need(not out.exists(),'refuse overwrite actual audit')
 try:result=run(args.tag)
 except BaseException:result=dict(status='audit_failed',error=traceback.format_exc(),original_receipt_not_modified=True,formal_Goodput=False)
 result.update(schema='ROOT.V117.actual-linear-mixed197-independent-audit.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  actual_HOST_argv=sys.orig_argv,auditor_sources={str(Path(__file__).resolve()):sha(__file__),str(D/'analyze_v110_actual_cpu_context.py'):sha(D/'analyze_v110_actual_cpu_context.py')},
  Torch_imported='torch' in sys.modules,NPU_started=False)
 with out.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(output=str(out),sha256=sha(out),status=result['status'],original_status=result.get('original_status'),
  component=result.get('actual_component_contract'),cost=result.get('actual_whole_component_cost'),error=result.get('error'))))
 return 0 if result['status'] in ('passed','original_run_failed') else 1
if __name__=='__main__':raise SystemExit(main())
