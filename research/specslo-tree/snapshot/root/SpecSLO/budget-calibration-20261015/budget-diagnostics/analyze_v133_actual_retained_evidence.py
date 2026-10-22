"""Independent, CPU-only V133 retained-evidence audit; preserve failed receipt."""
import argparse,copy,gzip,hashlib,heapq,json,os,sys,tarfile,time,traceback
from datetime import datetime,timezone
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-kernel-verdict-repeatfix-v133';TAG='two-case-teacherPA-repeatfix-v133-1'
CPU_SHA='ea84345221db6107ec7d33df8d1ecb236355411503cbf3d93ceb94e31b8fb9dd';FREEZE_SHA='8359df2d8c89aae37b50380870198e99b25284a823cd3be62f78d12f94c5614d';TAR_SHA='baf4132f477b2b9fe211dd730ede557e24fecec7cdfff87116c47a39f7d6f32b'
ROOT_AUDIT_SHA='6c6228f4cdd2fd2aab332fd325918d9969d27d8e876c6f51c3ab934760188c11'
ENV={k:'1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')};ENV['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
def need(v,m):
 if not v:raise ValueError('V133 independent audit: '+m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def desc(p):
 p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_bytes())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def archive(path,expected):
 names={Path(p).name:h for p,h in expected.items()}
 with tarfile.open(path,'r:gz') as stream:
  ms=stream.getmembers();need(len(ms)==len(names)==len({m.name for m in ms}) and {m.name for m in ms}==set(names) and all(m.isfile() and Path(m.name).name==m.name for m in ms),'exact local tar coverage')
  for m in ms:need(hashlib.sha256(stream.extractfile(m).read()).hexdigest()==names[m.name],'exact tar member bytes '+m.name)
 return dict(**desc(path),members=len(names),all_member_bytes_verified=True,external_inputs_SHA_only=True)
def differences(a,b,path=''):
 if type(a)!=type(b):return [dict(path=path,kind='type',left=type(a).__name__,right=type(b).__name__)]
 if type(a) is dict:
  need(set(a)==set(b),'comparison exact dictionary keys '+path)
  return [d for k in a for d in differences(a[k],b[k],path+'/'+k)]
 if type(a) is list:
  need(len(a)==len(b),'comparison exact array length '+path)
  return [d for i,(x,y) in enumerate(zip(a,b)) for d in differences(x,y,path+'/'+str(i))]
 return [] if a==b else [dict(path=path,kind='value',left=a,right=b)]
def tensor_records(v):
 if type(v) is dict:
  if 'artifact' in v and 'tensor_manifest' in v:yield v
  if 'hidden_artifact' in v and 'hidden_tensor_manifest' in v and v['hidden_artifact'].get('path'):
   yield dict(artifact=v['hidden_artifact'],tensor_manifest=v['hidden_tensor_manifest'])
  for x in v.values():yield from tensor_records(x)
 elif type(v) is list:
  for x in v:yield from tensor_records(x)
def run():
 need(all(os.environ.get(k)==v for k,v in ENV.items()),'strict bounded CPU/autoload environment required')
 import torch
 torch.set_num_threads(1)
 need(torch.get_num_threads()==1 and 'torch_npu' not in sys.modules,'CPU-only Torch import, no NPU backend import')
 root=P/'actual'/TAG;rp=P/f'receipt-{TAG}.json';receipt=read(rp);immutable_before={str(p):sha(p) for p in root.rglob('*') if p.is_file()};immutable_before[str(rp)]=sha(rp)
 need(receipt['status']=='failed' and receipt['child']['exit_code']==0 and len(receipt['errors'])==1 and 'V125 actual worker/complete audit differs' in receipt['errors'][0],'retain exact failed outer / successful child status')
 need(receipt['owned_remaining']==[] and receipt['experimental_sources_unchanged'] is receipt['external_sources_unchanged'] is True,'actual owned cleanup and unchanged maps')
 need(receipt['physical_cards']==[2,5,6,7] and read(root/'worker-exit.json')==dict(closed=True,all_exited=True),'actual physical card mapping and exited workers')
 for side in ('before','after'):
  state=receipt['devices_'+side];need(state['all_idle'] is state['four_910B2'] is True and all(x['idle'] is True and x['exit_code']==0 for x in state['processes'].values()),'retained actual pre/post idle')
 cp=P/'CPU_RESULT_V125.json';fp=P/'PREPARED_SOURCE_HASHES_V125.json';cpu=read(cp);freeze=read(fp)
 need(sha(cp)==CPU_SHA and sha(fp)==FREEZE_SHA and cpu['status']=='passed' and cpu['tests_run']==16 and cpu['preparation_epoch']==freeze['preparation_epoch']=='V133' and freeze['CPU_sha256']==CPU_SHA,'actual sixteen HOST / freeze identity')
 need(cpu['NPU_initialized'] is cpu['model_initialized'] is cpu['Torch_imported'] is False and cpu['input_hashes']==cpu['input_hashes_after']==freeze['files'] and cpu['external_input_hashes']==cpu['external_input_hashes_after']==freeze['external_files'],'HOST before/after source closure')
 for path,wanted in {**freeze['files'],**freeze['external_files']}.items():need(sha(path)==wanted,'current frozen source SHA '+path)
 expected={**freeze['files'],str(cp):CPU_SHA,str(fp):FREEZE_SHA};need(receipt['source_hashes']==expected and receipt['external_input_hashes']==freeze['external_files'],'actual startup maps equal prepared maps')
 prepared=P/'PREPARED_SOURCE_ARCHIVE_V133.tar.gz';need(sha(prepared)==TAR_SHA,'prepared archive identity');archives=[archive(prepared,expected),archive(receipt['source_snapshot'],expected)];need(archives[-1]['sha256']==receipt['source_snapshot_sha256'],'actual source snapshot identity')
 for side in ('pre','post'):
  v=receipt[side+'_verifier'];need(v['exit_code']==0 and sha(v['log'])==v['log_sha256'],'retained original runtime verifier')
 need(receipt['actual_sourced_runtime_shells_before']==receipt['actual_sourced_runtime_shells_after'] and receipt['runtime_aliases_before']==receipt['runtime_aliases_after'],'runtime source shell and alias stability')
 for r in receipt['actual_sourced_runtime_shells_before']:need(sha(r['path'])==r['sha256'],'actual runtime shell bytes')
 sys.path.insert(0,str(P));import check_v125,audit_kernel_verdict_v125 as audit
 check_v125.source_contract()
 paths=[root/f'rank{r}-kernel-verdict.json.gz' for r in range(4)];reports=[json.loads(gzip.decompress(p.read_bytes())) for p in paths]
 original=audit.audit_reports(reports,private_directory=root)
 completed=read(root/'completed.json');need(completed['action']==receipt['action'] and completed['formal_Goodput'] is False,'actual completed envelope')
 need(completed['replies']==[['calibrated',r,dict(rows=0,timings=0)] for r in range(4)],'exact allfour actual replies')
 retained=completed['actual_component_audit'];diff=differences(original,retained)
 need(diff==[dict(path='/actual_private_bytes',kind='value',left=9271692,right=8959495)],'only actual_private_bytes differs, no type/schema/tensor difference')
 late=[desc(root/n) for n in ('completed.json','worker-exit.json')];total=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
 need(total==original['actual_private_bytes'] and total-sum(x['bytes'] for x in late)==retained['actual_private_bytes'] and sum(x['bytes'] for x in late)==312197,'exact late-file size accounting')
 runtime=(P/'kernel_verdict_runtime_v125.py').read_text();supervisor=(P/'supervise_kernel_verdict_v125.py').read_text()
 need(runtime.index('audit=audit_reports(reports,private_directory=directory)')<runtime.index("save(directory,'completed.json'")<runtime.index("save(directory,'worker-exit.json'"),'actual producer audit then two late file writes')
 need("completed.get('actual_component_audit')!=audited" in supervisor,'actual outer compares whole volatile audit result')
 root_audit_path=D/'ROOT_V133_ACTUAL_COMPLETION_JSON_BOUNDARY_AUDIT.json';need(sha(root_audit_path)==ROOT_AUDIT_SHA,'root independently preserved comparison identity');root_audit=read(root_audit_path)
 need(root_audit['actual_original_audit']==original and root_audit['original_supervisor_status']=='failed' and root_audit['exact_accounted_post_audit_delta_bytes']==312197,'independent original auditor / accounting agrees ROOT result')
 # Independently enumerate and CPU-load every retained .pt, rather than accept a subset of references.
 cache={};record_count=0;tensor_count=0;numel=0
 for report in reports:
  for record in tensor_records(report):
   p=Path(record['artifact']['path']).resolve();info=record['artifact'];record_count+=1
   need(p.is_relative_to(root/'tensors') and p.suffix=='.pt' and sha(p)==info['sha256'] and p.stat().st_size==info['bytes'],'independent tensor artifact bytes')
   if str(p) in cache:
    need(cache[str(p)][1]==record['tensor_manifest'],'repeated artifact manifest must agree');continue
   values=torch.load(p,weights_only=True,map_location='cpu');need(type(values) is dict and set(values)==set(record['tensor_manifest']),'independent loaded exact tensor keys')
   for key,t in values.items():
    m=record['tensor_manifest'][key];need(type(t) is torch.Tensor and t.device.type=='cpu' and t.dtype==torch.bfloat16 and list(t.shape)==m['shape'] and m['dtype']=='torch.bfloat16' and m['finite'] is True and bool(torch.isfinite(t).all()),'independent finite BF16 tensor geometry')
    need(hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==m['sha256'],'independent raw tensor SHA')
    tensor_count+=1;numel+=t.numel()
   cache[str(p)]=(values,record['tensor_manifest'])
 need(set(cache)=={str(p.resolve()) for p in root.rglob('*.pt')} and len(cache)==46,'all46 retained tensors files closed, no unreferenced private tensor')
 def vals(record):return cache[str(Path(record['artifact']['path']).resolve())][0]
 def logits(records,point,key):
  values=[];tokens=[];pair=point['pair_token_ids']
  for rank,record in enumerate(records,1):
   rows=[x for x in record['capture']['rows'] if x['point_id']==point['point_id']];need(len(rows)==1,'independent point unique row');row=rows[0]
   lo,hi=row['local_vocab_start'],row['local_vocab_end'];need(lo==(rank-1)*50646 and hi==rank*50646,'three shard actual contiguous endpoints')
   shard=vals(record['raw'])[key].float().tolist();need(len(shard)==50646,'independent shard full size');values.extend(shard[:max(0,min(hi,151936)-lo)]);tokens.append(row['production_token'])
  need(len(values)==151936 and len(set(tokens))==1,'independent valid fullvocab and returned tokens')
  ranked=heapq.nlargest(2,enumerate(values),key=lambda p:(p[1],-p[0]));winner=ranked[0][0];need(winner==tokens[0],'independent exact fullvocab argmax vs production')
  return dict(production_token=winner,global_top2=[dict(token_id=i,score=s) for i,s in ranked],global_max_tie_count=sum(x==ranked[0][1] for x in values),top1_minus_top2=ranked[0][1]-ranked[1][1],pair_scores=[dict(token_id=i,score=values[i]) for i in pair],pair_first_minus_second=values[pair[0]]-values[pair[1]],valid_vocab=151936,padded_vocab_suffix_excluded=True,actual_logits_reprojected=False)
 summaries=[];independent_logits=[];targetactors=[r['result'] for r in reports[1:]]
 for finding in original['findings']:
  pt=finding['point'];pointid=pt['point_id'];case=pt['case'];productions={};controls={}
  for mode in ('FULL197','causal197'):
   obs=[next(x for x in a['observations'] if x['case']==case and x['arm']==mode) for a in targetactors];pi=next(i for i,x in enumerate(obs[0]['points']) if x['point_id']==pointid)
   productions[mode]=logits(obs,pt,f'p{pi}/local_logits');need(productions[mode]==finding['actual_production_logits'][mode],'independent production fullvocab result')
   control=[next(x for x in a['head_cross_projections'] if x['point_id']==pointid and x['source_arm']==mode) for a in targetactors];controls[mode]=logits(control,pt,'local_logits');need(controls[mode]==finding['same_actual_hidden_M1_counterfactual_logits'][mode],'independent same-hidden headM1 result')
  teacher=[next(x for x in a['teacher_forced_PA'] if x['point']['point_id']==pointid) for a in targetactors];pa=logits(teacher,pt,'local_logits');need(pa==finding['independent_teacher_forced_PA_M1_logits'],'independent teacher M1 result')
  summaries.append(dict(point_id=pointid,production={k:v['production_token'] for k,v in productions.items()},same_hidden_M1={k:v['production_token'] for k,v in controls.items()},teacher_forced_PA_M1=pa['production_token'],production_max_tie_count={k:v['global_max_tie_count'] for k,v in productions.items()},head_M1_changes_token=finding['head_M1_changes_token'],teacher_forced_not_own_AR=True,unique_kernel_root_cause_proved=False))
  independent_logits.append(dict(point_id=pointid,actual_production=productions,same_hidden_M1=controls,teacher_forced_PA_M1=pa))
 negative=[]
 for label,mutate in [('commonstage-label',lambda r:r[0]['boundary_events'][0].update(stage='wrong')),('teacher-input',lambda r:r[1]['result']['teacher_forced_PA'][0]['calls'][0].update(input_token_id=-1)),('Native-failure',lambda r:r[1]['result']['final_Native'].update({next(iter(audit.FAILURE_COUNTERS)):1}))]:
  changed=copy.deepcopy(reports);mutate(changed)
  try:audit.audit_reports(changed,private_directory=root)
  except Exception as exc:negative.append(dict(tamper=label,rejected=True,error=type(exc).__name__+': '+str(exc)))
  else:raise ValueError('tampered evidence accepted: '+label)
 need(all(sha(p)==h for p,h in immutable_before.items()),'all actual raw tensors/reports/failed receipt unchanged after audit')
 return dict(status='retained_diagnostic_evidence_passed_original_supervisor_failed',errors=[],original_supervisor_status='failed',actual_child_exit_code=0,receipt=desc(rp),actual_elapsed_seconds=receipt['elapsed_seconds'],sources=dict(local=len(freeze['files']),external_SHA_only_inputs=len(freeze['external_files']),all_current_SHA_checked=True,CPU=desc(cp),freeze=desc(fp),actual_HOST_tests=16,archives=archives,external_bytes_not_claimed_archived=True),startup_and_commonstages=dict(ranks=[0,1,2,3],exact_common_boundary_count=len(reports[0]['boundary_events']),MC2_preflights_all_four_original_checks_passed=True,Native_constructor_calls_per_rank=1,startup_graph_inventory=0,new_target_hidden_keys=4,case_count=2,target_qualification_per_rank=8,independent_same_M_reference_per_rank=8,target_zero_sweep_counts=[4,4],actual_independent_bank_epochs_per_target=list(range(1,12)),FULL_causal_FULL_restoration_cases=2,teacher_forced_PA_points=3),raw_reports=[desc(p) for p in paths],original_strict_auditor=dict(path=str(P/'audit_kernel_verdict_v125.py'),sha256=sha(P/'audit_kernel_verdict_v125.py'),reexecuted_unmodified=True,passed=True,canonical_result_sha256=hashlib.sha256(canonical(original)).hexdigest(),all197_hidden_and_all64_queryKV_fullvector_device_gates_not_HOST_recomputed=True),independent_tensors=dict(files=46,manifest_reference_count=record_count,tensor_count=tensor_count,scalar_elements=numel,all_raw_tensor_SHA_finite_dtype_shape_recomputed=True,all_actual_pt_files_referenced=True,artifact_files=[desc(p) for p in sorted(map(Path,cache))],tensor_bytes_by_target_rank=original['tensor_bytes_by_target_rank']),independent_fullvocab_logits=independent_logits,findings=summaries,postlude_failure=dict(unique_difference=diff,retained_private_bytes=retained['actual_private_bytes'],live_private_bytes=total,late_files=late,exact_delta_bytes=312197,producer_source_order_verified=True,tuple_or_intkey_JSON_hypothesis_rejected=True,original_strict_checks_unchanged=True,original_failed_receipt_preserved=True,requires_no_device_rerun=True),negative_controls=negative,root_independent_comparison=desc(root_audit_path),retained_pre_post_four_cards_idle=True,owned_remaining=[],private_actual_bytes=total,model_or_NPU_started_by_audit=False,torch_npu_imported='torch_npu' in sys.modules,whole_workload_qualified=False,independent_AR_qualified=False,unique_kernel_root_cause_proved=False,causal_backend_adoption_recommended=False,formal_Goodput=False,actual_GP=None)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();start=time.perf_counter();utc=datetime.now(timezone.utc).isoformat()
 try:value=run()
 except BaseException:value=dict(status='audit_failed',errors=[traceback.format_exc()],NPU_started=False,formal_Goodput=False)
 value.update(schema='V133.external-independent-retained-tensor-and-postlude-HOST-audit.v1',started_UTC=utc,finished_UTC=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start,actual_argv=sys.argv,actual_python=sys.executable,actual_env={k:os.environ.get(k) for k in ENV},runtime_wrapper_used=False,auditor=desc(__file__))
 with Path(args.output).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(status=value['status'],output=args.output,errors=value['errors'],elapsed_seconds=value['elapsed_seconds'])));return int(value['status']=='audit_failed')
if __name__=='__main__':raise SystemExit(main())
