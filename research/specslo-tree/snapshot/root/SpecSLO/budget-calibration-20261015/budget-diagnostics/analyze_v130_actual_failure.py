"""Independent V130 failed-attempt journal/source/partial-evidence HOST audit."""
import argparse,ast,copy,gzip,hashlib,importlib.abc,json,math,re,sys,tarfile,traceback,zipfile
from datetime import datetime,timezone
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');P=B/'tree-kernel-verdict-launchfix-v130';D=B/'budget-diagnostics';TAG='two-case-teacherPA-launchfix-v1'
CPU_SHA='853cc92e0421d160c2eb91b7786c90f401f25a20efaf44b656af02b47dfd7118';FREEZE_SHA='6bf027872934686dea4db14a499434a054b6d2bd54285ddc71e0fdfb5712ad24';TAR_SHA='e4f062c8f8e370afe00a62820c426cdaeb9efefcf635df8e1ea51547ade8f96d'

def need(v,m):
 if not v:raise ValueError('V130 failure: '+m)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def desc(p):
 p=Path(p).resolve();return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def read(p):return json.loads(Path(p).read_bytes())
def tar(path,expected):
 names={Path(p).name:s for p,s in expected.items()}
 with tarfile.open(path,'r:gz') as source:
  ms=source.getmembers();need(len(ms)==len(names)==len({m.name for m in ms}) and {m.name for m in ms}==set(names) and all(m.isfile() and Path(m.name).name==m.name for m in ms),'exact local archive members')
  for m in ms:need(hashlib.sha256(source.extractfile(m).read()).hexdigest()==names[m.name],'archive member bytes')
 return dict(**desc(path),members=len(names),all_bytes_recomputed=True)
class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST forbids '+fullname)
def refs(value):
 if type(value) is dict:
  if type(value.get('path')) is str and value['path'].endswith('.pt'):yield value
  for v in value.values():yield from refs(v)
 elif type(value) is list:
  for v in value:yield from refs(v)

def run():
 root=P/'actual'/TAG;rp=P/f'receipt-{TAG}.json';receipt=read(rp)
 need(receipt['status']=='failed' and receipt['child']['exit_code']==1 and receipt['owned_remaining']==[]
  and receipt['experimental_sources_unchanged'] is receipt['external_sources_unchanged'] is True,'terminal failed/source/owned cleanup')
 for side in ('before','after'):
  state=receipt['devices_'+side];need(state['all_idle'] is state['four_910B2'] is True and all(v['idle'] is True and v['exit_code']==0 for v in state['processes'].values()),'retained pre/post four card idle')
 need(receipt['physical_cards']==[2,5,6,7] and read(root/'worker-exit.json')==dict(closed=True,all_exited=True),'physical cards/all workers exited')
 cp=P/'CPU_RESULT_V125.json';fp=P/'PREPARED_SOURCE_HASHES_V125.json';cpu=read(cp);freeze=read(fp)
 need(sha(cp)==CPU_SHA and sha(fp)==FREEZE_SHA and cpu['status']=='passed' and cpu['tests_run']==13 and cpu['preparation_epoch']=='V130'
  and cpu['NPU_initialized'] is cpu['model_initialized'] is False and cpu['input_hashes']==cpu['input_hashes_after']==freeze['files']
  and cpu['external_input_hashes']==cpu['external_input_hashes_after']==freeze['external_files'] and freeze['CPU_sha256']==CPU_SHA,'actual13 HOST/freeze')
 for path,wanted in {**freeze['files'],**freeze['external_files']}.items():need(sha(path)==wanted,'source SHA changed')
 expected={**freeze['files'],str(cp):CPU_SHA,str(fp):FREEZE_SHA};need(receipt['source_hashes']==expected and receipt['external_input_hashes']==freeze['external_files'],'actual source maps')
 prepared=P/'PREPARED_SOURCE_ARCHIVE_V130.tar.gz';need(sha(prepared)==TAR_SHA,'prepared tar SHA');archives=[tar(prepared,expected),tar(receipt['source_snapshot'],expected)]
 need(archives[1]['sha256']==receipt['source_snapshot_sha256'],'actual tar SHA')
 for side in ('pre','post'):
  verifier=receipt[side+'_verifier'];need(verifier['exit_code']==0 and sha(verifier['log'])==verifier['log_sha256'],'actual retained verifier')
 need(receipt['actual_sourced_runtime_shells_before']==receipt['actual_sourced_runtime_shells_after'] and receipt['runtime_aliases_before']==receipt['runtime_aliases_after'],'runtime shell/alias stability')
 for item in receipt['actual_sourced_runtime_shells_before']:need(sha(item['path'])==item['sha256'],'runtime source shell SHA')
 errors=[read(root/f'rank{r}-last-error.json') for r in range(4)];sequences=[]
 for rank,record in enumerate(errors):
  need(record['rank']==rank and record['original_failure_preserved'] is True and record['formal_Goodput'] is False,'actual4 failure envelope')
  events=record['boundary_events'];need(len(events)==67 and [v['sequence'] for v in events]==list(range(67)),'actual67 contiguous stages')
  for i,event in enumerate(events):
   rows=event['ranks'];need(len(rows)==4 and [v['rank'] for v in rows]==list(range(4)) and all(v['sequence']==i and v['error'] is None for v in rows),'actual common sequence/error packets')
   if i<66:need(all(v['stage']==event['stage'] for v in rows),'stages0..65 must agree')
   else:
    need(rows[0]['stage']=='V125/N16B4-partial/observe/FULL197/prepare-local' and all(v['stage']=='V125/N16B4-partial/observe/FULL197/repeat/prepare-local' for v in rows[1:]),'exact original failing label split')
  sequences.append(events)
 need(all([e['ranks'] for e in seq]==[e['ranks'] for e in sequences[0]] for seq in sequences),'all4 failure journal packet identity')
 guard=Firewall();sys.meta_path.insert(0,guard);sys.path.insert(0,str(P))
 try:
  import check_v125, audit_kernel_verdict_v125 as audit
  check_v125.source_contract()
  # Keep all original numerical qualification/reference/cross/zero checks;
  # cut only the future complete-bank/repeat terminal gates unavailable at failure.
  tree=ast.parse((P/'audit_kernel_verdict_v125.py').read_text());fn=copy.deepcopy(next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name=='qualification'))
  stop=next(i for i,v in enumerate(fn.body) if isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='banks' for t in v.targets))
  fn.body=fn.body[:stop];namespace=dict(vars(audit));exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),str(P/'audit_kernel_verdict_v125.py'),'exec'),namespace)
  partial=[];artifacts={};cross=[]
  for rank,record in enumerate(errors):
   a=record['partial_actor']
   need(a['teacher_forced_PA']==[] and a['restoration_repeats']==[] and a['cost_measured'] is False,'no later teacher/repeat/cost evidence')
   if rank==0:need(a['qualification']==a['references']==a['observations']==a['head_cross_projections']==[],'draft cannot attest target values')
   else:
    namespace['qualification'](a)
    need(len(a['bank_evidence'])==6 and len(a['observations'])==len(a['head_cross_projections'])==2,'retained completed partial counts')
    need([r['arm'] for r in a['observations']]==['FULL197','causal197'] and all(r['case']=='N16B4-partial' for r in a['observations']),'first case two observed production arms')
    for x in refs(a):
     path=Path(x['path']).resolve();need(path.is_relative_to(root/'tensors') and path.stat().st_size==x['bytes'] and sha(path)==x['sha256'],'actual tensor file path/bytes/SHA')
     with zipfile.ZipFile(path) as z:need(z.testzip() is None,'torch ZIP storage CRC')
     artifacts[str(path)]=desc(path)
    for c in a['cross_backend']:cross.append(dict(rank=rank,case=c['case'],all197hidden_Native_match=c['all197hidden']['Native_match'],all64queryKV_Native_match=c['all64queryKV']['Native_match'],actual_real_verdict_and_final_prompt_equal=c['actual_real_verdict_and_final_prompt_equal'],component_numerically_qualified=c['component_numerically_qualified']))
   partial.append(dict(rank=rank,qualifications=len(a['qualification']),independent_references=len(a['references']),zero_counts=[len(z) for z in a['zero_traces']],production_observations=len(a['observations']),M1_head_controls=len(a['head_cross_projections']),teacher_forced_PA=0,restoration_repeats=0,complete_component_qualification=False))
  files=set(root.rglob('*.pt'));need(set(map(Path,artifacts))==files and len(files)==14,'all14 actual tensors are referenced:12 raw+2leader hidden')
 finally:sys.meta_path.remove(guard)
 need(not any(root.glob('rank*-kernel-verdict.json')) and not (root/'completed.json').exists(),'no complete report/completion may be inferred')
 return dict(status='failed_attempt_evidence_audited',errors=[],receipt=desc(rp),actual_elapsed_seconds=receipt['elapsed_seconds'],original_run_status='failed',
  sources=dict(local=len(freeze['files']),external_SHA_inputs=len(freeze['external_files']),all_SHA_checked=True,CPU=desc(cp),freeze=desc(fp),actual_HOST_tests=13,archives=archives,external_bytes_not_in_local_archive=True),
  failure=dict(sequence=66,preceding_all4_completed_boundaries=66,all_local_callback_errors_None=True,draft_label=sequences[0][-1]['ranks'][0]['stage'],target_label=sequences[0][-1]['ranks'][1]['stage'],
   source_cause='repeat depends on target-only first evidence set inside retain_observation; draft first remains None; third arm takes unequal labels',cleanup_protocol_did_not_continue_to_later_TP=True),
  boundary_error_artifacts=[desc(root/f'rank{r}-last-error.json') for r in range(4)],partial=partial,raw_cross_backend_markers=cross,
  tensor_artifacts=list(artifacts.values()),actual_tensor_files=14,raw_logits_KV_bundles=12,leader_hidden_snapshot_files=2,tensor_file_SHA_and_ZIP_CRC_recomputed=True,tensor_value_manifest_or_numerics_independently_recomputed=False,
  unchanged_original_qualification_reference_cross_zero_checks_reexecuted=True,unavailable_complete_bank_repeat_terminal_checks_excluded_and_not_granted=True,
  owned_remaining=[],retained_four_cards_pre_post_idle=True,worker_exit=desc(root/'worker-exit.json'),
  whole_component_completed=False,teacher_forced_PA_completed=False,FULL_causal_FULL_restoration_completed=False,new_production_qualification_granted=False,
  unique_kernel_root_cause_proved=False,full463_qualified=False,formal_Goodput=False,actual_GP=None,NPU_initialized=False,Torch_imported='torch' in sys.modules)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();start=datetime.now(timezone.utc).isoformat()
 try:value=run()
 except BaseException:value=dict(status='audit_failed',errors=[traceback.format_exc()],NPU_initialized=False,formal_Goodput=False)
 value.update(schema='V130.external-independent-terminal-FAILURE-HOST-audit.v1',started_UTC=start,finished_UTC=datetime.now(timezone.utc).isoformat(),actual_argv=sys.argv,actual_python=sys.executable,runtime_wrapper_used=False,auditor=desc(__file__))
 with Path(args.output).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
 print(json.dumps(dict(status=value['status'],output=args.output,errors=value['errors'])));return int(value['status']=='audit_failed')
if __name__=='__main__':raise SystemExit(main())
