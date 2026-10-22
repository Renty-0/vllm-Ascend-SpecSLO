"""Failure-only retained-byte audit; never invokes the full-only V193 auditor."""
import ast,datetime,gzip,hashlib,importlib.abc,importlib.util,json,sys,tarfile,time
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics';P=B/'tree-causal80-fused-bonus-verdict-full-v193'
BASE=B/'tree-high-normal-cap4-native-families-v148'
TAG='causal80-fused-verdict-v193-full463-v1';A=P/'actual'/TAG
RECEIPT=P/('receipt-'+TAG+'.json')
EXPECTED_RECEIPT='06f7d033f279c8a2745377116787db5a49fa6a2e5207d808d87385fa82967df4'
OUT=D/'V193_INDEPENDENT_TERMINAL_FAILURE_AUDIT.json'

class Firewall(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend','transformers'):
   raise ImportError('V193 failure audit has no backend: '+fullname)

def need(value,message):
 if value is not True:raise RuntimeError('V193 failure HOST audit: '+message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def gz(path):return json.loads(gzip.decompress(Path(path).read_bytes()))
def merge(*maps):
 result={}
 for values in maps:
  for path,digest in values.items():
   need(path not in result or result[path]==digest,'conflicting pinned bytes')
   result[path]=digest
 return result

def source_audit(receipt):
 freeze=P/'FROZEN_SOURCE193.json';cpu=P/'CPU_RESULT193.json'
 need(sha(freeze)==receipt['readiness']['freeze_sha256'] and sha(cpu)==receipt['readiness']['CPU_sha256'],
      'terminal original freeze/CPU identity')
 f=read(freeze)
 need(f['ROOT_source_reviewed']is True and f['CPU_sha256']==sha(cpu),'original ROOT source gate')
 maps=(f['files'],f['external_files'],f['retained_HOST_evidence'],f['execution_guard_inputs'],f['peer_reviews'])
 inputs=merge(*maps,{str(cpu):sha(cpu)})
 need(read(f['root_review']['path'])['reviewed_inputs']==inputs,'original exact ROOT reviewed inputs')
 sources=merge(inputs,{str(freeze):sha(freeze),f['root_review']['path']:f['root_review']['sha256']})
 for path,digest in sources.items():need(sha(path)==digest,'current selected frozen source bytes '+path)
 need(receipt['readiness']['root_review']==f['root_review'] and receipt['readiness']['peers']==f['peer_reviews']
      and receipt['readiness']['dependency_archives']==f['dependency_archives_verified_in_place'],
      'same prepared review/peer/dependency identity')
 c=read(cpu)
 need(c['status']=='passed' and c['local_before']==c['local_after']==f['files']
      and c['external_before']==c['external_after']==f['external_files'],'unchanged original CPU source maps')
 archive=receipt['readiness']['archive'];path=Path(archive['path'])
 need(path==P/'PREPARED_SOURCE_ARCHIVE193.tar.gz' and sha(path)==archive['sha256'],'own tar binary SHA')
 with tarfile.open(path,'r:gz')as bundle:
  members=bundle.getmembers()
  need(len(members)==len(sources)==archive['members'] and len({m.name for m in members})==len(members)
       and {m.name for m in members}=={p.lstrip('/')for p in sources},'exact source archive inventory')
  for member in members:
   need(member.isfile() and hashlib.sha256(bundle.extractfile(member).read()).hexdigest()==sources['/'+member.name],
        'all regular member bytes '+member.name)
 return dict(freeze_sha256=sha(freeze),CPU_sha256=sha(cpu),source_map_counts=[len(m)for m in maps],
             all_current_selected_bytes_exact=True,regular_archive_members=len(sources),archive=archive),sources

def baseline_audit(receipt):
 evidence={}
 for phase in ('pre','post'):
  row=receipt[phase+'_verifier'];path=Path(row['path'])
  need(row['exit_code']==0 and sha(path)==row['sha256'],'original verifier byte/result')
  candidates=[]
  for line in path.read_text().splitlines():
   try:value=json.loads(line)
   except json.JSONDecodeError:continue
   if type(value)is dict and 'checked_frozen_files'in value:candidates.append(value)
  need(len(candidates)==1 and candidates[0]==row['actual_result'],'exact raw verifier result')
  value=candidates[0]
  need(value['status']=='pass' and value['errors']==[]
       and (value['checked_frozen_files'],value['checked_external_files'],value['checked_packages'])==(10496,15,212),
       'original10496/15/212 full prepost PASS')
  evidence[phase]=dict(path=str(path),sha256=sha(path),actual_result=value)
 return evidence

def error_audit():
 records=[];common=None;pins={}
 marker='RuntimeError: V146: actual all4 common stage '
 for rank in range(4):
  path=A/f'rank{rank}-last-error.json';row=read(path);pins[str(path)]=sha(path)
  need(row['rank']==rank and row['formal_Goodput']is False and marker in row['error'],'rank error record')
  packet=ast.literal_eval(row['error'].split(marker)[-1].strip())
  need([r['rank']for r in packet]==list(range(4)) and all(r['sequence']==119
       and r['label']=='V193/causal80/pass0/cycle0/current-fused-BONUS-original-operator-and-model'for r in packet),
       'all4 same actual common failure boundary')
  need([r['rank']for r in packet if r['error']is not None]==[1]
       and 'actual same-device long contiguous M80/S16'in packet[1]['error']
       and 'verdict193.py'in packet[1]['error'] and 'line 146, in _queue'in packet[1]['error'],
       'leader-only original compound ABI guard failure propagated to all4')
  if common is None:common=packet
  else:need(packet==common,'identical retained common-stage consensus')
  partial=row['partial_causal80']
  need(partial['rank']==rank and partial['completed']is partial['math_independent_qualified']is False
       and partial['traces']==[] and partial['old_padding_behavior_equal']is None,
       'M80 is partial, not qualified')
  records.append(dict(rank=rank,path=str(path),sha256=sha(path),partial_completed=False,complete_M80_traces=0))
  p=A/'causal80-own-current-process'/f'rank{rank}-partial-component-error.json.gz'
  local=gz(p);pins[str(p)]=sha(p)
  need(local['partial']['completed']is False and local['partial']['traces']==[],
       'separate retained partial report agrees')
 need(len(list((A/'causal80-own-current-process').iterdir()))==4,'only four M80 partial files retained')
 text=(P/'verdict193.py').read_text();tree=ast.parse(text)
 queue=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_queue')
 condition=next(n for n in ast.walk(queue)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)
                and n.func.id=='require'and len(n.args)>1 and isinstance(n.args[1],ast.Constant)
                and n.args[1].value=='actual same-device long contiguous M80/S16')
 return dict(stage_sequence=119,stage_label=common[0]['label'],actual_callback_error_rank=1,
  actual_callback_traceback=common[1]['error'],compound_ABI_guard=ast.unparse(condition.args[0]),
  guard_source=dict(path=str(P/'verdict193.py'),sha256=sha(P/'verdict193.py'),line=condition.lineno),
  which_individual_operand_failed='unavailable: dtype/device/contiguity/length/metadata operands were not retained individually',
  original_BONUS_kernel_result_retained=False,complete_M80_trace_or_qualification=False,
  four_rank_error_records=records,actual_common_packets=common),pins

def base_component_audit(sources):
 directory=A/'variable-B4-own-full-process';files=sorted(directory.iterdir())
 need(len(files)==124 and all(p.is_file()and p.name.endswith('.json.gz')for p in files),'all124 original B4 raw files')
 path=BASE/'audit_v146.py';need(sources.get(str(path))==sha(path),'pinned original bounded component auditor')
 sys.path.insert(0,str(BASE))
 spec=importlib.util.spec_from_file_location('_V193_original_B4_component_only',path)
 module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 audit=module.audit_reports(directory)  # Component only, never audit193.audit/full-only.
 need(audit['component_numerical_qualified']is True,'actual original B4 bounded component completed')
 rows=[]
 for rank in range(4):
  report=gz(directory/f'rank{rank}-component.json.gz')
  need(report['completed']is report['component_numerical_qualified']is report['cache_released']is True
       and report['sealed']is False and report['complete_protocol_zero_traces']==2,'own original B4 status')
  inv=report['inventory_delta'];before={n:len(v)for n,v in inv['before'].items()};after={n:len(v)for n,v in inv['after'].items()}
  expected=dict(entries=0,draft_entries=60,target_entries=54)if rank==0 else dict(entries=0,draft_entries=0,target_entries=245)
  need(before==expected and after==dict(expected,target_entries=54 if rank==0 else 247),'original Native dict245->247 or114 retained')
  rows.append(dict(rank=rank,completed=True,numerical_component_qualified=True,component_sealed=False,
                   complete_protocol_zero_traces=2,Native_before=before,Native_after=after))
 return dict(original_component_auditor_path=str(path),original_component_auditor_sha256=sha(path),
             full_only_auditor_invoked=False,audit=audit,rank_summaries=rows,
             raw_file_count=124,raw_SHA256={str(p):sha(p)for p in files})

def main():
 sys.meta_path.insert(0,Firewall());start=time.monotonic()
 need(sha(RECEIPT)==EXPECTED_RECEIPT,'explicit terminal authorization receipt SHA')
 r=read(RECEIPT)
 need(r['status']=='failed' and r['child']['exit_code']==1 and r['owned_remaining']==[]
      and r['sources_unchanged']is True and r['devices_after']['all_idle']is True,'terminal failure/cleanup')
 need(r['physical_cards']==[2,5,6,7] and r['limits']['child_execution_seconds']==7200
      and r['elapsed_seconds']<7200 and len(r['errors'])==1
      and 'actual child completed within7200seconds'in r['errors'][0],
      'generic nonzero-child assertion, not a timeout')
 source,sources=source_audit(r);baseline=baseline_audit(r);error,error_pins=error_audit()
 exited=read(A/'worker-exit.json');need(exited==dict(closed=True,all_exited=True),'four worker processes closed')
 absent=[A/'completed.json',A/'windows',A/'prewarm']
 need(all(not p.exists()for p in absent) and not list(A.glob('rank*-local-diagnostic.json')),
      'no full/prewarm worker success or trace result')
 base=base_component_audit(sources)
 need(not any(n.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend')for n in sys.modules),'no backend imported')
 output=dict(schema='V193.independent-terminal-failure-HOST-audit.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  status='passed_failure_audit',candidate=str(P),tag=TAG,receipt=dict(path=str(RECEIPT),sha256=sha(RECEIPT)),
  actual_run_status='failed',actual_child_exit_code=1,elapsed_seconds=r['elapsed_seconds'],timeout=False,
  outer_error_classification='within7200 is the original elapsed<limit AND child.returncode==0 assertion; elapsed1100.55<7200, child1 failed.',
  source=source,baseline_prepost=baseline,owned_cleanup=dict(owned_remaining=[],sources_unchanged=True,
   actual_receipt_all4_idle=True,devices_queried_by_this_audit=False,worker_exit=exited),
  actual_failure=error,original_B4_component=base,absent_success_artifacts=[str(p)for p in absent],
  prewarm_or_warm65_or_twozero65_or_seal_or_full463_completed=False,
  new_M80_or_fused_operator_qualification_completed=False,performance_result_available=False,
  raw_Goodput_or_TPOT_not_assigned=True,whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False,
  full_only_auditor_or_test_suite_invoked=False,Torch_imported=False,NPU_used=False,model_called=False,
  frozen_source_written=False,audit_elapsed_seconds=time.monotonic()-start,
  script=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
  retained_failure_pins={**error_pins,str(A/'worker-exit.json'):sha(A/'worker-exit.json')},
  scope_limit='Original B4 raw audit passes bounded sameM/protocol gates only. New causal80 partial does not qualify fused math, full463, canonical AR or any performance metric.')
 with OUT.open('x')as stream:json.dump(output,stream,indent=2,sort_keys=True);stream.write('\n')
 print(json.dumps(dict(path=str(OUT),sha256=sha(OUT),status=output['status'],source_members=source['regular_archive_members'],
                      original_B4_raw_files=124,elapsed_seconds=output['audit_elapsed_seconds'])))

if __name__=='__main__':main()
