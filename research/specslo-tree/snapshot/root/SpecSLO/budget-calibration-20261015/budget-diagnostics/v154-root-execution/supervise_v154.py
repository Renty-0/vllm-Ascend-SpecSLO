"""ROOT-owned single execution lock, original verifier, bounded P154 only."""
import argparse
import datetime
import fcntl
import importlib.util
import json
import gzip
import os
import resource
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
from common_guard import (P,ACTION,LOCK,MAX_SECONDS,MAX_TAG_BYTES,MAX_AGGREGATE_BYTES,
 SCOPE,require,fresh,prepared,original_helpers,sha,storage)

def verifier_report(record):
 """Original retained verifier must actually report10496/15/212 PASS."""
 values=[]
 for line in Path(record['path']).read_text().splitlines():
  try:value=json.loads(line)
  except json.JSONDecodeError:continue
  if type(value) is dict and 'checked_frozen_files' in value:values.append(value)
 require(len(values)==1,'one actual baseline verifier result')
 value=values[0]
 require(value['status']=='pass' and value['errors']==[] and value['checked_frozen_files']==10496
  and value['checked_external_files']==15 and value['checked_packages']==212,
  'actual full baseline10496/15/212 verifier PASS')
 return dict(**record,actual_result=value)

def completed_schema(completed,exited,audited):
 require(completed['action']==ACTION and completed['replies']==[['calibrated',i,
  dict(cases=1,complete_zeros=0,timings=0)] for i in range(4)] and exited==dict(closed=True,all_exited=True),
  'actual four P154 workers completed/exited')
 require(audited['status'] in ('passed-component-evidence','passed-negative-subset') and completed['audit']==audited
  and audited['hidden_KV_recomputed_by_HOST'] is False
  and audited['two_complete_zeros_qualified'] is audited['full_qualification'] is audited['production_gates_opened'] is audited['formal_Goodput'] is audited['adopted'] is False,
  'correct limited discovery/positive-consume actual schema, never production qualification')
 require((audited['positive_count']>0)==(audited['status']=='passed-component-evidence')
  and audited['positive_count']+audited['negative_count']==32,'actual real parent outcome counts')
 return 'passed_component' if audited['positive_count'] else 'discovery_subset'


def native_discovery_evidence(reports,expected):
 """Subtract all21 actual discovery counters; no zero-sweep claim."""
 expected=set(expected);require(len(expected)==21,'original21 Native counter schema')
 count=0
 for report in reports:
  value=report['result'];before,after,delta=value['Native_before'],value['Native_after'],value['Native_delta']
  require(set(before)==set(after)==set(delta)==expected,'actual Native fields')
  for key in expected:
   require(type(before[key]) is type(after[key]) is type(delta[key]) is int
    and before[key]>=0 and after[key]-before[key]==delta[key]>=0,'actual Native monotone subtraction')
   if any(word in key for word in ('failed','fallback','disabled','unclassified','validation_failures')):
    require(after[key]==0,'actual discovery no Native failures/fallbacks')
   count+=1
 require(count==84,'four actual rank counter subtractions')
 return dict(counter_fields=21,actual_rank_reports=4,raw_deltas_recomputed=count,
  two_complete_zero_traces_claimed=False,zero_discovery_qualified=False)


def actual_audit(directory):
 # Import exactly the known local audit, never enumerate arbitrary sys.modules
 # or use getattr on pseudo modules such as torch.classes.
 old=list(sys.path)
 try:
  sys.path.insert(0,str(P))
  module=sys.modules.get('scope154')
  if module is not None:
   origin=vars(module).get('__file__');require(type(origin)is str and Path(origin).resolve()==P/'scope154.py','known actual scope154 module origin')
  path=P/'audit154.py';spec=importlib.util.spec_from_file_location('_ROOT_V154_actual_audit',path)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  reports=[json.loads(gzip.decompress((directory/f'rank{rank}-component.json.gz').read_bytes())) for rank in range(4)]
  value=module.audit(reports);evidence=native_discovery_evidence(reports,module.NATIVE_COUNTERS)
  completed=json.loads((directory/'completed.json').read_text());exited=json.loads((directory/'worker-exit.json').read_text())
  status=completed_schema(completed,exited,value)
  return value,status,evidence
 finally:sys.path[:]=old


def child_limits():
 resource.setrlimit(resource.RLIMIT_FSIZE,(MAX_TAG_BYTES,MAX_TAG_BYTES))

def launch_argv(tag):
 # Fixed local launcher and original retained runtime only. No inherited
 # experiment flags, model path overrides or arbitrary child command.
 return ['/usr/bin/env','-i','HOME=/root','USER=root','LOGNAME=root','SHELL=/bin/bash','LANG=C.UTF-8',
  'PATH=/root/miniconda3/envs/vllm-hust-dev/bin:/usr/local/bin:/usr/local/sbin:/usr/bin:/usr/sbin:/bin',
  '/bin/bash',str(Path(__file__).resolve().parent/'launch_v154.sh'),'--action',ACTION,'--tag',tag]

def execute_locked(tag):
 frozen,sources,readiness=prepared()
 path=P/f'receipt-{tag}.json';log=P/f'local-{tag}.log';actual=P/'actual'/tag
 for name in (path,log,actual,P/f'{tag}-pre-verifier.log',P/f'{tag}-post-verifier.log'):fresh(name)
 require(not (P/'actual').is_symlink(),'actual root stays in private P154')
 storage(tag)
 helpers=original_helpers();save=helpers['save'];verify=helpers['verify']
 row=dict(schema='V154.root-owned-local-component-supervisor.v1',status='running',
  started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=tag,action=ACTION,
  errors=[],readiness=readiness,child=None,physical_cards=[2,5,6,7],
  execution_lock=dict(path=str(LOCK),nonblocking_exclusive=True,scope='this supervisor lifetime; no outer flock'),
  limits=dict(child_execution_seconds=MAX_SECONDS,actual_tag_bytes=MAX_TAG_BYTES,aggregate_private_bytes=MAX_AGGREGATE_BYTES),
  numeric_source_obligations_only=True,
  qualification_scope='one natural-positive R1 discovery, conditional new-union/child/next; no two complete zeros or full qualification',**SCOPE)
 with path.open('x') as stream:json.dump(row,stream)
 child=None;h=None
 def terminate(signum,frame):raise RuntimeError('V154 supervisor received signal '+str(signum))
 old_handlers={sig:signal.signal(sig,terminate) for sig in (signal.SIGTERM,signal.SIGINT)}
 try:
  h=helpers['helper']() # Fresh module namespace; parent's globals/file unchanged.
  row['devices_before']=h.device_state();require(row['devices_before'].get('all_idle') is True,'actual four physical NPUs idle')
  row['pre_verifier']=verifier_report(verify(tag,'pre'));storage(tag)
  # Recheck self guard/source bytes immediately before launching actual work.
  prepared()
  argv=launch_argv(tag)
  row['child']=dict(argv=argv,log=str(log))
  with log.open('x') as stream:
   child=subprocess.Popen(argv,cwd=P,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,preexec_fn=child_limits)
   row['child']['pid']=child.pid;start=time.monotonic()
   while child.poll() is None:
    row['elapsed_seconds']=time.monotonic()-start;row['private_storage']=storage(tag);save(path,row)
    require(row['elapsed_seconds']<MAX_SECONDS,'bounded1800second actual child')
    time.sleep(1)
   row['elapsed_seconds']=time.monotonic()-start;row['child']['exit_code']=child.wait()
  require(row['elapsed_seconds']<MAX_SECONDS and child.returncode==0,'actual child completed within1800seconds')
  storage(tag)
  prepared()
  row['actual_component_audit'],row['actual_local_coverage_status'],row['independent_native_discovery_arithmetic']=actual_audit(actual)
  row['actual_execution_complete']=True

 except BaseException:row['errors'].append(traceback.format_exc())
 finally:
  try:helpers['owned_cleanup'](child,h,row)
  except BaseException:row['errors'].append(traceback.format_exc())
  try:
   row['sources_unchanged']=all(sha(name)==wanted for name,wanted in sources.items())
   require(row['sources_unchanged'],'source/CPU/freeze/guard/ROOT/peer bytes unchanged')
   prepared()
  except BaseException:row['errors'].append(traceback.format_exc())
  # Post verifier and final actual idle proof are attempted even after a failed
  # child/cleanup/source check, while this same supervisor still owns ROOT lock.
  try:row['post_verifier']=verifier_report(verify(tag,'post'))
  except BaseException:row['errors'].append(traceback.format_exc())
  try:
   row['devices_after']=h.device_state() if h else None
   require(row['devices_after'] is not None and row['devices_after'].get('all_idle') is True,'actual four NPUs idle after owned cleanup')
   row['private_storage_final']=storage(tag)
  except BaseException:row['errors'].append(traceback.format_exc())
  row['status']='complete' if not row['errors'] and row.get('actual_execution_complete') else 'failed'
  row['finished_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(path,row)
  for sig,handler in old_handlers.items():signal.signal(sig,handler)
 print(json.dumps(dict(status=row['status'],receipt=str(path),errors=len(row['errors']),**SCOPE)))
 if row['status']!='complete':raise SystemExit(1)

def main():
 import re
 parser=argparse.ArgumentParser();parser.add_argument('--action',choices=(ACTION,),required=True);parser.add_argument('--tag',required=True)
 args=parser.parse_args();require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag) is not None,'safe fresh private tag')
 prepared() # HOST byte/archive checks before any device query or helper call.
 lock=LOCK.open('a+')
 try:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  execute_locked(args.tag)
 finally:
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()

if __name__=='__main__':main()
