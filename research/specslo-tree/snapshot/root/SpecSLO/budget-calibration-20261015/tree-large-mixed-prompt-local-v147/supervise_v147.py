"""One bounded explicit component run. ROOT alone schedules the four NPUs."""
import argparse,datetime,fcntl,gzip,hashlib,importlib.util,json,os,re,signal,subprocess,sys,tarfile,time,traceback
from pathlib import Path
from scope_v147 import P,B,ACTION,require,sha
LOCK=B/'.ROOT_NPU_EXECUTION.lock'
MAX_SECONDS=3600;MAX_PRIVATE_BYTES=256*1024**2

def save(path,row):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(row,indent=2)+'\n');tmp.replace(path)

def helper():
 path=Path('/root/SpecSLO/resume-20261012/run_baseline_calibration-v1.py')
 spec=importlib.util.spec_from_file_location('_V147_owned_process_helpers',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 module.PHYSICAL=(2,5,6,7);return module

def verify(tag,label):
 argv=['bash',str(B/'serial-v1/runtime.sh'),'python','-B','/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/verify_lock.py','/root/data/specslo-baselines/nano-pearl-ascend-v1/lock.json']
 result=subprocess.run(argv,text=True,capture_output=True,timeout=180)
 path=P/f'{tag}-{label}-verifier.log'
 with path.open('x') as s:s.write(result.stdout+result.stderr)
 require(result.returncode==0,'retained CANN9/Native pre/post verifier')
 return dict(argv=argv,exit_code=result.returncode,path=str(path),sha256=sha(path))

def archive(tag):
 from check_v147 import collect,external,CPU,FREEZE
 retained=json.loads(FREEZE.read_text())['retained_attempt_files']
 sources={**collect(),**external(),**retained,str(CPU):sha(CPU),str(FREEZE):sha(FREEZE)}
 path=P/'source-snapshots'/f'{tag}.tar.gz';path.parent.mkdir(exist_ok=True)
 with tarfile.open(path,'x:gz') as tar:
  for p,h in sorted(sources.items()):require(sha(p)==h,'archive source changed');tar.add(p,arcname=p.lstrip('/'),recursive=False)
 with tarfile.open(path,'r:gz') as tar:
  members=tar.getmembers();require(len(members)==len(sources) and len({m.name for m in members})==len(sources),'archive unique complete inputs')
  for m in members:require(m.isfile() and hashlib.sha256(tar.extractfile(m).read()).hexdigest()==sources['/'+m.name],'archive exact bytes')
 return dict(path=str(path),sha256=sha(path),members=len(sources)),sources

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--action',choices=(ACTION,),required=True);parser.add_argument('--tag',required=True)
 args=parser.parse_args();require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.tag) is not None,'private unique tag')
 from check_v147 import require_ready,collect,external
 readiness=require_ready() # Refuses before device query unless ROOT reviewed/froze.
 path=P/f'receipt-{args.tag}.json';require(not path.exists() and not (P/'actual'/args.tag).exists(),'preserve previous attempts')
 lock=LOCK.open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 row=dict(schema='V147.actual-component-supervisor.v1',status='running',started_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  tag=args.tag,action=ACTION,errors=[],readiness=readiness,physical_cards=[2,5,6,7],
  execution_lock=dict(path=str(LOCK),nonblocking_exclusive=True,scope='this supervisor lifetime'),
  limits=dict(seconds=MAX_SECONDS,private_bytes=MAX_PRIVATE_BYTES),formal_Goodput=False,cost_measured=False)
 with path.open('x') as s:json.dump(row,s)
 child=None;h=None;sources=None
 def terminate(signum,frame):raise RuntimeError('V147 supervisor received signal '+str(signum))
 old_handlers={sig:signal.signal(sig,terminate) for sig in (signal.SIGTERM,signal.SIGINT)}
 try:
  h=helper();row['devices_before']=h.device_state();require(row['devices_before'].get('all_idle') is True,'all four physical cards idle')
  row['pre_verifier']=verify(args.tag,'pre');row['archive'],sources=archive(args.tag)
  argv=['bash',str(P/'launch_v147.sh'),'--action',ACTION,'--tag',args.tag]
  log=P/f'local-{args.tag}.log'
  with log.open('x') as stream:
   child=subprocess.Popen(argv,cwd=P,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
   row['child']=dict(pid=child.pid,argv=argv,log=str(log));start=time.monotonic()
   while child.poll() is None:
    row['elapsed_seconds']=time.monotonic()-start
    row['private_bytes']=sum(p.stat().st_size for p in P.rglob('*') if p.is_file());save(path,row)
    require(row['elapsed_seconds']<MAX_SECONDS and row['private_bytes']<MAX_PRIVATE_BYTES,'bounded component time/storage')
    time.sleep(5)
   row['child']['exit_code']=child.wait()
  require(child.returncode==0,'actual child failed; preserved report/checkpoints/log')
  directory=P/'actual'/args.tag
  completed=json.loads((directory/'completed.json').read_text());exited=json.loads((directory/'worker-exit.json').read_text())
  require(completed['action']==ACTION and completed['replies']==[['calibrated',i,dict(arms=2,prompt_phases=18,timings=0)] for i in range(4)]
   and exited==dict(closed=True,all_exited=True),'all four component workers completed/exited')
  from audit_v147 import audit_reports
  row['actual_component_audit']=audit_reports(directory);row['actual_execution_complete']=True
 except BaseException:row['errors'].append(traceback.format_exc())
 finally:
  if child is not None and h is not None:
   for sig in (signal.SIGTERM,signal.SIGKILL):
    members=h.owned_group(child.pid)
    for member in members:
     try:os.kill(member['pid'],sig)
     except ProcessLookupError:pass
    if members:time.sleep(2)
   row['owned_remaining']=h.owned_group(child.pid)
   if row['owned_remaining']:row['errors'].append('owned processes remain')
   try:row['child']['exit_code']=child.wait(timeout=15)
   except subprocess.TimeoutExpired:row['errors'].append('child could not be reaped')
  else:row['owned_remaining']=[]
  try:
   row['sources_unchanged']=sources is None or all(sha(p)==hsh for p,hsh in sources.items())
   require(row['sources_unchanged'],'source bytes changed')
   row['post_verifier']=verify(args.tag,'post')
   row['devices_after']=h.device_state() if h else None
   require(row['devices_after'] is not None and row['devices_after'].get('all_idle') is True,'all four cards idle after cleanup')
  except BaseException:row['errors'].append(traceback.format_exc())
  row['status']='complete' if not row['errors'] and row.get('actual_execution_complete') else 'failed'
  row['finished_UTC']=datetime.datetime.now(datetime.timezone.utc).isoformat();save(path,row)
  for sig,handler in old_handlers.items():signal.signal(sig,handler)
  fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
 print(json.dumps(dict(status=row['status'],receipt=str(path),errors=len(row['errors']))))
 if row['status']!='complete':raise SystemExit(1)

if __name__=='__main__':main()
