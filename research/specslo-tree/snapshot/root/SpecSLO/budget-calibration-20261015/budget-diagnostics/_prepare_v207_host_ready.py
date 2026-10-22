"""SOURCE/HOST-only closure preparation; this script never freezes or launches."""
import hashlib,importlib.util,json,sys,time
from pathlib import Path
P=Path('/root/SpecSLO/budget-calibration-20261015/tree-normal-cap3-original-geometries-full-v207')
B=P.parent;D=B/'budget-diagnostics';G=D/'v207-root-execution'
sys.path.insert(0,str(P))
from base207 import sha,read,require
from check207 import local_files,external_files
c=read(P/'CPU_RESULT207.json')
require(c['status']=='passed' and c['tests_run']==6 and c['failures']==c['errors']==0,'six current new HOST checks')
local=local_files();external=external_files()
require(local==c['local_before']==c['local_after'] and external==c['external_before']==c['external_after'],'same current source CPU')
attempts={str(p):sha(p)for p in sorted(P.glob('HOST_ATTEMPT207_*.json'))}
retained={str(p):sha(p)for root in (P/'full-host-attempts',P/'e-host-attempts') for p in sorted(root.rglob('*'))if p.is_file()}
prepared=dict(schema='V207.current-six-HOST-prepared-source.v1',files=local,external_files=external,CPU_sha256=sha(P/'CPU_RESULT207.json'),
 actual_HOST_attempts=attempts,retained_HOST_files=retained,actual_new_tests=6,old_suites_rerun=False,ROOT_source_reviewed=False,not_frozen=True)
path=P/'PREPARED_HOST207.json';path.write_text(json.dumps(prepared,indent=2,sort_keys=True)+'\n')
start=time.perf_counter();spec=importlib.util.spec_from_file_location('_V207_cold_guard_source_only',G/'guard207.py');guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
state=guard.source_state(archive=True);guards=guard.guard_inputs();storage=guard.storage()
try:
 from check207 import require_ready
 require_ready()
except FileNotFoundError as exc:
 require(Path(exc.filename)==P/'FROZEN_SOURCE207.json','entry blocked only by absent ROOT freeze')
 blocked=dict(type=type(exc).__name__,path=exc.filename)
else:raise RuntimeError('unreviewed entry unexpectedly enabled')
require(not any(name.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend')for name in sys.modules),'cold guard uses no backend')
require(local_files()==local and external_files()==external,'cold source unchanged')
row=dict(schema='V207.cold-source-archive-and-guard-HOST.v1',status='passed',elapsed_seconds=time.perf_counter()-start,
 source_state=state,guard_inputs=guards,storage=storage,ROOT_freeze_absent=True,actual_disabled_entry=blocked,
 current_six_HOST=True,parent_suites_rerun=False,parent_device_qualification_inherited=False,
 archive_bytes_read_in_place=True,old_binary_archive_copied=False,Torch_library_imported=False,NPU_used=False,model_constructed=False,
 freeze_created=False,entry_launched=False)
out=D/('V207_COLD_SOURCE_ARCHIVE_AND_GUARD_HOST_'+str(time.time_ns())+'.json')
with out.open('x')as f:json.dump(row,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(cold_path=str(out),cold_sha256=sha(out),prepared_sha256=sha(path),CPU_sha256=sha(P/'CPU_RESULT207.json'),
 local=len(local),external=len(external),attempts=len(attempts),retained=len(retained),guards=len(guards),elapsed_seconds=row['elapsed_seconds'])))
