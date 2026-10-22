"""Independent ROOT HOST closure review; no candidate or device import."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,tarfile
D=Path('/root/SpecSLO/budget-calibration-20261015/budget-diagnostics')
P=D/'final-tree-normalized-bridge-v142'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def main():
 assert sys.version_info[:2]==(3,11)
 cp=P/'HOST_RESULT.json';fp=P/'PREPARED_SOURCE.json';a=P/'PREPARED_SOURCE_ARCHIVE.tar.gz'
 c,f,pins=read(cp),read(fp),read(P/'SOURCE_PINS.json')
 assert c['schema']=='V142.nine-new-bridge-HOST-tests.v1' and c['status']=='passed'
 assert c['tests_run']==9 and c['errors']==c['failures']==0
 assert c['NPU_initialized'] is c['Torch_imported'] is c['NPU_queries_executed'] is c['current_V140_raw_read'] is False
 assert c['backend_import_attempts']==[] and c['parent_five_HOST_tests_reused_by_SHA'] is True
 assert c['V131_or_old_suites_repeated'] is c['candidate_selected'] is f['candidate_selected'] is False
 assert c['source_before']==c['source_after']==f['files'] and c['external_sources']==pins['files']
 assert len(f['files'])==8 and len(pins['files'])==17 and f['HOST_sha256']==sha(cp)
 assert sha(cp)=='596aeb310cd43b222611dec90d52b98223256eff10050cbe35ca313c6264a0e5'
 assert sha(fp)=='61e5f9f7f04b643219439736dbd197444b4d300135cb06702c224ae3aea2c21a'
 assert sha(a)=='34a993fcd3fd2f5a70c7fe9a3ea1c226b66ee0bc8ad7073c58dff17961fc35de'
 expected={**f['files'],**pins['files'],str(cp):sha(cp),str(fp):sha(fp)}
 for n,h in expected.items():assert sha(n)==h,n
 with tarfile.open(a) as archive:
  members=archive.getmembers();assert len(members)==10 and len({m.name for m in members})==10
  wanted=set(f['files'])|{str(cp),str(fp)};seen=set()
  for m in members:
   assert m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
   path=str(P/m.name);assert path in wanted
   assert hashlib.sha256(archive.extractfile(m).read()).hexdigest()==expected[path]
   seen.add(path)
  assert seen==wanted
 assert (P/'normalization.py').read_bytes()==(D/'final-tree-normalized-bridge-v137/normalization.py').read_bytes()
 checks=[]
 for script,args,code in [('prepare.py',['--require-ready'],0),('bridge.py',['--check'],1)]:
  argv=[sys.executable,'-B',str(P/script),*args]
  r=subprocess.run(argv,capture_output=True,text=True,timeout=60,cwd=P)
  assert r.returncode==code,r.stdout+r.stderr
  value=json.loads(r.stdout)
  if code==1:assert value['status']=='blocked_candidate_binding_pending' and value['NPU_queries_executed'] is False
  checks.append(dict(argv=argv,returncode=r.returncode,result=value))
 for n,h in expected.items():assert sha(n)==h,n
 out=D/'ROOT_V142_PREPARED_SOURCE_HOST_AUDIT.json'
 record=dict(schema='ROOT.V142.prepared-source-HOST-audit.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='passed',directory=str(P),actual_python=sys.executable,actual_HOST_tests=9,parent_five_reused_not_rerun=True,CPU_sha256=sha(cp),freeze_sha256=sha(fp),archive_sha256=sha(a),archive_members_all_bytes_checked=10,local_sources=8,external_sources=17,normalization_parent_bytes_identical=True,checks=checks,candidate_selected=False,NPU_started=False,formal_Goodput=False)
 with out.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
 print(json.dumps(dict(status='passed',output=str(out),sha256=sha(out),NPU_started=False)))
if __name__=='__main__':main()
