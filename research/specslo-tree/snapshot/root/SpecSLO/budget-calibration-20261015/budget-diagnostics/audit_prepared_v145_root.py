"""Independent ROOT HOST closure review; no candidate or device import."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys,tarfile
D=Path('/root/SpecSLO/budget-calibration-20261015/budget-diagnostics')
P=D/'final-tree-normalized-bridge-v145'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def main():
 assert sys.version_info[:2]==(3,11)
 cp=P/'HOST_RESULT.json';fp=P/'PREPARED_SOURCE.json';a=P/'PREPARED_SOURCE_ARCHIVE.tar.gz'
 c,f,pins=read(cp),read(fp),read(P/'SOURCE_PINS.json')
 assert c['schema']=='V145.thirteen-new-bridge-HOST-tests.v1' and c['status']=='passed'
 assert c['tests_run']==13 and c['errors']==c['failures']==0
 assert c['NPU_initialized'] is c['Torch_imported'] is c['NPU_queries_executed'] is c['current_V143_raw_read'] is False
 assert c['backend_import_attempts']==[] and c['parent_five_normalizer_tests_reused_by_SHA'] is c['parent_nine_HOST_tests_reused_by_SHA'] is True
 assert c['V131_or_old_suites_repeated'] is c['candidate_selected'] is f['candidate_selected'] is False
 assert c['source_before']==c['source_after']==f['files'] and c['external_sources']==pins['files']
 assert len(f['files'])==9 and len(pins['files'])==35 and f['HOST_sha256']==sha(cp)
 assert sha(cp)=='e97cb01dc1c9b8516c4e8faa875193cdd359b5333b36e2875d4d4a98ec073e8a'
 assert sha(fp)=='d637ed78fef3b54c3f137d2a36f26fb5f62d93b789065ea8c7d70bdd9d5e1ba5'
 assert sha(a)=='b88c70b84b836f75d8f37490931985923b8e2b3aa1c471fec85cd2f3e37704cb'
 failed={r['path']:r['sha256'] for r in c['prior_failed_HOST_attempts']}
 assert len(failed)==2
 expected={**f['files'],**pins['files'],str(cp):sha(cp),str(fp):sha(fp),**failed}
 for n,h in expected.items():assert sha(n)==h,n
 with tarfile.open(a) as archive:
  members=archive.getmembers();assert len(members)==13 and len({m.name for m in members})==13
  wanted=set(f['files'])|{str(cp),str(fp)}|set(failed);seen=set()
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
 out=D/'ROOT_V145_PREPARED_SOURCE_HOST_AUDIT.json'
 record=dict(schema='ROOT.V145.prepared-source-HOST-audit.v1',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='passed',directory=str(P),actual_python=sys.executable,actual_HOST_tests=13,parent_five_and_nine_reused_not_rerun=True,CPU_sha256=sha(cp),freeze_sha256=sha(fp),archive_sha256=sha(a),archive_members_all_bytes_checked=13,local_sources=9,external_sources=35,normalization_parent_bytes_identical=True,checks=checks,candidate_selected=False,NPU_started=False,formal_Goodput=False)
 with out.open('x') as stream:json.dump(record,stream,indent=2);stream.write('\n')
 print(json.dumps(dict(status='passed',output=str(out),sha256=sha(out),NPU_started=False)))
if __name__=='__main__':main()
