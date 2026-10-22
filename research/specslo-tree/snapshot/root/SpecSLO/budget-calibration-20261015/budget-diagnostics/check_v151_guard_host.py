"""Three small guard schema/arithmetic/byte-fixture checks; no core suite."""
import ast,copy,hashlib,importlib.abc,json,sys,tempfile,time
from pathlib import Path
D=Path(__file__).resolve().parent;G=D/'v151-root-execution'
class NoBackend(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):raise ImportError('guard HOST firewall '+name)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 sys.meta_path.insert(0,NoBackend());sys.path.insert(0,str(G));start=time.perf_counter();checks=[]
 import common_guard as c
 import supervise_v151 as s
 selected={str(p):sha(p) for p in G.iterdir() if p.is_file()}
 for p in G.glob('*.py'):compile(p.read_text(),str(p),'exec')
 with tempfile.TemporaryDirectory(prefix='host-fixture-',dir=G) as temporary:
  t=Path(temporary)
  # Actual parser/completion functions, synthetic schemas clearly separated
  # from any NPU result or qualification receipt.
  value=dict(status='pass',errors=[],checked_frozen_files=10496,checked_external_files=15,checked_packages=212)
  log=t/'verifier.log';log.write_text(json.dumps(value)+'\n');s.verifier_report(dict(path=str(log)))
  value['checked_frozen_files']=10495;log.write_text(json.dumps(value)+'\n')
  try:s.verifier_report(dict(path=str(log)));raise AssertionError('bad baseline count accepted')
  except RuntimeError:pass
  checks.append('actual verifier parser accepts10496/15/212 and rejects10495; all guard Python compiles')
  for status in ('passed_subset','passed_component'):
   audit=dict(status=status,structural_audit=True,hidden_KV_recomputed_by_HOST=False,production_adopted=False,parent4_qualified=False,formal_Goodput=False)
   completed=dict(action=c.ACTION,replies=[['calibrated',i,dict(cases=2,complete_zeros=2,timings=0)] for i in range(4)],audit=audit)
   assert s.completed_schema(completed,dict(closed=True,all_exited=True),audit)==status
   bad=copy.deepcopy(audit);bad['production_adopted']=True
   try:s.completed_schema(dict(completed,audit=bad),dict(closed=True,all_exited=True),bad);raise AssertionError('adoption accepted')
   except RuntimeError:pass
  keys=['fixture_counter'+str(i) for i in range(21)];zero=dict.fromkeys(keys,0)
  reports=[dict(result=dict(complete_zero=[[dict(Native_before=dict(zero),Native_after=dict(zero),Native_delta=dict(zero)) for j in range(2)] for k in range(2)])) for r in range(4)]
  assert s.native_zero_evidence(reports,keys)['raw_deltas_recomputed']==336
  reports[2]['result']['complete_zero'][1][0]['Native_after'][keys[0]]=1
  try:s.native_zero_evidence(reports,keys);raise AssertionError('nonzero Native delta accepted')
  except RuntimeError:pass
  checks.append('correct completed.audit subset/component schemas; adoption rejected; independent336 raw zero differences and nonzero rejection')
  # Exercise exact ROOT/peer SHA and archive bytes using private tiny fixtures.
  a=t/'a';a.write_bytes(b'a\n');g=t/'guard';g.write_bytes(b'guard\n');peer=t/'guard-peer.json';peer.write_text(json.dumps(dict(status='passed-source-review')))
  state=dict(local={str(a):sha(a)},external={str(g):sha(g)},evidence={});guards={str(g):sha(g)}
  peers=[(str(c.SOURCE_PEER),c.SOURCE_PEER_SHA),(str(peer),sha(peer))]
  flat=c.merge(state['local'],state['external'],state['evidence'],{str(c.CPU):c.CPU_SHA},guards,dict(peers))
  review=t/'ROOT-fixture.json';review.write_text(json.dumps(dict(status='passed_source_review',reviewed_inputs=flat)))
  binding=c.review_binding(review,sha(review),peers,state,guards);assert binding[2]==flat
  try:c.review_binding(review,'0'*64,peers,state,guards);raise AssertionError('wrong ROOT SHA accepted')
  except RuntimeError:pass
  tar=t/'fixture.tar.gz';info=c.write_archive(tar,state['local']);assert info['members']==1
  link=t/'link';link.symlink_to(a)
  try:c.checked_map({str(link):sha(a)});raise AssertionError('source symlink accepted')
  except RuntimeError:pass
  checks.append('explicit ROOT/source-peer/guard-peer SHA binding; exact regular archive member bytes; source symlink rejected')
 assert selected=={str(p):sha(p) for p in G.iterdir() if p.is_file()}
 assert len(checks)==3 and not any(n in sys.modules for n in ('torch','torch_npu','vllm','vllm_ascend'))
 row=dict(schema='V151.execution-guard-small-HOST.v1',status='passed',actual_checks=checks,actual_check_count=3,elapsed_seconds=time.perf_counter()-start,
  guard_source_hashes=selected,selected166_current_bytes_rehashed=False,selected166_prior_peer_SHA=c.SOURCE_PEER_SHA,
  core_CPU_SHA=c.CPU_SHA,core_suite_rerun=False,model_initialized=False,NPU_initialized=False,Torch_imported=False,
  baseline_verifier_executed=False,device_helper_executed=False,freeze_or_actual_archive_created=False,root_lock_opened=False)
 path=D/'V151_EXECUTION_GUARD_HOST_CHECK.json'
 with path.open('x') as f:json.dump(row,f,indent=2);f.write('\n')
 print(json.dumps(dict(path=str(path),sha256=sha(path),status='passed',actual_check_count=3,seconds=row['elapsed_seconds'])))
if __name__=='__main__':main()
