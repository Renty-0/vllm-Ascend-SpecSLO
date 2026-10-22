"""Three small guard checks; no core rerun, backend/helper/device/lock work."""
import ast,copy,hashlib,importlib.abc,json,sys,tempfile,time,types
from pathlib import Path
D=Path(__file__).resolve().parent
G=D/'v153-root-execution'

class NoBackend(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend','transformers','safetensors'):
   raise ImportError('P153 guard HOST firewall '+name)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
 sys.meta_path.insert(0,NoBackend());sys.path.insert(0,str(G));start=time.perf_counter();checks=[]
 import common_guard as c
 import supervise_v153 as s
 selected={str(p):sha(p) for p in G.iterdir() if p.is_file()}
 for path in G.glob('*.py'):compile(path.read_text(),str(path),'exec')
 # Only one necessary current-source check, no constructor or core suite.
 source=c.source_state();guards=c.guard_inputs()
 assert len(source['local'])==19 and len(source['external'])==132 and len(source['evidence'])==8
 with tempfile.TemporaryDirectory(prefix='host-fixture-',dir=G) as temporary:
  t=Path(temporary)
  result=dict(status='pass',errors=[],checked_frozen_files=10496,checked_external_files=15,checked_packages=212)
  log=t/'verifier.log';log.write_text(json.dumps(result)+'\n');s.verifier_report(dict(path=str(log)))
  result['checked_frozen_files']=10495;log.write_text(json.dumps(result)+'\n')
  try:s.verifier_report(dict(path=str(log)))
  except RuntimeError:pass
  else:raise AssertionError('wrong original baseline count accepted')
  checks.append('exact current19/132 source and8 retained evidence; six guards/original templates;10496/15/212 parser and wrong-count rejection')
  audit=dict(status='passed_subset',passed_subset=True,raw_tensor_numerics_recomputed_on_HOST=False,
   protocol_and_counter_arithmetic_recomputed=True,MC2_645_qualified=False,adopted=False,formal_Goodput=False,
   cost_measured=False,full463_live_qualified=False)
  completed=dict(action=c.ACTION,replies=[['calibrated',i,dict(prompt_phases=18,next_drafts=6,timings=0)] for i in range(4)],audit=audit)
  assert s.completed_schema(completed,dict(closed=True,all_exited=True),audit)=='passed_subset'
  for key in ('MC2_645_qualified','adopted','cost_measured'):
   bad=dict(audit,**{key:True})
   try:s.completed_schema(dict(completed,audit=bad),dict(closed=True,all_exited=True),bad)
   except RuntimeError:pass
   else:raise AssertionError('unqualified subset incorrectly promoted '+key)
  keys=['fixture_counter'+str(i) for i in range(21)];zero=dict.fromkeys(keys,0)
  reports=[dict(traces=[dict(pass_index=p,scope=name,zero=p>0,
   Native=dict(before=dict(zero),after=dict(zero),delta=dict(zero))) for p in range(3)
   for name in ('V153-whole2301','V153-final-Q1-multipage-dummy')]) for rank in range(4)]
  assert s.native_zero_evidence(reports,keys)['raw_deltas_recomputed']==336
  reports[2]['traces'][4]['Native']['after'][keys[0]]=1
  try:s.native_zero_evidence(reports,keys)
  except RuntimeError:pass
  else:raise AssertionError('stored zero hides nonzero raw counter')
  checks.append('actual P153 completed.audit/replies schema; MC2645/adopt/cost rejected; independent16x21=336 raw differences and nonzero rejection')
  # Execute the actual receipt row constructor before any child/helper call.
  supervisor_source=(G/'supervise_v153.py').read_text();tree=ast.parse(supervisor_source)
  entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='execute_locked')
  row_node=next(n for n in entry.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='row' for x in n.targets))
  namespace=dict(datetime=s.datetime,tag='HOST-real-row-fixture',ACTION=c.ACTION,readiness={},LOCK=c.LOCK,
   MAX_SECONDS=c.MAX_SECONDS,MAX_TAG_BYTES=c.MAX_TAG_BYTES,MAX_AGGREGATE_BYTES=c.MAX_AGGREGATE_BYTES,SCOPE=c.SCOPE)
  require_no_duplicates={k.arg for k in row_node.value.keywords if k.arg is not None}.isdisjoint(c.SCOPE)
  assert require_no_duplicates
  exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(row_node)],type_ignores=[])),
   str(G/'supervise_v153.py')+'::actual-row-constructor','exec'),namespace)
  assert namespace['row']['cost_measured'] is False and namespace['row']['MC2_645_qualified'] is False
  bad=copy.deepcopy(row_node);bad.value.keywords.insert(0,ast.keyword(arg='cost_measured',value=ast.Constant(value=False)))
  try:exec(compile(ast.fix_missing_locations(ast.Module(body=[bad],type_ignores=[])),'<original-bad-row-variant>','exec'),dict(namespace))
  except TypeError as exc:assert 'cost_measured' in str(exc)
  else:raise AssertionError('original duplicate-cost row did not fail')
  checks.append('real supervisor receipt row AST executes without helper/device; original duplicate-cost variant raises TypeError')
  # Reuse exact function code in a fresh fixture namespace, never alter old/new
  # module globals or write a pretend real peer record into D.
  a=t/'source';a.write_bytes(b'fixture-source\n');guard=t/'guard';guard.write_bytes(b'fixture-guard\n')
  core=t/'source-peer.json';core.write_text(json.dumps(dict(status='passed-source-review',candidate=str(c.P))))
  peer=t/'guard-peer.json';peer.write_text(json.dumps(dict(status='passed-source-review')))
  state=dict(local={str(a):sha(a)},external={str(guard):sha(guard)},evidence={});gm={str(guard):sha(guard)}
  peers=[(str(core),sha(core)),(str(peer),sha(peer))]
  flat=c.merge(state['local'],state['external'],state['evidence'],{str(c.CPU):c.CPU_SHA},gm,dict(peers))
  root=t/'ROOT-review.json';root.write_text(json.dumps(dict(status='passed_source_review',reviewed_inputs=flat)))
  namespace=dict(c.__dict__,SOURCE_PEER=core)
  binding=types.FunctionType(c.review_binding.__code__,namespace)
  assert binding(root,sha(root),peers,state,gm)[2]==flat
  try:binding(root,'0'*64,peers,state,gm)
  except RuntimeError:pass
  else:raise AssertionError('wrong explicit ROOT SHA accepted')
  archive=t/'fixture.tar.gz';known=next(iter(c.MANAGED_MODEL_ALIASES))
  archived=dict(state['local'],**{known:c.MANAGED_MODEL_ALIASES[known]['sha256']})
  assert c.write_archive(archive,archived)['members']==2
  link=t/'source-alias';link.symlink_to(a)
  try:c.checked_map({str(link):sha(a)})
  except RuntimeError:pass
  else:raise AssertionError('source alias accepted')
  checks.append('future source-peer explicit caller SHA and exact ROOT flat review; tiny regular archive plus exact managed public-model metadata bytes; arbitrary alias rejected')
 assert selected=={str(p):sha(p) for p in G.iterdir() if p.is_file()}
 assert not any(n in sys.modules for n in ('torch','torch_npu','vllm','vllm_ascend'))
 row=dict(schema='V153.execution-guard-small-HOST.v1',status='passed',checks=checks,actual_check_count=4,
  elapsed_seconds=time.perf_counter()-start,guard_source_hashes=selected,core_CPU_SHA=c.CPU_SHA,
  current19_132_checked_once=True,core_suite_rerun=False,NPU_initialized=False,model_initialized=False,Torch_imported=False,
  baseline_verifier_executed=False,device_helper_executed=False,root_lock_opened=False,actual_raw_or_fusion_read=False,
  freeze_or_actual_archive_created=False)
 path=D/'V153_EXECUTION_GUARD_HOST_CHECK_REVISION2.json'
 with path.open('x') as stream:json.dump(row,stream,indent=2);stream.write('\n')
 print(json.dumps(dict(path=str(path),sha256=sha(path),status='passed',actual_check_count=4,seconds=row['elapsed_seconds'])))

if __name__=='__main__':main()
