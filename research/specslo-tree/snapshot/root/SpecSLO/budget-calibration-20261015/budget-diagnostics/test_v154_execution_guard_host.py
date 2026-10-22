"""Four small guard HOST checks; no helper/verifier/device execution."""
import ast,copy,hashlib,importlib.util,json,subprocess,sys,time,unittest
from pathlib import Path
D=Path(__file__).resolve().parent;G=D/'v154-root-execution'
sys.path.insert(0,str(G))
import common_guard as g
def load(name):
 spec=importlib.util.spec_from_file_location('_HOST154_'+name,G/(name+'.py'))
 value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value
supervisor=load('supervise_v154');prepare=load('prepare_v154')
def assignment(path,function,name):
 tree=ast.parse(path.read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
 return copy.deepcopy(next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
def execute_row(node,namespace):
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'V154-actual-row-HOST','exec'),namespace)
 return namespace[node.targets[0].id]

class Checks(unittest.TestCase):
 def test_actual_retained_source_and_launcher(self):
  state=g.source_state();guards=g.guard_inputs()
  self.assertEqual((len(state['local']),len(state['external'])),(17,195))
  self.assertEqual(len(state['evidence']),5)
  self.assertTrue(all(g.sha(p)==h for p,h in guards.items()))
  for path in G.glob('*.py'):ast.parse(path.read_text())
  subprocess.run(['/bin/bash','-n',str(G/'launch_v154.sh')],check=True)
  argv=supervisor.launch_argv('HOST-not-executed')
  self.assertEqual(argv[:2],['/usr/bin/env','-i'])
  self.assertIn(str(G/'launch_v154.sh'),argv)
  self.assertFalse(g.FREEZE.exists());self.assertFalse((g.P/'actual').exists())
 def test_original_private_helper_bodies_and_bindings(self):
  helpers=g.original_helpers();tree=ast.parse((g.PARENT/'supervise_v147.py').read_text())
  for name in ('save','helper','verify'):
   node=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name));ns={}
   exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'HOST154-unchanged-original','exec'),ns)
   self.assertEqual(helpers[name].__code__.co_code,ns[name].__code__.co_code)
   self.assertEqual(helpers[name].__code__.co_names,ns[name].__code__.co_names)
   self.assertEqual(helpers[name].__globals__['P'],g.P)
  self.assertEqual(set(helpers),{'save','helper','verify','owned_cleanup'})
 def test_execute_actual_receipt_and_freeze_dict_AST(self):
  node=assignment(G/'supervise_v154.py','execute_locked','row')
  ns=dict(vars(supervisor),tag='HOST-row-only',readiness={'HOST':True})
  row=execute_row(node,ns)
  self.assertIs(row['cost_measured'],False);self.assertIs(row['full_qualification'],False)
  self.assertIs(row['two_complete_zero_traces'],False)
  self.assertEqual(row['limits'],dict(child_execution_seconds=1800,actual_tag_bytes=32*1024**2,aggregate_private_bytes=64*1024**2))
  bad=copy.deepcopy(node);bad.value.keywords.insert(0,ast.keyword(arg='cost_measured',value=ast.Constant(False)))
  with self.assertRaises(TypeError):execute_row(bad,dict(vars(supervisor),tag='HOST-negative-duplicate',readiness={}))
  frozen=assignment(G/'prepare_v154.py','main','frozen')
  value=execute_row(frozen,dict(vars(prepare),state=dict(local={},external={},evidence={}),guards={},root={},peers={}))
  self.assertIs(value['ROOT_source_reviewed'],True);self.assertIs(value['actual_device_qualification'],False)
  self.assertIs(value['cost_measured'],False);self.assertIs(value['two_complete_zero_traces'],False)
 def test_real_completed_schema_and84_counter_subtractions(self):
  audit=dict(status='passed-component-evidence',positive_count=4,negative_count=28,hidden_KV_recomputed_by_HOST=False,
   two_complete_zeros_qualified=False,full_qualification=False,production_gates_opened=False,formal_Goodput=False,adopted=False)
  completed=dict(action=g.ACTION,replies=[['calibrated',i,dict(cases=1,complete_zeros=0,timings=0)] for i in range(4)],audit=audit)
  exited=dict(closed=True,all_exited=True)
  self.assertEqual(supervisor.completed_schema(completed,exited,audit),'passed_component')
  negative=dict(audit,status='passed-negative-subset',positive_count=0,negative_count=32)
  self.assertEqual(supervisor.completed_schema(dict(completed,audit=negative),exited,negative),'discovery_subset')
  bad=dict(audit,formal_Goodput=True)
  with self.assertRaises(RuntimeError):supervisor.completed_schema(dict(completed,audit=bad),exited,bad)
  counters=json.loads((g.P/'CPU_RESULT154.json').read_text())['input_hashes']
  # Read the exact21 source tuple via AST literals, without importing candidates.
  scope=Path(json.loads((g.P/'INPUTS154.json').read_text())['P152'])/'scope152.py'
  ns={};tree=ast.parse(scope.read_text())
  node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NATIVE_COUNTERS' for t in n.targets))
  exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'HOST154-original21','exec'),ns)
  keys=ns['NATIVE_COUNTERS'];values={k:0 for k in keys}
  reports=[dict(result=dict(Native_before=values.copy(),Native_after=values.copy(),Native_delta=values.copy())) for i in range(4)]
  value=supervisor.native_discovery_evidence(reports,keys)
  self.assertEqual(value['raw_deltas_recomputed'],84);self.assertIs(value['zero_discovery_qualified'],False)
  reports[0]['result']['Native_delta'][keys[0]]=1
  with self.assertRaises(RuntimeError):supervisor.native_discovery_evidence(reports,keys)

def main():
 before={str(p):g.sha(p) for p in G.iterdir() if p.is_file()};start=time.perf_counter()
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 after={str(p):g.sha(p) for p in G.iterdir() if p.is_file()}
 row=dict(schema='V154.execution-guard-HOST.v1',status='passed' if result.wasSuccessful() else 'failed',
  tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.perf_counter()-start,
  guard_before=before,guard_after=after,guard_source_unchanged=before==after,
  test_source=dict(path=str(Path(__file__)),sha256=g.sha(__file__)),
  actual_receipt_row_AST_executed=True,actual_freeze_row_AST_executed=True,old_duplicate_keyword_negative_TypeError_tested=True,
  core_suite_repeated=False,helper_or_verifier_executed=False,Torch_imported='torch' in sys.modules,NPU_used=False,
  source_frozen=g.FREEZE.exists(),physical_cards_queried=False,constructor_or_model_executed=False)
 assert before==after and 'torch' not in sys.modules and not g.FREEZE.exists()
 path=D/('V154_EXECUTION_GUARD_HOST_CHECK.json' if result.wasSuccessful() else 'V154_EXECUTION_GUARD_HOST_FAILED_'+str(time.time_ns())+'.json')
 with path.open('x') as f:json.dump(row,f,indent=2,sort_keys=True);f.write('\n')
 print(json.dumps(dict(path=str(path),sha256=g.sha(path),status=row['status'],tests=result.testsRun,elapsed=row['elapsed_seconds'])))
 return int(not result.wasSuccessful())
if __name__=='__main__':raise SystemExit(main())
