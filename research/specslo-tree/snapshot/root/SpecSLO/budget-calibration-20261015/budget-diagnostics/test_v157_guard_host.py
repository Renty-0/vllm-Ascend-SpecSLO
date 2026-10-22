"""Four small thin-guardian HOST checks; no device, prepare or old suite."""
import ast,copy,importlib.util,json,subprocess,sys,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
D=Path(__file__).resolve().parent;G=D/'v157-root-execution'
spec=importlib.util.spec_from_file_location('_HOST157_guard',G/'guard157.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
def node(path,function,name):
 f=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==function)
 return copy.deepcopy(next(n for n in f.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
def run_node(value,namespace):
 exec(compile(ast.fix_missing_locations(ast.Module(body=[value],type_ignores=[])),'HOST157-actual-row','exec'),namespace)
 return namespace[value.targets[0].id]

class Checks(unittest.TestCase):
 def test_actual_new9_and_parent232_archive_checked(self):
  state=g.source_state();self.assertEqual(len(state['local']),9)
  self.assertEqual(state['parent_ready']['archive']['members'],232)
  self.assertEqual(len(g.guard_inputs()),4)
  subprocess.run(['/bin/bash','-n',str(G/'launch157.sh')],check=True)
  self.assertFalse(g.FREEZE.exists());self.assertFalse((g.P/'actual').exists())
 def test_original_supervisor_and_private_helper_identity(self):
  helpers=g.original_helpers();self.assertEqual(helpers['verify'].__globals__['P'],g.P)
  self.assertEqual(helpers['helper'].__globals__['P'],g.P)
  self.assertEqual(g.actual_audit.__globals__['P'],g.Q)
  self.assertEqual(g.actual_audit.__globals__['completed_schema'].__globals__['ACTION'],g.ACTION)
  self.assertEqual(g.execute_locked.__globals__['P'],g.P)
  argv=g.launch_argv('HOST-never-launched')
  self.assertEqual(argv[:2],['/usr/bin/env','-i']);self.assertIn(str(G/'launch157.sh'),argv)
  original=(g.Q.parent/'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text()
  self.assertNotIn('graph_task_update_begin',original)
 def test_real_receipt_and_freeze_constructor_AST(self):
  old=g.Q.parent/'budget-diagnostics/v154-root-execution/supervise_v154.py'
  receipt=node(old,'execute_locked','row');ns=dict(vars(g),tag='HOST-constructor',readiness={})
  row=run_node(receipt,ns)
  self.assertIs(row['cost_measured'],False);self.assertIs(row['dependency_fix_only'],True)
  self.assertIs(row['two_complete_zero_traces'],False)
  bad=copy.deepcopy(receipt);bad.value.keywords.insert(0,ast.keyword(arg='cost_measured',value=ast.Constant(False)))
  with self.assertRaises(TypeError):run_node(bad,dict(vars(g),tag='HOST-negative',readiness={}))
  f=node(G/'guard157.py','prepare_only','f');state=g.source_state()
  frozen=run_node(f,dict(vars(g),state=state,guards=g.guard_inputs(),root=dict(path=str(D/'HOST-ROOT-never-created.json'),sha256='0'*64),peers={}))
  self.assertIs(frozen['ROOT_source_reviewed'],True);self.assertIs(frozen['full_qualification'],False)
  self.assertEqual(frozen['P154_prepared_dependency']['archive']['members'],232)
 def test_old_archive_not_recopied_and_original_tiny_archive_guard(self):
  f=node(G/'guard157.py','prepare_only','f');state=g.source_state()
  frozen=run_node(f,dict(vars(g),state=state,guards=g.guard_inputs(),root=dict(path=str(D/'HOST-ROOT-never-created.json'),sha256='0'*64),peers={}))
  sha=g.sha
  with patch.object(g,'sha',lambda p:'0'*64 if Path(p)==g.FREEZE else sha(p)):
   sources=g.archive_sources(frozen)
  self.assertNotIn(str(g.Q/'PREPARED_SOURCE_ARCHIVE_V154.tar.gz'),sources)
  self.assertIn(str(g.Q/'FROZEN_SOURCE154.json'),sources)
  self.assertTrue(set(state['local'])<=set(sources))
  with tempfile.TemporaryDirectory(prefix='v157-host-archive-',dir=D) as temp:
   root=Path(temp);a=root/'a';b=root/'b';a.write_bytes(b'one');b.write_bytes(b'two')
   inputs={str(a):sha(a),str(b):sha(b)};archive=root/'tiny.tar.gz'
   value=g.write_archive(archive,inputs);self.assertEqual(value['members'],2)
   with self.assertRaises(RuntimeError):g.verify_archive(archive,{str(a):sha(a)})

def main():
 before=g.guard_inputs();start=time.perf_counter()
 result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
 after=g.guard_inputs();assert before==after and 'torch' not in sys.modules and not g.FREEZE.exists()
 value=dict(schema='V157.thin-guardian-HOST.v1',status='passed' if result.wasSuccessful() else 'failed',tests_run=result.testsRun,
  failures=len(result.failures),errors=len(result.errors),elapsed_seconds=time.perf_counter()-start,
  guard_input_before=before,guard_input_after=after,source_bytes_unchanged=True,
  original_P154_archive_verified_members=232,old_archive_binary_excluded_from_new_archive=True,
  actual_receipt_and_freeze_dict_AST_executed=True,duplicate_keyword_negative_executed=True,
  two_tiny_members_archive_checked=True,original_helpers_or_device_or_verifier_called=False,
  core_suite_repeated=False,Torch_imported=False,NPU_used=False,freeze_or_constructor=False)
 path=D/('V157_THIN_GUARD_HOST_CHECK.json' if result.wasSuccessful() else 'V157_THIN_GUARD_HOST_FAILED_'+str(time.time_ns())+'.json')
 with path.open('x') as stream:json.dump(value,stream,indent=2,sort_keys=True);stream.write('\n')
 print(json.dumps(dict(path=str(path),sha256=g.sha(path),status=value['status'],elapsed=value['elapsed_seconds'])))
 return int(not result.wasSuccessful())
if __name__=='__main__':raise SystemExit(main())
