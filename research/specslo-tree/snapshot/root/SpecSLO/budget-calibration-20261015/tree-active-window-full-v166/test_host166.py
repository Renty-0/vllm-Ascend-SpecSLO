"""Five new source/orchestration checks; no inherited suite or real backend."""
import ast,copy,functools,importlib.util,json,sys,tempfile,types,typing,unittest
from pathlib import Path
from unittest.mock import patch
from base166 import P,BASE,B,require,checked,inherited_module

def cell(value):return (lambda:value).__closure__[0]
def assignment(path,function,name):
 fn=next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==function)
 return copy.deepcopy(next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)))
def execute(node,ns):
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'V166-HOST-actual-constructor','exec'),ns)
 return ns[node.targets[0].id]

class Host(unittest.TestCase):
 def test_private_constructor_and_all_inventory_IO_source_inverses(self):
  from source166 import entry_namespace,ENTRY_CHANGES,startup_auditor
  import progress166
  with tempfile.TemporaryDirectory(prefix='HOST166-IO-',dir=P) as temp:
   output=Path(temp)
   original=inherited_module('tree_entry');old_install=original.install
   namespace=entry_namespace(output)
   self.assertEqual(namespace['P'],BASE);self.assertEqual(namespace['OUTPUT'],output)
   self.assertIs(original.install,old_install)
   text=(BASE/'tree_entry.py').read_text();modified=text
   for old,new,count in ENTRY_CHANGES:self.assertEqual(modified.count(old),count);modified=modified.replace(old,new)
   for old,new,count in reversed(ENTRY_CHANGES):modified=modified.replace(new,old)
   self.assertEqual(modified,text)
   with patch.object(progress166,'_OUTPUT',None):
    evidence=progress166.configure(output)
    for name in ('qualify_prefill','qualify_target','qualify_draft','preflight_MC2','memory_checkpoint'):
     function=getattr(progress166,name);self.assertEqual(function.__globals__['OUTPUT'],output)
    self.assertTrue(evidence['five_inventory_bodies_preserved'])
   audit=startup_auditor(output);self.assertEqual(audit.__globals__['OUTPUT'],output)
   self.assertIn('progress166',audit.__code__.co_names)
   # Execute each real adapted destination expression, without its model body.
   expressions=["OUTPUT/'runtime_tree_decode.py'","OUTPUT/'generation-progress'","OUTPUT/'startup-progress'"]
   self.assertTrue(all(eval(v,dict(OUTPUT=output)).is_relative_to(output) for v in expressions))
   self.assertFalse((output/'runtime_tree_decode.py').exists())

 def test_actual_new_shell_and_live_installer_source_assertions(self):
  from source166 import entry_namespace,original_observation
  from trace166 import make_trace
  from types import FunctionType
  original=inherited_module('poisson_coverage_probe')
  with tempfile.TemporaryDirectory(prefix='HOST166-binding-',dir=P) as temp:
   output=Path(temp);entry=entry_namespace(output)
   deco={};exec(compile('def factory(func):\n def decorate_context(*a,**k):\n  return func(*a,**k)\n return decorate_context\n','/HOST/torch/utils/_contextlib.py','exec'),deco)
   inference=lambda:lambda f:functools.update_wrapper(deco['factory'](f),f)
   native=types.ModuleType('HOST166_native');native.__dict__.update(torch=types.SimpleNamespace(inference_mode=inference),
    PearlPipelineState=object,NativeSamplingParams=object,Sequence=typing.Sequence,Any=typing.Any)
   runtime=(BASE/'runtime_tree_decode.py').read_text();runtime_path=output/'runtime_tree_decode.py';runtime_path.write_text(runtime)
   exec(compile(runtime,str(runtime_path),'exec'),vars(native));delegate=native._generate_spec_rhythm_tree_decode
   fresh=original._find_code(entry['install'].__code__,'fresh_decode')
   values=dict(decode_impl=delegate,policy=None)
   shell=FunctionType(fresh,entry,'fresh_decode',closure=tuple(cell(values.get(n)) for n in fresh.co_freevars))
   cls=type('NativePearlEngine',(),dict(_generate_spec_rhythm_tree_decode=shell));native.NativePearlEngine=cls
   proxy=original_observation(entry,output);proxy._assert_original_decode(cls()._generate_spec_rhythm_tree_decode)
   trace,state=make_trace(24,entry=entry,output=output)
   install=trace.__globals__['install'];cells=dict(zip(install.__code__.co_freevars,install.__closure__))
   # Original permit really rejects this HOST call; only the fresh namespace is
   # then given an explicit stub. This is not a device launch authorization.
   with self.assertRaises(Exception):install(native,lambda e:None)
   cells['namespace'].cell_contents['require_bounded_scope']=lambda:None
   install(native,lambda e:None)
   trace.__globals__['_assert_live_decode'](cls()._generate_spec_rhythm_tree_decode)
   self.assertIn('initial_batch_size = min(initial_batch_size, 24)',state['source'])
   self.assertIs(trace.__globals__['standalone_report'],__import__('scope166').report)
   self.assertEqual(trace.__globals__['original']._expected_code(BASE/'tree_entry.py','fresh_decode'),proxy._expected_code(BASE/'tree_entry.py','fresh_decode'))

 def test_standalone_issuer_cannot_be_old_S35_or_stale_engine(self):
  import scope166
  proof=dict(audit=dict(component_numerical_qualified=True))
  engine=types.SimpleNamespace(_v166_qualified_component=proof,_v166_actual_startup=dict(actual_native_entry_counts=dict(entries=0,draft_entries=60,target_entries=54)),
   is_draft=True,device=types.SimpleNamespace(type='npu'),
   cache_allocation=None,cache_block_tables=None,graph_runner=types.SimpleNamespace(graph_cache_sealed=False))
  with self.assertRaises(RuntimeError):scope166.issue(engine)
  scope=scope166.issue(engine,component_proof=proof)
  self.assertEqual(scope.require(engine),frozenset());self.assertEqual(scope166.report(scope,engine)['N'],[])
  with self.assertRaises(RuntimeError):scope.require(copy.copy(engine))
  engine._v166_qualified_component=copy.deepcopy(proof)
  with self.assertRaises(RuntimeError):scope.require(engine)
  worker=next(n for n in ast.parse((P/'runtime166.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='worker')
  source=ast.unparse(worker)
  self.assertNotIn('qualify_structures',source);self.assertIn('prepare_original_draft_family(engine)',source)
  self.assertIn('extension.qualify_component(',source);self.assertIn('warm_to_fixedpoint(',source)
  self.assertIn('extension.loaded_origins()',source)

 def test_real_fixedpoint_loop_role_allrank_convergence_and_bound(self):
  import trace166
  original=inherited_module('startup_receipts_v75')
  node=copy.deepcopy(next(n for n in ast.parse((P/'trace166.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='warm_to_fixedpoint'))
  node.body=[n for n in node.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
  calls=[];packet_count=[];attempts={}
  engine=types.SimpleNamespace(rank=0,topology=types.SimpleNamespace(draft_leader_rank=0),
   groups=types.SimpleNamespace(verification_coordination_group=object()),_dynamic_tree_generation=0)
  counters=original.NATIVE_COUNTERS
  def maker(cap,**kw):
   def trace(e,*,count,role,directory,qualified_s35_scope):
    calls.append((cap,count,role,str(directory)));e._dynamic_tree_generation+=1
    attempt=attempts.get(cap,0);attempts[cap]=attempt+1
    delta={k:int(i==0 and attempt==0) for i,k in enumerate(counters)}
    return dict(native=dict(before={k:0 for k in counters},after=delta,delta=delta),source_generation=e._dynamic_tree_generation)
   return trace,{}
  def gather(rows,packet,group):
   rows[:]=[dict(packet,rank=i) for i in range(4)];packet_count.append(len(packet['native']['delta']))
  ns=dict(vars(trace166),make_trace=maker,split_snapshot=lambda e,**kw:dict(next=e._dynamic_tree_generation+1),
          dist=types.SimpleNamespace(all_gather_object=gather))
  exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'HOST166-actual-fixedpoint-body','exec'),ns)
  with tempfile.TemporaryDirectory(prefix='HOST166-warm-',dir=P) as temp:
   result=ns['warm_to_fixedpoint'](engine,scope=object(),stage=lambda n,f:f(),output=temp,entry={},admission_windows=(32,24,16))
  self.assertTrue(result['all_windows_converged']);self.assertEqual([c[:3] for c in calls],[(w,65,'warmup65') for w in (32,32,24,24,16,16)])
  self.assertEqual(packet_count,[21]*6);self.assertEqual(len({c[3] for c in calls}),6)

 def test_guard_actual_receipt_freeze_constructors_and_owned_namespace(self):
  gpath=B/'budget-diagnostics/v166-root-execution/guard166.py'
  spec=importlib.util.spec_from_file_location('_HOST166_guard',gpath);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
  source=B/'budget-diagnostics/v154-root-execution/supervise_v154.py'
  row=assignment(source,'execute_locked','row');ns=dict(vars(g),tag='HOST-never-launched',readiness={})
  result=execute(row,ns);self.assertEqual(result['limits']['child_execution_seconds'],7200)
  self.assertEqual(result['limits']['actual_tag_bytes'],256*1024**2);self.assertFalse(result['cost_measured'])
  bad=copy.deepcopy(row);bad.value.keywords.insert(0,ast.keyword(arg='cost_measured',value=ast.Constant(False)))
  with self.assertRaises(TypeError):execute(bad,dict(vars(g),tag='HOST-bad',readiness={}))
  f=assignment(gpath,'prepare_only','f')
  state=dict(local={},external={},evidence={},parent_archive=dict(HOST=True))
  frozen=execute(f,dict(vars(g),state=state,root={},peers={},sha=lambda p:'0'*64))
  self.assertTrue(frozen['ROOT_source_reviewed']);self.assertEqual(frozen['dependency_archives_verified_in_place'],dict(HOST=True))
  helpers=g.original_helpers();self.assertEqual(helpers['helper'].__globals__['P'],P)
  self.assertEqual(helpers['verify'].__globals__['P'],P);self.assertEqual(g.execute_locked.__globals__['P'],P)
  self.assertEqual(g.main.__globals__['MAX_AGGREGATE_BYTES'],512*1024**2)
  self.assertEqual(g.launch_argv('HOST-never-launched')[:2],['/usr/bin/env','-i'])
  self.assertIn('LOCK_EX',ast.unparse(ast.parse((source).read_text())))

if __name__=='__main__':unittest.main()
