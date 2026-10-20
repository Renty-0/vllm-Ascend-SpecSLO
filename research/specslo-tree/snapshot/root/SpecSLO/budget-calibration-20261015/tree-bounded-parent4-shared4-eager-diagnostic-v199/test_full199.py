"""Four necessary real HOST ABI/control tests; no model or backend imports."""
import ast,copy,datetime,json,sys,types,unittest
from pathlib import Path
from base199 import P,B,BASE,require
DETAILS={}
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
def rhythm():
 source=(NATIVE/'spec_rhythm.py').read_text().replace('from vllm_ascend.spec_decode.pearl.roofline import ProfiledRoofline','class ProfiledRoofline: pass')
 m=types.ModuleType('_v199_actual_HOST_rhythm');sys.modules[m.__name__]=m;exec(compile(source,str(NATIVE/'spec_rhythm.py')+'::HOST-import-only-roofline-stub','exec'),vars(m));return m
def actual_rollback_next_control_regression(test):
 """Execute the actor's real restore/next calls and next's strict first ledger gate.

 Tensor/model work is replaced by explicit carriers. This tests ordering only;
 the original numerical/ledger/KV oracle remains unchanged for the actual run.
 """
 actor=ast.parse((P/'actor199.py').read_text());run=next(n for n in actor.body if isinstance(n,ast.FunctionDef)and n.name=='run_case');body=next(n for n in run.body if isinstance(n,ast.Try)).body
 index=next(i for i,n in enumerate(body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='rollback_next'for t in n.targets))
 restore=body[index-1];assignment=body[index]
 test.assertIsInstance(restore,ast.Expr);test.assertEqual(ast.unparse(restore.value.func),'restore_source')
 # It is an unconditional peer-stage path, not a draft-role-local branch.
 test.assertEqual(ast.unparse(restore.value.args[-1].body.func),'stage')
 nexttree=ast.parse((P/'next199.py').read_text());wanted={'parameters','restore_source','proof'};nodes=[copy.deepcopy(n)for n in nexttree.body if isinstance(n,ast.FunctionDef)and n.name in wanted];test.assertEqual({n.name for n in nodes},wanted)
 fasttree=ast.parse((BASE/'fast_tree_draft_v5.py').read_text());suffix=copy.deepcopy(next(n for n in fasttree.body if isinstance(n,ast.FunctionDef)and n.name=='missing_committed_suffix'))
 fast=types.ModuleType('fast_tree_draft_v5');exec(compile(ast.fix_missing_locations(ast.Module(body=[suffix],type_ignores=[])),'actual-missing-suffix-HOST','exec'),vars(fast))
 torch=types.ModuleType('torch');previous={name:sys.modules.get(name)for name in('torch','fast_tree_draft_v5')};sys.modules['torch']=torch;sys.modules['fast_tree_draft_v5']=fast
 prefix={i:dict(safe_before=128,tokens={})for i in range(64)};whole={i:dict(safe_before=128,tokens={128:128,129:1000+i,130:2000+i,131:3000+i})for i in range(64)}
 mixed={'full8':'unchanged-actual-source-carrier'};events=[]
 class ReachedNext(RuntimeError):pass
 def rebuild(e,contexts,repairs,stage,*,role):
  test.assertEqual(role,'draft');e.ledger=copy.deepcopy(prefix);events.append('fresh-prefix-reset')
 def fixture(e,case,changed):events.append('actual-parent4');return {'actual_fixture':True}
 def invoke(e,f):test.assertIs(f['actual_fixture'],True);e.ledger=copy.deepcopy(whole);events.append('actual-shared4');return copy.deepcopy(mixed)
 def wave(*args,**kwargs):events.append('actual-next3-boundary');return [],{'unchanged':'carrier'}
 epoch=types.SimpleNamespace(rebuild=rebuild,ledger=lambda e:copy.deepcopy(e.ledger),protect=lambda e,lengths:['protected-prefix']*56)
 draft=types.SimpleNamespace(fixture=fixture,invoke=invoke,cpu_output=lambda value:value)
 values=dict(require=require,epoch=epoch,draft=draft,waves=wave)
 exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual-next-source-HOST','exec'),values)
 e=types.SimpleNamespace(is_draft=True,draft_vocab_size=151936,ledger=copy.deepcopy(prefix));case=types.SimpleNamespace(repair_vector=(0,)*64);trace=dict(fixture={'contexts':[{}]*64},mixed=mixed)
 logical={i:types.SimpleNamespace(token_ids=list(range(129)))for i in range(64)}
 def stage(label,fn):
  events.append(label)
  if label.startswith('rollback-own-next3/warm1/'):raise ReachedNext('first exact next ledger gate and next3 boundary passed')
  return fn()
 def nextproof(*args,**kwargs):return values['proof'](*args,**kwargs)
 namespace=dict(e=e,c=case,trace=trace,rollback={'logical':logical},stage=stage,nextproof=nextproof,restore_source=values['restore_source'])
 def unit(rows):return compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(rows),type_ignores=[])),'actual-actor-rollback-next-HOST','exec')
 try:
  with test.assertRaisesRegex(RuntimeError,'fresh source ledger restored before changed nextroot'):exec(unit([assignment]),namespace)
  test.assertNotIn('actual-next3-boundary',events)
  e.ledger=copy.deepcopy(prefix);events.clear()
  with test.assertRaisesRegex(ReachedNext,'first exact next ledger gate'):exec(unit([restore,assignment]),namespace)
  test.assertEqual(events.count('actual-next3-boundary'),1)
  test.assertLess(events.index('rollback-actual-source/actual-whole-parent3-shared4-source-restore'),events.index('rollback-own-next3/actual-invalidation-or-promotion-repair-inputs'))
  test.assertIn('fresh source ledger restored before changed nextroot',ast.unparse(next(n for n in nodes if n.name=='proof')))
  DETAILS['rollback_source_order']=dict(actual_actor_two_statements=True,actual_next_restore_and_strict_initial_ledger_gate=True,original_last_only_path_rejected=True,actual_parent_shared4_restore_precedes_snapshot=True,first_next3_boundary_reached=True,common_unconditional_stage=True,ledger_gate_relaxed=False,numerical_model_work_executed=False)
 finally:
  for name,old in previous.items():
   if old is None:sys.modules.pop(name,None)
   else:sys.modules[name]=old

class Tests(unittest.TestCase):
 def test_real_private_typed_E4_and_context_cleanup(self):
  import budget199,split_budget_v132,scope199,live199
  original=split_budget_v132.DraftWorkLimit;self.assertEqual(original().eager_per_request,3)
  ns=budget199.classes();rows=[]
  for c in scope199.cases():
   w=budget199.work_plan(c);self.assertEqual(w.work_limit.eager_per_request,4);self.assertEqual(w.allocated_selected_candidates,64+4*c.eager);self.assertEqual(w.target_limit.queries,128);self.assertEqual(scope199.bucket(c.count),20)
   rows.append(dict(E=c.eager,R=c.repair_count,F=scope199.bucket(c.eager),U=c.union_M,M=scope199.bucket(c.count),selected=w.allocated_selected_candidates))
  with self.assertRaises(ValueError):original(eager_per_request=4)
  self.assertIs(split_budget_v132.DraftWorkLimit,original)
  e=types.SimpleNamespace(_v173_qualified_component={})
  with self.assertRaisesRegex(RuntimeError,'intentional'):
   with live199.diagnostic(e,scope199.cases()[0]):
    self.assertTrue(hasattr(e,'_v199_diagnostic'));raise RuntimeError('intentional')
  self.assertFalse(hasattr(e,'_v199_diagnostic'))
  self.assertIsNone(live199.validate(e,None,[],[],{},[],[],4))
  with self.assertRaises(RuntimeError):live199.validate(e,None,[],[],{16:(object(),[1,2,3,4])},[],[],4)
  DETAILS['private_typed']=dict(actual_classes=True,original_cap3_identity_unchanged=True,five_actual_case_rows=rows,empty_source_preserves_original_startup=True,nonempty_without_permission_rejected=True,finally_revoked=True)

 def test_actual_native_prefix_allocator_and_device_index_producer(self):
  import epoch199
  ns={};path=NATIVE/'native_cache.py';exec(compile(path.read_text(),str(path),'exec'),ns);pages=[]
  for draft in(True,False):
   cache=ns['NativePrefixCache'](544,32,128)
   class Engine:
    is_draft=draft;_cache_storage_sequence_capacity=527;_v151_source_epoch=1;_tree_graph_scratch_sequence_ids=tuple(range(463,527));config=types.SimpleNamespace(kvcache_block_size=128)
    def _allocate_cache(self,prompts,**kw):self.cache_allocation=cache.allocate(prompts,False,sequence_capacity=527);self.cache_block_tables=self.cache_allocation.block_tables
    def _activate_cache_sequence(self,i,p,**kw):cache.activate_sequence(i,p,enable_prefix_caching=False)
    def _ensure_cache_capacity(self,ids,pos):cache.ensure_capacity(ids,pos)
   e=Engine();e._allocate_cache([[i+1]*128 for i in range(64)])
   for owner in epoch199.scratch_ids(e):e._activate_cache_sequence(owner,[0])
   e._ensure_cache_capacity(list(range(64)),[160]*64)
   for owner in epoch199.scratch_ids(e):e._ensure_cache_capacity([owner],[32])
   value=epoch199.lease(e);self.assertEqual(value['actual_unique_pages'],166 if draft else 192);self.assertEqual(len(value['actual_block_tables']),102 if draft else 128)
   self.assertTrue(all(e.cache_allocation.block_tables[i][0]>=0 and e.cache_allocation.block_tables[i][1]>=0 for i in range(64)))
   pages.append(dict(role='draft'if draft else'target',pages=value['actual_unique_pages'],scratch=value['private_scratch']))
  # Execute the actual independent PA permutation with separate CPU/NPU carriers;
  # an index allocated on engine.device must fail against CPU context_lens.
  import draft199
  from scope199 import cases
  for case in cases():
   rows=draft199.contexts(case);self.assertEqual(len(rows),64);self.assertEqual(len(draft199.counts(case)),64)
   self.assertEqual([len(v['token_ids'])for v in rows],[129+r for r in case.repair_vector]);self.assertEqual(draft199.counts(case)[32:],[0]*32)
  class Tensor:
   def __init__(self,values,device):self.values=list(values);self.device=device
   def index_select(self,dim,index):
    if index.device!=self.device:raise RuntimeError('wrong index device')
    return Tensor([self.values[i]for i in index.values],self.device)
  fake=types.SimpleNamespace(long=object(),tensor=lambda v,**kw:Tensor(v,kw['device']))
  from dataclasses import make_dataclass
  Meta=make_dataclass('Meta',[(x,object)for x in('slot_mapping','context_lens','block_tables','actual_seq_lengths_q','sequence_lens')])
  def pa(e,ids,positions,tokens,physical):
   count=len(ids);return Tensor(tokens+[0]*(physical-count),'npu'),Tensor(positions+[0]*(physical-count),'npu'),Meta(Tensor(list(range(count))+[-1]*(physical-count),'npu'),Tensor([p+1 for p in positions]+[1]*(physical-count),'cpu'),Tensor(list(range(physical)),'npu'),tuple(range(1,count+1)),tuple([p+1 for p in positions]+[1]*(physical-count)))
  source=ast.parse((P/'draft199.py').read_text());fn=next(n for n in source.body if isinstance(n,ast.FunctionDef)and n.name=='union_call');unit=ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[]));own=dict(vars(draft199),PA=pa)
  old=sys.modules.get('torch');sys.modules['torch']=fake
  try:
   exec(compile(unit,'actual-union-call-device-HOST','exec'),own)
   from scope199 import Case
   c=Case(3,1);x,p,m=own['union_call'](types.SimpleNamespace(),c,{'catchups':[[9]]},[[1,2,3,4]]*16)
   self.assertEqual(m.context_lens.device,'cpu');self.assertEqual(m.context_lens.values,[133,133,133,1,129,1,1,1]);self.assertEqual(m.actual_seq_lengths_q,tuple(range(1,6)))
   with self.assertRaises(RuntimeError):Tensor([1],'cpu').index_select(0,Tensor([0],'npu'))
  finally:
   if old is None:sys.modules.pop('torch',None)
   else:sys.modules['torch']=old
  DETAILS['allocator_metadata']=dict(actual_native_cache=True,roles=pages,actual_union_call_device_sensitive=True,old_CPU_context_NPU_index_rejected=True)

 def test_actual_controller_promote_original_planner_and_capture_binding(self):
  r=rhythm();states={i:r.SpecRhythmRuntimeState(i,0 if i<16 else 1,max_gamma=4)for i in range(32)};ctl=r.SpecRhythmPipelineController(states)
  normal=[]
  for i in range(16):
   t=ctl.new_ticket(i,gamma=4,eager=False);ctl.publish([t]);normal.append(t)
  parent=ctl.new_ticket(16,gamma=4,eager=False);ctl.publish([parent]);child=ctl.new_ticket(16,gamma=4,eager=True);ctl.publish([child])
  promoted=ctl.finish_verification(16,fully_accepted=True,proposed_tokens=4,accepted_tokens=4,delivered_tokens=5,draft_confidence=.8,ema_alpha=.1);self.assertIs(promoted,child);ctl.finish_cycle(1)
  plan=ctl.build_plan(list(range(32)),merge_ready_homes=True,max_target_requests=16,verification_budget=96,ready_candidate_counts={i:t.gamma for i,t in ctl.ready.items()});self.assertEqual(len(plan.target_request_indices),16);self.assertTrue(all(ctl.ready[i]is normal[i]for i in plan.target_request_indices))
  for i in plan.target_request_indices:ctl.finish_verification(i,fully_accepted=False,proposed_tokens=4,accepted_tokens=1,delivered_tokens=2,draft_confidence=.5,ema_alpha=.1)
  ctl.finish_cycle(plan.target_home_batch_id);tail=ctl.build_plan(list(range(32)),merge_ready_homes=True,max_target_requests=16,verification_budget=96,ready_candidate_counts={i:t.gamma for i,t in ctl.ready.items()});self.assertEqual(list(tail.target_request_indices),[16]);self.assertIs(ctl.validate_verification(16),child)
  # Current scheme honestly records no joint16 child in this default ordering.
  runtime=ast.parse((P/'runtime199.py').read_text());worker=next(n for n in runtime.body if isinstance(n,ast.FunctionDef)and n.name=='worker');calls=[(n.lineno,ast.unparse(n.func))for n in ast.walk(worker)if isinstance(n,ast.Call)]
  capture=[n for n,f in calls if f=='capture_original_protocol_before_startup'];install=[n for n,f in calls if f=="entry['install']"];ctor=[n for n,f in calls if f=='native.NativePearlEngine'];self.assertEqual(len(capture),len(ctor));self.assertEqual(len(capture),1);self.assertLess(capture[0],install[0]);self.assertLess(install[0],ctor[0]);self.assertNotIn('compile_protocol(', (P/'protocol199.py').read_text())
  self.assertIn('native_protocol.commit_states is e._v199_commit_states',(P/'protocol199.py').read_text());self.assertIn('from extension199 import loaded_origins as assert_origins',(P/'component199.py').read_text())
  # Actual original Native decoder DEFINITION and strict capture are executed
  # with decorator/import symbols only; no decoder, model, Tensor or collective.
  import inspect,importlib.util,builtins
  torch=types.ModuleType('torch');torch.__path__=[];torch.inference_mode=lambda:lambda fn:fn
  dist=types.ModuleType('torch.distributed');torch.distributed=dist
  oldtorch=sys.modules.get('torch');olddist=sys.modules.get('torch.distributed');sys.modules['torch']=torch;sys.modules['torch.distributed']=dist
  try:
   tree=ast.parse((NATIVE/'native_engine.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='NativePearlEngine');fn=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='_generate_spec_rhythm_tree_decode'))
   ns=dict(torch=torch,__file__=str(NATIVE/'native_engine.py'));future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
   exec(compile(ast.fix_missing_locations(ast.Module(body=[future,fn],type_ignores=[])),str(NATIVE/'native_engine.py'),'exec'),ns)
   native=types.SimpleNamespace(NativePearlEngine=type('OriginalSourceCarrier',(),{'_generate_spec_rhythm_tree_decode':ns['_generate_spec_rhythm_tree_decode']}))
   from original_tree_cycle_v75 import capture_original_protocol_before_startup,_require_compiled,CycleContractError
   captured=capture_original_protocol_before_startup(native);_require_compiled(captured)
   # Execute the NEW worker's exact explicit original commit_states import and assignment.
   binding=[]
   for item in ast.walk(worker):
    if isinstance(item,ast.Import)and any(a.name=='original_native_protocol_v75'for a in item.names):binding.append(item)
    if isinstance(item,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='commit_states'for t in item.targets):binding.append(item)
   binding.sort(key=lambda n:n.lineno);unit=ast.fix_missing_locations(ast.Module(body=binding,type_ignores=[]));values={};exec(compile(unit,'actual-runtime-commit-binding','exec'),values)
   import original_native_protocol_v75
   self.assertIs(values['commit_states'],original_native_protocol_v75.commit_states)
   oldimport=builtins.__import__
   def missing(name,*args,**kwargs):
    if name=='original_native_protocol_v75':raise ModuleNotFoundError('missing explicit original protocol provider')
    return oldimport(name,*args,**kwargs)
   builtins.__import__=missing
   try:
    with self.assertRaises(ModuleNotFoundError):exec(compile(unit,'actual-runtime-missing-provider','exec'),{})
   finally:builtins.__import__=oldimport
   original_entry=ast.parse((BASE/'tree_entry.py').read_text());fresh=copy.deepcopy(next(n for n in ast.walk(original_entry)if isinstance(n,ast.FunctionDef)and n.name=='fresh_decode'));freshns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[fresh],type_ignores=[])),str(BASE/'tree_entry.py'),'exec'),freshns);native.NativePearlEngine._generate_spec_rhythm_tree_decode=freshns['fresh_decode']
   with self.assertRaises(CycleContractError):capture_original_protocol_before_startup(native)
   _require_compiled(captured)
   bad=dict(captured,original_decode_sha256='wrong')
   with self.assertRaises(CycleContractError):_require_compiled(bad)
  finally:
   if oldtorch is None:sys.modules.pop('torch',None)
   else:sys.modules['torch']=oldtorch
   if olddist is None:sys.modules.pop('torch.distributed',None)
   else:sys.modules['torch.distributed']=olddist
  DETAILS['real_controller']=dict(actual_promotion_identity=True,wholeproposal_original_planner=True,normal_first16_then_child_tail=True,never_force_M80=True,original_preinstall_capture_once=True,actual_capture_remains_valid_after_wrapper=True,missing_provider_and_late_capture_rejected=True,commit_states_explicit_identity=True)
  actual_rollback_next_control_regression(self)

 def test_actual_guard_receipt_projection_and_no_production(self):
  G=B/'budget-diagnostics/v199-root-execution';tree=ast.parse((G/'guard199.py').read_text());scope=next(n for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SCOPE'for t in n.targets));ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[scope],type_ignores=[])),'actual-guard-scope','exec'),ns)
  original=B/'budget-diagnostics/v154-root-execution/supervise_v154.py';execute=next(n for n in ast.parse(original.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='execute_locked');row=next(n for n in ast.walk(execute)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='row'for t in n.targets))
  ns.update(datetime=datetime,tag='HOST199',ACTION='bounded-parent4-child4-shared4-diagnostic',readiness={},LOCK=B/'.ROOT_NPU_EXECUTION.lock',MAX_SECONDS=7200,MAX_TAG_BYTES=256*1024**2,MAX_AGGREGATE_BYTES=512*1024**2)
  exec(compile(ast.fix_missing_locations(ast.Module(body=[row],type_ignores=[])),'actual-original-row-HOST','exec'),ns);self.assertIs(ns['row']['production_permission'],False);self.assertIs(ns['row']['full463'],False)
  for name in('runtime199.py','scope199.py','actor199.py'):self.assertNotIn('activate_after_component(', (P/name).read_text())
  from live199 import fast_source
  source=fast_source();self.assertEqual(source,(P/'fast199.generated.py').read_text());before=(B/'tree-live-shared4-union-v159/fast159.generated.py').read_text();self.assertEqual(source.replace('        from live199 import fused_prelude,validate\n        validate(engine,native,plans,ids,sources,repairs,frontiers,steps)\n','        from live159 import fused_prelude\n'),before)
  self.assertIn('safe_storage177.py',(G/'guard199.py').read_text());self.assertIn("f['ROOT_source_reviewed'] is True",(G/'guard199.py').read_text())
  import target199,protocol199,next199,runtime199,extension199,audit199
  import component199
  captured=[];original=component199.functions
  def real_factory(path,namespace,**kwargs):
   if path==BASE/'actor_v146.py':
    text=path.read_text()
    for old,new,count in kwargs['changes']:
     self.assertEqual(text.count(old),count);text=text.replace(old,new)
    reverse=text
    for old,new,count in reversed(kwargs['changes']):reverse=reverse.replace(new,old)
    self.assertEqual(reverse,path.read_text());self.assertIn('from extension199 import loaded_origins as assert_origins',text)
    captured.append(kwargs['changes']);raise RuntimeError('HOST-before-any-model-actor')
   return original(path,namespace,**kwargs)
  component199.functions=real_factory
  try:
   with self.assertRaisesRegex(RuntimeError,'HOST-before-any-model-actor'):component199.baseline(None,None,None,None,None)
  finally:component199.functions=original
  self.assertEqual(len(captured),1)
  DETAILS['guard_source']=dict(actual_original_row=True,frozen177_storage=True,ROOTfreeze_required=True,exact_whole_fast_inverse=True,no_production_issuer=True,no_full463=True)
