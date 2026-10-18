"""Seven current executable HOST ABI/control checks; no backend or model suite."""
import ast,copy,datetime,json,sys,types,unittest
from dataclasses import dataclass
from pathlib import Path
from base205 import P,B,BASE,require,sha
NS=types.SimpleNamespace
DETAILS={}
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')

class Vector:
 def __init__(self,v):self.v=list(v)
 def detach(self):return self
 def cpu(self):return self
 def tolist(self):return list(self.v)
 def numel(self):return len(self.v)

def plan(width,depth,prefix,maximum,candidate_budget):
 b=candidate_budget
 return NS(width=width,depth=depth,prefix_len=prefix,candidate_budget=b,parent_indices=Vector([-1,*range(b-1)]),positions=Vector(range(prefix,prefix+b+1)),cache_positions=Vector(range(prefix,prefix+b+1)))

def native_predicate():
 tree=ast.parse((NATIVE/'native_model.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='_is_contiguous_linear_tree_plan');ns={}
 exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(fn)],type_ignores=[])),'actual-Native-linear-predicate-HOST','exec'),ns);return ns[fn.name]

class Tests(unittest.TestCase):
 def test_actual_controls_three_config_deltas_and_issuer(self):
  import mapper205 as m
  cls=m.controls_class();ns=cls.__mro__[1].__init__.__globals__;ns['observe_cpu_device_context']=lambda *a,**kw:dict(passed=True)
  class Policy:
   def allocate(self,*a,**kw):raise AssertionError('original gain allocator must stay intact')
  policy=Policy()
  def shape(policy):
   def call(*a,**kw):return policy
   return call
  rhythm=NS(SpecRhythmBudgetShaper=type('Shaper',(),{'shape':shape(policy)}));native=NS()
  control=cls(native=native,rhythm=rhythm,policy=policy,torch=NS(),constructor_state=NS(),is_draft=False,configuration=dict(target_cpu_context_clean=True,draft_cpu_context_clean=True))
  @dataclass(frozen=True)
  class Config:
   spec_rhythm_max_eager_tokens:int=5
   spec_rhythm_draft_token_budget:object=None
   spec_rhythm_eager_reserve_tokens:int=0
  e=NS(config=Config(),cache_allocation=None,cache_block_tables=None,is_draft=False,graph_runner=NS(target_entries={k:NS(runtime_validated=True)for k in(m.GREEDY,m.HIDDEN)},disabled_entry_keys=set(),graph_cache_sealed=False))
  old=e.config
  try:
   control.select('candidate',engine=e);self.assertEqual(control.config_snapshots['delta'],{'spec_rhythm_max_eager_tokens':(5,3),'spec_rhythm_draft_token_budget':(None,192),'spec_rhythm_eager_reserve_tokens':(0,32)})
   for n in range(9,16):self.assertEqual(control.mapper(n,5*n,'causal')[:2],(128,38))
   e._v205_parent_component={};e.cache_allocation=object()
   with m.qualification_geometry(e):
    for n in range(9,16):self.assertEqual(control.mapper(n,5*n,'causal'),(80,16,[5]*(16-n)))
   self.assertFalse(hasattr(e,'_v205_usage_geometry'));e.cache_allocation=None
   evidence=dict(audit=dict(math_independent_qualified=True));e._v205_actual_usage=evidence;m.issue(e,evidence)
   for n in range(9,16):self.assertEqual(control.mapper(n,5*n,'causal'),m.expected_geometry(n,5*n,'causal'))
   for args in ((8,48,'full9'),(8,40,'causal'),(16,80,'causal'),(9,44,'causal')):
    self.assertEqual(control.mapper(*args),m.parent_module('mapper173').dispatch(control,*args))
   evidence['audit']['math_independent_qualified']=1
   with self.assertRaises(RuntimeError):control.mapper(9,45,'causal')
   evidence['audit']['math_independent_qualified']=True
   e.graph_runner.target_entries[m.GREEDY]=NS(runtime_validated=True)
   with self.assertRaises(RuntimeError):control.mapper(9,45,'causal')
  finally:control.select('original',engine=e)
  self.assertIs(e.config,old)
  DETAILS['controls']=dict(actual_constructor_select_restore=True,three_original_config_deltas=True,baseline_new_domain_disabled=True,current_issuer_required=True,wrong_actual_entry_identity_rejected=True,seven_exact_geometries=True)

 def test_actual_native_cache_fullbank_and_real_original_pad(self):
  import bank205 as b,mapper205 as m,target205 as t,target_layout as layout
  ns={};exec(compile((NATIVE/'native_cache.py').read_text(),str(NATIVE/'native_cache.py'),'exec'),ns);Cache=ns['NativePrefixCache']
  class Engine:
   def __init__(self):self.cache=Cache(544,32,128);self.cache_allocation=None;self.is_draft=True;self.config=NS(kvcache_block_size=128,max_model_len=4096);self._tree_graph_scratch_sequence_ids=tuple(range(463,527))
   def _allocate_cache(self,prompts,**kw):self.cache_allocation=self.cache.allocate(prompts,False,sequence_capacity=527);self.cache_block_tables=object()
   def _activate_cache_sequence(self,i,tokens,**kw):self.cache.activate_sequence(i,tokens,enable_prefix_caching=False)
   def _ensure_cache_capacity(self,ids,positions):self.cache.ensure_capacity(ids,positions)
   def _run_packed_hidden(self,tokens,ids,pos,**kw):
    assert len(tokens)==len(ids)==len(pos)and len(set(ids))==1 and pos==list(range(len(tokens)));return NS(clone=lambda:NS(carrier=True))
  names=('torch','vllm_ascend.spec_decode.pearl.tree','vllm_ascend.spec_decode.pearl.native_model');prior={n:sys.modules.get(n)for n in names}
  fake=types.ModuleType('torch');fake.isfinite=lambda v:NS(all=lambda:True)
  tr=types.ModuleType(names[1]);tr.cached_cpu_tree_speculation_plan=plan
  model=types.ModuleType(names[2]);predicate=native_predicate();predicate_calls=[]
  model._is_contiguous_linear_tree_plan=lambda p:predicate_calls.append(p)or predicate(p)
  sys.modules.update(dict(zip(names,(fake,tr,model))))
  oldhash=b.tensor_hash;oldprefix=b.prefix_evidence;oldenv=layout.envelope
  b.tensor_hash=lambda v:'HOST-allocation-carrier';b.prefix_evidence=lambda e,h:dict(owners=sorted(h),no_model_KV_compared=True)
  records=[]
  try:
   native=NS(NativePearlEngine=Engine);Engine._pad_spec_rhythm_target_tree_graph=layout.pad_target;m.install_pad(native)
   for case in b.cases():
    e=Engine();bank=b.DualBank(e,case)
    for i in range(32):bank.prefill_finite(i,bank.prefill_call(i))
    self.assertEqual(bank.prefix_ready()['pages'],128);self.assertEqual(len(bank.contexts),16);self.assertEqual(bank.guard()['dummy_owners'],list(range(463,527)))
    n=case['N'];e._v205_parent_component={};e.graph_runner=NS(graph_cache_sealed=False)
    layout.envelope=lambda a,q,f:(80,16,[5]*(16-a))
    work=dict(_engine=e,source=dict(ids=list(range(n)),contexts=[dict(token_ids=v)for v in bank.contexts[:n]]),plans=[plan(2,4,len(v)-1,4096,4)for v in bank.contexts[:n]],rows=[[11,12,13,14]for _ in range(n)])
    with m.qualification_geometry(e):
     a=t.frame(case,work);r=t.frame(case,work,reference=True)
    self.assertEqual(a[3],list(range(n))+list(range(463,463+16-n)));self.assertEqual(r[3],list(range(16,16+n))+list(range(492,492+16-n)))
    self.assertEqual(sum(p.candidate_budget+1 for p in a[0]),80);self.assertEqual(len(a[0]),16)
    self.assertTrue(all(x is y for x,y in zip(a[0][:n],work['plans'],strict=True)))
    bad=list(a);bad[3]=bad[3][:-1]+[0]
    with self.assertRaises(RuntimeError):m.check_frame(e,work['plans'],a[1][:n],work['rows'],work['source']['ids'],(*bad,n))
    old=dict(case,M=128,S=38,dummy_lengths=case['oldpad'],old_control=True)
    oa=t.frame(old,work);orr=t.frame(old,work,reference=True)
    self.assertEqual(len(oa[0]),38);self.assertEqual(sum(p.candidate_budget+1 for p in orr[0]),128)
    self.assertTrue(set(oa[3][n:]).isdisjoint(orr[3][n:]));records.append(dict(N=n,real=n,M=80,S=16,dummy=len(a[3])-n,pages=bank.page_count,old_dummy=38-n))
   fullplans=[plan(2,4,127,4096,4)for _ in range(16)];count=len(predicate_calls)
   e._pad_spec_rhythm_target_tree_graph(fullplans,[99]*16,[[1,2,3,4]]*16,list(range(16)),[],[])
   self.assertEqual(len(predicate_calls)-count,16) # only the original pad predicate, no extra wrapper scan
  finally:
   b.tensor_hash=oldhash;b.prefix_evidence=oldprefix;layout.envelope=oldenv
   for n,old in prior.items():
    if old is None:sys.modules.pop(n,None)
    else:sys.modules[n]=old
  DETAILS['real_pad_bank']=dict(actual_NativePrefixCache=True,actual_original_pad_function=True,actual_Native_linear_predicate=True,all32_prefill_prefix_ready=True,full16_reference_offset16=True,seven=records,model_math_claimed=False)

 def test_actual_subset_work_commit_and_integer_BONUS(self):
  import actor205,target205,audit_usage205
  fake=types.ModuleType('torch');fake.isfinite=lambda v:NS(all=lambda:True);old=sys.modules.get('torch');sys.modules['torch']=fake
  try:
   for n in range(9,16):
    plans=[NS(candidate_budget=4,depth=4)for _ in range(16)]
    full=dict(plans=plans,rows=[[10,11,12,13]for _ in range(16)],confidences=[.5]*16,selected=[list(range(4))]*16,source=dict(ids=list(range(16)),contexts=[dict(token_ids=[i,99],output_offset=0)for i in range(16)]),draft_proof={'full16':True})
    # The real typed require_work also inspects topology only through budget.
    for p in plans:p.parent_indices=Vector([-1,0,1,2])
    w=actor205.subset(NS(),full,n,1);self.assertEqual(w['source']['ids'],list(range(n)));self.assertEqual(w['work_plan'].allocated_selected_candidates,4*n)
    history=[[i,99]for i in range(16)];bank=NS(contexts=copy.deepcopy(history),reference_contexts=copy.deepcopy(history),guard=lambda:{'actual_carrier':True})
    commit=dict(committed_tokens=[[10]for _ in range(n)],accepted_paths=[[-1]*4 for _ in range(n)],actual_logical_tokens=[v+[10]for v in history[:n]])
    t=target205.commit_reference(NS(is_draft=True),bank,w,None,commit)
    self.assertEqual(bank.contexts[n:],history[n:]);self.assertEqual(len(bank.contexts),16);self.assertTrue(t['unselected_owners_preserved'])
    rows=[[10,11,12,13]]*n;head=[10,11,12,13,777]*n;emitted,paths=audit_usage205.verdict(rows,head)
    self.assertEqual(emitted,[[10,11,12,13,777]]*n);self.assertEqual(paths,[list(range(4))]*n)
   # Execute the old comparator's actual prepare/model/drain sequence with a
   # corrupted prefix carrier; it must reject before the next reference TP.
   names=('metadata','original_call','query_KV','prefix_evidence');prior={k:getattr(target205,k)for k in names};state={'corrupted':False,'model_calls':0}
   target205.metadata=lambda *a,**k:(None,None,NS(slot_mapping=NS(long=lambda:None)))
   def broken_call(*a,**k):state['corrupted']=True;return None,None
   target205.original_call=broken_call;target205.query_KV=lambda *a:('finite-carrier',);target205.prefix_evidence=lambda *a:state['corrupted']
   def forbidden_model(*a):state['model_calls']+=1;raise AssertionError('reference TP must not run')
   try:
    with self.assertRaisesRegex(RuntimeError,'old candidate protects all16 prefixes'):
     target205.old_control(NS(is_draft=False,model=forbidden_model),dict(N=9,Q=45,oldM=128,oldS=38,oldpad=[3]*29),NS(contexts=[[1,2]]*16),{},lambda n,f:f())
    self.assertEqual(state['model_calls'],0)
   finally:
    for k,v in prior.items():setattr(target205,k,v)
  finally:
   if old is None:sys.modules.pop('torch',None)
   else:sys.modules['torch']=old
  DETAILS['actual_subset']=dict(actual_typed_work=True,unpublished_owners_not_advanced=True,retained_bank16=True,reference_offset16=True,actual_original_BONUS_integer_recomputed=True)

 def test_actual_runtime_projection_guard_row_and_fourfield_schema(self):
  import runtime205,component205,trace205,scope205,check205,extension205,audit205
  code=ast.parse((P/'runtime205.py').read_text());worker=next(n for n in code.body if isinstance(n,ast.FunctionDef)and n.name=='worker')
  self.assertEqual(sum(isinstance(n,ast.Call)and ast.unparse(n.func)=='native.NativePearlEngine'for n in ast.walk(worker)),1)
  self.assertEqual(sum(isinstance(n,ast.Call)and ast.unparse(n.func)=='capture_original_protocol_before_startup'for n in ast.walk(worker)),1)
  self.assertIs(trace205.metrics.__globals__['sha'],sha);self.assertIn('prefill205',(P/'component205.py').read_text());self.assertIn('actor205',(P/'component205.py').read_text())
  tree=ast.parse((NATIVE/'native_graph.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='NativeGraphExecution');values={};exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='dataclasses',names=[ast.alias(name='dataclass')],level=0),copy.deepcopy(node)],type_ignores=[])),'actual-Native-execution-schema','exec'),values)
  from dataclasses import asdict
  real=asdict(values['NativeGraphExecution']('replay',replay_executed=True));self.assertEqual(set(real),{'mode','fallback_reason','capture_attempted','replay_executed'});self.assertNotIn('used_aclgraph',real)
  G=B/'budget-diagnostics/v205-root-execution';guard=ast.parse((G/'guard205.py').read_text());scope=next(n for n in guard.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SCOPE'for t in n.targets));ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[scope],type_ignores=[])),'real-guard-scope','exec'),ns)
  execute=next(n for n in ast.parse((B/'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='execute_locked');row=next(n for n in ast.walk(execute)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='row'for t in n.targets))
  ns.update(datetime=datetime,tag='HOST205',ACTION='HOST',readiness={},LOCK=B/'.ROOT_NPU_EXECUTION.lock',MAX_SECONDS=7200,MAX_TAG_BYTES=256*1024**2,MAX_AGGREGATE_BYTES=512*1024**2)
  exec(compile(ast.fix_missing_locations(ast.Module(body=[row],type_ignores=[])),'actual-receipt-row','exec'),ns);self.assertEqual(ns['row']['new_target_keys'],0)
  with self.assertRaises(FileNotFoundError):check205.require_ready()
  self.assertIn('safe_storage177.py',(G/'guard205.py').read_text())
  for shell_name in('startup205.sh','launch205.sh'):
   shell=(G/shell_name).read_text();self.assertNotIn('budget-calibration-20561015',shell)
   self.assertIn('task_root='+str(P),shell);self.assertIn(str(B/'serial-v1'),shell)
  self.assertTrue((B/'serial-v1/runtime.sh').is_file());self.assertTrue((B/'serial-v1/map-current.py').is_file())
  launch_node=copy.deepcopy(next(n for n in ast.parse((B/'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='launch_argv'))
  launch_text=ast.unparse(launch_node).replace('launch_v154.sh','launch205.sh')
  from base205 import ACTION
  launch_ns=dict(Path=Path,__file__=str(G/'guard205.py'),ACTION=ACTION)
  exec(compile(launch_text,'actual-original-launch-argv-HOST','exec'),launch_ns)
  actual_argv=launch_ns['launch_argv']('HOST205')
  self.assertEqual(actual_argv[-6:],['/bin/bash',str(G/'launch205.sh'),'--action',ACTION,'--tag','HOST205'])
  for path in(G/'launch205.sh',G/'startup205.sh',B/'serial-v1/runtime.sh',B/'serial-v1/map-current.py'):self.assertTrue(path.is_file())
  DETAILS['actual_launch_paths']=dict(original_launch_argv_AST_executed=True,argv=actual_argv,launch_startup_runtime_map_paths_exist=True,no_subprocess_launched=True)
  from base205 import parent_ready,PARENT_ACTION,P197
  from diagnostic_authorization import activate_local_diagnostic,require_startup_scope
  readiness=dict(parent=parent_ready())
  main=next(n for n in code.body if isinstance(n,ast.FunctionDef)and n.name=='main')
  specrow=next(n for n in main.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='spec'for t in n.targets))
  values=dict(readiness=readiness,PARENT_ACTION=PARENT_ACTION)
  exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(specrow)],type_ignores=[])),'actual-runtime-parent-permit-spec','exec'),values)
  specification=values['spec'];self.assertEqual(specification['action'],'split192-candidate-full463')
  expected=dict(CPU_sha256=sha(BASE/'CPU_RESULT_V132.json'),freeze_sha256=sha(BASE/'PREPARED_SOURCE_HASHES_V132.json'))
  self.assertEqual(specification['identity'],expected)
  with activate_local_diagnostic(specification)as permit:self.assertIs(require_startup_scope(),permit)
  from launch_guard import LaunchBlocked
  with self.assertRaises(LaunchBlocked):require_startup_scope()
  wrong=dict(specification,identity=dict(CPU_sha256=sha(P197/'CPU_RESULT197.json'),freeze_sha256=sha(P197/'FROZEN_SOURCE197.json')))
  with self.assertRaisesRegex(RuntimeError,'worker source identity differs'):
   with activate_local_diagnostic(wrong):raise AssertionError('foreign197 identity must not activate P148')
  DETAILS['actual_parent_permit']=dict(runtime_spec_AST_executed=True,parent_identity_is_P148=True,actual_original_activate_positive=True,foreign197_identity_rejected=True,finally_context_revoked=True)
  DETAILS['runtime_guard']=dict(actual_new_import_binding=True,one_original_constructor_capture=True,metrics_sha_bound=True,real_Native_fourfields=True,actual_guard_receipt_row=True,ROOT_freeze_required=True)

 def test_actual_fork_fallback_ABI_and_combined_scope_report(self):
  import mapper205 as m,target_layout as layout,scope205
  predicate=native_predicate()
  tr_name='vllm_ascend.spec_decode.pearl.tree';model_name='vllm_ascend.spec_decode.pearl.native_model'
  previous={name:sys.modules.get(name)for name in(tr_name,model_name)}
  tr=types.ModuleType(tr_name);tr.cached_cpu_tree_speculation_plan=plan
  model=types.ModuleType(model_name);model._is_contiguous_linear_tree_plan=predicate
  sys.modules.update({tr_name:tr,model_name:model})
  tree=ast.parse((NATIVE/'native_engine.py').read_text())
  fn=copy.deepcopy(next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='_pad_spec_rhythm_target_tree_graph'))
  values=dict(envs=NS(VLLM_ASCEND_SPECRHYTHM_TREE_GRAPH=True),cached_cpu_tree_speculation_plan=plan,
   PearlPipelineState=lambda tokens,prompt_length:NS(token_ids=tokens))
  exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])),
   'original-Native-generic-pad-HOST', 'exec'),values)
  original_native=values[fn.name]
  old_source=B/'tree-causal80-small-cohort-active32-full-v202/full-host-attempts/BEFORE_FULL9_REVISION2/P202/mapper202.py'
  old_install=copy.deepcopy(next(n for n in ast.parse(old_source.read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='install_pad'))
  oldns=dict(require=require,check_frame=m.check_frame,active=lambda e:True)
  exec(compile(ast.fix_missing_locations(ast.Module(body=[old_install],type_ignores=[])),'preserved-pre-REV2-real-wrapper','exec'),oldns)
  original_family=layout.plan_family;original_envelope=layout.envelope;records=[]
  class Engine:
   def __init__(self):
    self.config=NS(enforce_eager=False,spec_rhythm_stable_graphs=True,spec_rhythm_tree_width=2,spec_rhythm_tree_depth=4,max_model_len=4096)
    self._tree_graph_scratch_sequence_ids=tuple(range(463,527));self._v205_usage_geometry=m._ISSUER
    self.cache_allocation=NS(block_tables=[[i]for i in range(527)]);self.cache_block_tables=object()
  try:
   for n in range(9,16):
    fork=[plan(2,4,127,4096,4)for _ in range(n)]
    for p in fork:p.parent_indices=Vector([-1,0,0,2]);p.positions=Vector([127,128,129,129,130])
    self.assertTrue(all(not predicate(p)for p in fork));e=Engine();args=(fork,[99]*n,[[10,11,12,13]for _ in range(n)],list(range(n)),[],[])
    # Actual frozen production parent has a narrower B/topology domain and
    # already rejects B4 forks. Preserve that ValueError; do not claim support.
    with self.assertRaisesRegex(ValueError,'causal B1..4'):
     layout.pad_target(e,*args)
    # Separate generic Native ABI boundary: the original Native pad itself
    # handles true forks. Family below is derived only from the actual Native
    # structural predicate, not a fabricated candidate flag or model result.
    layout.plan_family=lambda ps:'causal'if all(predicate(p)for p in ps)else'full9'
    emitted=[]
    def carrier_delegate(a,q,f):return m.expected_geometry(a,q,f)
    carrier_delegate._v205_control=NS(mapper=carrier_delegate,engine=e,mode='candidate')
    layout.envelope=carrier_delegate
    def generic(instance,*a):
     value=original_native(instance,*a);emitted.append(value);return value
    Engine._pad_spec_rhythm_target_tree_graph=generic
    oldns['install_pad'](NS(NativePearlEngine=Engine))
    with self.assertRaisesRegex(RuntimeError,'real current allB4'):
     e._pad_spec_rhythm_target_tree_graph(*args)
    self.assertEqual(len(emitted),1)
    Engine._pad_spec_rhythm_target_tree_graph=generic
    m.install_pad(NS(NativePearlEngine=Engine));out=e._pad_spec_rhythm_target_tree_graph(*args)
    self.assertIs(out,emitted[-1]);self.assertEqual(out[4],n)
    self.assertTrue(all(a is b for a,b in zip(out[0][:n],fork,strict=True)))
    # The same generic old padding is the wrong geometry for new causal
    # usage; the new strict frame gate must still reject it.
    causal=[plan(2,4,127,4096,4)for _ in range(n)]
    def bad_real_count(instance,*a):
     frame=generic(instance,*a);return (*frame[:4],n+1)
    Engine._pad_spec_rhythm_target_tree_graph=bad_real_count
    m.install_pad(NS(NativePearlEngine=Engine))
    with self.assertRaisesRegex(RuntimeError,'actual lease'):
     e._pad_spec_rhythm_target_tree_graph(causal,*args[1:])
    records.append(dict(N=n,Native_predicate_all_fork=True,parent_production_B4fork_rejected=True,
     generic_Native_frame_passthrough_identity=True,generic_M=sum(p.candidate_budget+1 for p in out[0]),generic_S=len(out[0])))
    layout.plan_family=original_family
   # Execute the actual scope report. Parent equality True must not overwrite
   # the already-computed current combined numerical-variant classification.
   e=NS();old=NS(require=lambda engine:frozenset(),_proof=dict(old_padding_behavior_equal=True,numerical_variant_only=False))
   s=NS(require=lambda engine:frozenset(),original=old,proof=dict(old_padding_behavior_equal=False,numerical_variant_only=True))
   report=scope205.report(s,e)
   self.assertIs(report['old_padding_behavior_equal'],False);self.assertIs(report['numerical_variant_only'],True)
  finally:
   layout.plan_family=original_family;layout.envelope=original_envelope
   for name,prior in previous.items():
    if prior is None:sys.modules.pop(name,None)
    else:sys.modules[name]=prior
  DETAILS['actual_full9_fallback']=dict(seven=records,original_Native_pad_AST_executed=True,original_Native_linear_predicate_executed=True,
   old_wrapper_generic_false_reject_reproduced=True,new_wrapper_exact_generic_return_identity=True,
   actual_parent_production_B4fork_rejection_unchanged=True,causal_bad_frame_still_rejected=True,
   scope_report_uses_actual_combined_classification=True,new_production_fork_qualification=False,model_math_claimed=False)

 def test_actual_selected_candidate_owner_and_original_restore_pad(self):
  import mapper205 as m,target_layout as layout
  cls=m.controls_class();ns=cls.__mro__[1].__init__.__globals__;ns['observe_cpu_device_context']=lambda *a,**kw:dict(passed=True)
  class Policy:
   def allocate(self,*a,**kw):raise AssertionError('allocator must remain original')
  policy=Policy()
  def shape(policy):
   def call(*a,**kw):return policy
   return call
  rhythm=NS(SpecRhythmBudgetShaper=type('Shaper',(),{'shape':shape(policy)}))
  control=cls(native=NS(),rhythm=rhythm,policy=policy,torch=NS(),constructor_state=NS(),is_draft=False,
   configuration=dict(target_cpu_context_clean=True,draft_cpu_context_clean=True))
  @dataclass(frozen=True)
  class Config:
   spec_rhythm_max_eager_tokens:int=5
   spec_rhythm_draft_token_budget:object=None
   spec_rhythm_eager_reserve_tokens:int=0
   max_model_len:int=4096
  class Engine:pass
  e=Engine();e.config=Config();e.cache_allocation=e.cache_block_tables=None;e.is_draft=False
  e._tree_graph_scratch_sequence_ids=tuple(range(463,527))
  e.graph_runner=NS(target_entries={k:NS(runtime_validated=True)for k in(m.GREEDY,m.HIDDEN)},disabled_entry_keys=set(),graph_cache_sealed=False)
  names=('vllm_ascend.spec_decode.pearl.tree','vllm_ascend.spec_decode.pearl.native_model');prior={n:sys.modules.get(n)for n in names}
  tr=types.ModuleType(names[0]);tr.cached_cpu_tree_speculation_plan=plan
  model=types.ModuleType(names[1]);model._is_contiguous_linear_tree_plan=native_predicate();sys.modules.update(dict(zip(names,(tr,model))))
  frames=[];record=[]
  def original(instance,*args):
   value=layout.pad_target(instance,*args);frames.append(value);return value
  Engine._pad_spec_rhythm_target_tree_graph=original
  m.install_pad(NS(NativePearlEngine=Engine));original_config=e.config
  def reserve():e.cache_allocation=NS(block_tables=[[i]for i in range(527)]);e.cache_block_tables=object()
  def release():e.cache_allocation=e.cache_block_tables=None
  try:
   control.select('candidate',engine=e);self.assertIs(layout.envelope,control.mapper)
   self.assertIs(control.mapper._v205_control,control);e._v205_parent_component={}
   evidence=dict(audit=dict(math_independent_qualified=True));e._v205_actual_usage=evidence;m.issue(e,evidence)
   for n in range(9,16):
    plans=[plan(2,4,0,4096,4)for _ in range(n)];args=(plans,[99]*n,[[1,2,3,4]for _ in range(n)],list(range(n)),[],[])
    reserve();candidate=e._pad_spec_rhythm_target_tree_graph(*args);self.assertIs(candidate,frames[-1]);self.assertEqual((sum(p.candidate_budget+1 for p in candidate[0]),len(candidate[0])),(80,16));release()
    control.select('original',engine=e);self.assertIs(e.config,original_config);self.assertIs(layout.envelope,control.original_envelope)
    self.assertTrue(hasattr(e,'_v205_usage_permission')) # permission persists honestly
    reserve();old=e._pad_spec_rhythm_target_tree_graph(*args);self.assertIs(old,frames[-1]);self.assertEqual((sum(p.candidate_budget+1 for p in old[0]),len(old[0])),(128,38));release()
    control.select('candidate',engine=e);reserve();again=e._pad_spec_rhythm_target_tree_graph(*args);self.assertEqual((sum(p.candidate_budget+1 for p in again[0]),len(again[0])),(80,16));release()
    # Ownership is actual envelope identity, not merely a surviving token.
    previous_envelope=layout.envelope;layout.envelope=control.original_envelope
    reserve();uninstalled=e._pad_spec_rhythm_target_tree_graph(*args);self.assertIs(uninstalled,frames[-1]);self.assertEqual(len(uninstalled[0]),38);release();layout.envelope=previous_envelope
    record.append(dict(N=n,candidate_M=80,original_M=128,original_frame_identity=True,permission_persists=True))
   def broken(instance,*args):
    frame=original(instance,*args);return (*frame[:4],frame[4]+1)
   Engine._pad_spec_rhythm_target_tree_graph=broken;m.install_pad(NS(NativePearlEngine=Engine));reserve()
   with self.assertRaisesRegex(RuntimeError,'actual lease'):
    e._pad_spec_rhythm_target_tree_graph(*args)
   release()
  finally:
   release();control.select('original',engine=e)
   for name,value in prior.items():
    if value is None:sys.modules.pop(name,None)
    else:sys.modules[name]=value
  DETAILS['actual_original_restore']=dict(actual_Controls_select_and_observe=True,current_issuer_issued_on_same_carrier_engine=True,
   actual_mapper_control_identity=True,actual_original_and_candidate_transitions=record,uninstalled_delegate_returns_original_frame=True,
   candidate_corrupted_frame_still_rejected=True,original_config_identity_restored=True,model_math_claimed=False)

 def test_actual_single_writer_shared_FS_and_common_failure(self):
  import multiprocessing,tempfile,queue
  parent=B/'tree-causal80-small-cohort-active32-full-v202/actor202.py'
  def output_lambda(path):
   tree=ast.parse(path.read_text())
   calls=[n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='stage'
    and n.args and isinstance(n.args[0],ast.Constant)and str(n.args[0].value).endswith('/output')]
   self.assertEqual(len(calls),1);self.assertIsInstance(calls[0].args[1],ast.Lambda)
   return copy.deepcopy(calls[0].args[1])
  old=output_lambda(parent);new=output_lambda(P/'actor205.py')
  # Actual AST inverse: only the writer predicate wraps the original mkdir.
  self.assertIsInstance(new.body,ast.IfExp)
  self.assertEqual(ast.dump(new.body.body,include_attributes=False),ast.dump(old.body,include_attributes=False))
  ctx=multiprocessing.get_context('fork');results=[]
  for scenario,node,preexisting in(('old-four-writers',old,False),('new-single-writer',new,False),('new-existing-dir',new,True)):
   with tempfile.TemporaryDirectory(prefix='V205-HOST-shared-FS-')as root:
    folder=Path(root)/'same-output'
    if preexisting:folder.mkdir()
    barrier=ctx.Barrier(4);errors=ctx.Array('i',[0]*4,lock=True);out=ctx.Queue()
    def worker(rank):
     error=None
     try:
      e=NS(rank=rank,topology=NS(draft_leader_rank=0))
      callback=eval(compile(ast.fix_missing_locations(ast.Expression(copy.deepcopy(node))),'actual-actor-output-lambda-HOST','eval'),dict(folder=folder,e=e))
      barrier.wait(timeout=10)
      try:callback()
      except BaseException as ex:error=type(ex).__name__+': '+str(ex)
      errors[rank]=int(error is not None);barrier.wait(timeout=10)
      out.put(dict(rank=rank,local_error=error,common_reject=any(errors[:]),folder_exists=folder.is_dir()))
     except BaseException as ex:out.put(dict(rank=rank,carrier_error=repr(ex)))
    processes=[ctx.Process(target=worker,args=(rank,))for rank in range(4)]
    try:
     for process in processes:process.start()
     packets=[out.get(timeout=20)for _ in range(4)]
     for process in processes:process.join(timeout=10)
     self.assertTrue(all(not process.is_alive()and process.exitcode==0 for process in processes))
    finally:
     for process in processes:
      if process.is_alive():process.terminate();process.join(timeout=5)
     out.close();out.join_thread()
    packets.sort(key=lambda v:v['rank']);self.assertEqual([r['rank']for r in packets],list(range(4)))
    self.assertFalse(any('carrier_error'in r for r in packets));self.assertTrue(all(r['folder_exists']for r in packets))
    failures=[r for r in packets if r['local_error']is not None]
    if scenario=='old-four-writers':
     self.assertEqual(len(failures),3);self.assertTrue(all(r['local_error'].startswith('FileExistsError:')for r in failures));self.assertTrue(all(r['common_reject']for r in packets))
    elif scenario=='new-single-writer':self.assertEqual(failures,[]);self.assertTrue(all(not r['common_reject']for r in packets))
    else:
     self.assertEqual([r['rank']for r in failures],[0]);self.assertTrue(failures[0]['local_error'].startswith('FileExistsError:'));self.assertTrue(all(r['common_reject']for r in packets))
    results.append(dict(scenario=scenario,actual_processes=4,packets=packets))
  DETAILS['actual_single_writer']=dict(actual_old_and_new_actor_mkdir_lambdas_AST_executed=True,real_four_process_shared_temporary_FS=True,
   original_stage_call_and_all4_boundary_unchanged=True,old_three_FileExistsErrors=True,new_all4_pass=True,
   preexisting_leader_FileExistsError_and_all4_common_reject=True,processes_closed=True,NPU_stage_fence_not_executed_or_claimed=True,
   model_math_claimed=False,cases=results)
