"""Four new executable HOST ABI/control checks; no backend or model suite."""
import ast,copy,datetime,json,sys,types,unittest
from dataclasses import dataclass
from pathlib import Path
from base202 import P,B,BASE,require,sha
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
  import mapper202 as m
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
   e._v202_parent_component={};e.cache_allocation=object()
   with m.qualification_geometry(e):
    for n in range(9,16):self.assertEqual(control.mapper(n,5*n,'causal'),(80,16,[5]*(16-n)))
   self.assertFalse(hasattr(e,'_v202_usage_geometry'));e.cache_allocation=None
   evidence=dict(audit=dict(math_independent_qualified=True));e._v202_actual_usage=evidence;m.issue(e,evidence)
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
  import bank202 as b,mapper202 as m,target202 as t,target_layout as layout
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
  model=types.ModuleType(names[2]);model._is_contiguous_linear_tree_plan=native_predicate()
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
    n=case['N'];e._v202_parent_component={};e.graph_runner=NS(graph_cache_sealed=False)
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
  finally:
   b.tensor_hash=oldhash;b.prefix_evidence=oldprefix;layout.envelope=oldenv
   for n,old in prior.items():
    if old is None:sys.modules.pop(n,None)
    else:sys.modules[n]=old
  DETAILS['real_pad_bank']=dict(actual_NativePrefixCache=True,actual_original_pad_function=True,actual_Native_linear_predicate=True,all32_prefill_prefix_ready=True,full16_reference_offset16=True,seven=records,model_math_claimed=False)

 def test_actual_subset_work_commit_and_integer_BONUS(self):
  import actor202,target202,audit_usage202
  fake=types.ModuleType('torch');old=sys.modules.get('torch');sys.modules['torch']=fake
  try:
   for n in range(9,16):
    plans=[NS(candidate_budget=4,depth=4)for _ in range(16)]
    full=dict(plans=plans,rows=[[10,11,12,13]for _ in range(16)],confidences=[.5]*16,selected=[list(range(4))]*16,source=dict(ids=list(range(16)),contexts=[dict(token_ids=[i,99],output_offset=0)for i in range(16)]),draft_proof={'full16':True})
    # The real typed require_work also inspects topology only through budget.
    for p in plans:p.parent_indices=Vector([-1,0,1,2])
    w=actor202.subset(NS(),full,n,1);self.assertEqual(w['source']['ids'],list(range(n)));self.assertEqual(w['work_plan'].allocated_selected_candidates,4*n)
    history=[[i,99]for i in range(16)];bank=NS(contexts=copy.deepcopy(history),reference_contexts=copy.deepcopy(history),guard=lambda:{'actual_carrier':True})
    commit=dict(committed_tokens=[[10]for _ in range(n)],accepted_paths=[[-1]*4 for _ in range(n)],actual_logical_tokens=[v+[10]for v in history[:n]])
    t=target202.commit_reference(NS(is_draft=True),bank,w,None,commit)
    self.assertEqual(bank.contexts[n:],history[n:]);self.assertEqual(len(bank.contexts),16);self.assertTrue(t['unselected_owners_preserved'])
    rows=[[10,11,12,13]]*n;head=[10,11,12,13,777]*n;emitted,paths=audit_usage202.verdict(rows,head)
    self.assertEqual(emitted,[[10,11,12,13,777]]*n);self.assertEqual(paths,[list(range(4))]*n)
  finally:
   if old is None:sys.modules.pop('torch',None)
   else:sys.modules['torch']=old
  DETAILS['actual_subset']=dict(actual_typed_work=True,unpublished_owners_not_advanced=True,retained_bank16=True,reference_offset16=True,actual_original_BONUS_integer_recomputed=True)

 def test_actual_runtime_projection_guard_row_and_fourfield_schema(self):
  import runtime202,component202,trace202,scope202,check202,extension202,audit202
  code=ast.parse((P/'runtime202.py').read_text());worker=next(n for n in code.body if isinstance(n,ast.FunctionDef)and n.name=='worker')
  self.assertEqual(sum(isinstance(n,ast.Call)and ast.unparse(n.func)=='native.NativePearlEngine'for n in ast.walk(worker)),1)
  self.assertEqual(sum(isinstance(n,ast.Call)and ast.unparse(n.func)=='capture_original_protocol_before_startup'for n in ast.walk(worker)),1)
  self.assertIs(trace202.metrics.__globals__['sha'],sha);self.assertIn('prefill202',(P/'component202.py').read_text());self.assertIn('actor202',(P/'component202.py').read_text())
  tree=ast.parse((NATIVE/'native_graph.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='NativeGraphExecution');values={};exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='dataclasses',names=[ast.alias(name='dataclass')],level=0),copy.deepcopy(node)],type_ignores=[])),'actual-Native-execution-schema','exec'),values)
  from dataclasses import asdict
  real=asdict(values['NativeGraphExecution']('replay',replay_executed=True));self.assertEqual(set(real),{'mode','fallback_reason','capture_attempted','replay_executed'});self.assertNotIn('used_aclgraph',real)
  G=B/'budget-diagnostics/v202-root-execution';guard=ast.parse((G/'guard202.py').read_text());scope=next(n for n in guard.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SCOPE'for t in n.targets));ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[scope],type_ignores=[])),'real-guard-scope','exec'),ns)
  execute=next(n for n in ast.parse((B/'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text()).body if isinstance(n,ast.FunctionDef)and n.name=='execute_locked');row=next(n for n in ast.walk(execute)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='row'for t in n.targets))
  ns.update(datetime=datetime,tag='HOST202',ACTION='HOST',readiness={},LOCK=B/'.ROOT_NPU_EXECUTION.lock',MAX_SECONDS=7200,MAX_TAG_BYTES=256*1024**2,MAX_AGGREGATE_BYTES=512*1024**2)
  exec(compile(ast.fix_missing_locations(ast.Module(body=[row],type_ignores=[])),'actual-receipt-row','exec'),ns);self.assertEqual(ns['row']['new_target_keys'],0)
  with self.assertRaises(FileNotFoundError):check202.require_ready()
  self.assertIn('safe_storage177.py',(G/'guard202.py').read_text())
  DETAILS['runtime_guard']=dict(actual_new_import_binding=True,one_original_constructor_capture=True,metrics_sha_bound=True,real_Native_fourfields=True,actual_guard_receipt_row=True,ROOT_freeze_required=True)
