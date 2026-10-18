"""New scheduling/source/allocator/planner HOST only; no model numerics."""
import ast,copy,json,sys,types,unittest
from pathlib import Path
from base197 import P,B,BASE,require
import empty197
DETAILS={}

def original_source():
 import live_decode_patch
 from window160 import adapt_source
 return adapt_source(live_decode_patch.build_live_runtime(),32)

def rhythm():
 import importlib.util
 path=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/spec_rhythm.py')
 source=path.read_text().replace('from vllm_ascend.spec_decode.pearl.roofline import ProfiledRoofline','class ProfiledRoofline: pass')
 module=types.ModuleType('_v197_actual_HOST_rhythm');sys.modules[module.__name__]=module
 exec(compile(source,str(path)+'::HOST-import-only-roofline-stub','exec'),vars(module));return module

class Tests(unittest.TestCase):
 def test_actual_generated_whole_inverse_and_admission_clock_publication(self):
  before=original_source();after=empty197.patch(before);self.assertTrue(empty197.assert_inverse(before,after))
  with self.assertRaises(RuntimeError):empty197.patch(after)
  with self.assertRaises(RuntimeError):empty197.unpatch(before)
  fn=empty197.admission(ast.parse(after));self.assertIn('receive_live_admissions',ast.unparse(fn.body[0]));self.assertEqual(ast.unparse(fn.body[1]),'if active:\n    return closed')
  # Run actual original admission function (no mathematical model carrier).
  code=ast.Module(body=[copy.deepcopy(fn)],type_ignores=[]);node=code.body[0]
  # Original annotations are not needed by this HOST call; actual body is exact.
  node.returns=None
  for a in (*node.args.args,*node.args.kwonlyargs):a.annotation=None
  class Tensor:
   def __init__(self,v):self.v=v
   def cpu(self):return self
   def item(self):return self.v[0]
  class State:
   def __init__(self,i):self.token_ids=[10+i]*3;self.prompt_length=3;self.committed_length=3
   @property
   def committed_completion_token_ids(self):return self.token_ids[self.prompt_length:]
  n=463;local=[State(i)for i in range(n)];draft=[State(i)for i in range(n)];target=local
  clock=[0.];seen=[];receives=[];publ=[];pref=[]
  states={i:types.SimpleNamespace(home_batch_id=i%2,arrival_wait_ms=0.,decode_elapsed_ms=0.,decode_start_elapsed_ms=None,delivered_tokens=0,add_decode_time=lambda t:None)for i in range(n)}
  params=[types.SimpleNamespace(arrival_ts=i*.25)for i in range(n)]
  def sample(prompts,ss,ids):
   self.assertFalse(ns['active']);pref.append(list(ids));return [100+i for i in ids]
  engine=types.SimpleNamespace(rank=1,topology=types.SimpleNamespace(target_leader_rank=1),config=types.SimpleNamespace(spec_rhythm_online_prefill=True,spec_rhythm_kv_ready_arrivals=False,max_num_batched_tokens=6),
   _prefill_and_sample_target_batch=sample,_deliver_committed_tokens=lambda i,p,s:publ.append(i),eos_token_ids=set())
  hooks=types.SimpleNamespace(reset_for_original_admission_fallback=lambda **kw:None,reserved_slots=lambda:0,filter_pending_candidates=lambda ids:ids,reserve_at_original_admission=lambda **kw:False)
  ns=dict(_v75_hooks=hooks,cache_activated=set(),cache_released=set(),self=engine,active=[],pending_admission=list(range(n)),initial_batch_size=32,
   receive_live_admissions=lambda **kw:receives.append(kw)or False,prefill_coalescing=False,
   request_params=params,has_slo_constraints=True,envs=types.SimpleNamespace(VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION=False),
   local_states=local,draft_states=draft,target_states=target,states=states,prefetched=set(),
   torch=types.SimpleNamespace(tensor=lambda v,**kw:seen.append('arrival/prefill-clock')or Tensor(v),float64=object()),time=types.SimpleNamespace(time=lambda:clock[0],perf_counter=lambda:clock[0]),
   _dt_cpu_broadcast=lambda *a,**kw:None,order_spec_rhythm_admission_params=lambda ids,**kw:list(ids),
   activate_online_cache=lambda ids:None,release_online_cache=lambda ids:None,_finished=lambda *a:False,completed={},finished_decode_elapsed={},counters={'spec_rhythm_prefill_batches':0,'spec_rhythm_prefill_requests':0,'spec_rhythm_prefill_singleton_batches':0,'spec_rhythm_prefill_pair_batches':0,'spec_rhythm_prefill_multi_batches':0,'spec_rhythm_peak_active_requests':0})
  exec(compile(ast.fix_missing_locations(code),'V197::actual-generated-admission-HOST','exec'),ns)
  # Future arrivals do not prefill, and no ready publication is held once empty.
  clock[0]=-1;ns['admit_available']();self.assertEqual(pref,[])
  clock[0]=0;ns['admit_available']();self.assertEqual(pref,[[0]]);self.assertEqual(publ,[0])
  counts=(len(seen),len(pref),len(publ));clock[0]=1000
  ns['admit_available']();self.assertEqual((len(seen),len(pref),len(publ)),counts)
  self.assertEqual(len(receives),3) # live receive remains before the active guard.
  ns['active'].clear()
  while ns['pending_admission']:
   old=len(pref);ns['admit_available']();self.assertEqual(len(pref),old+1)
   self.assertEqual(len(pref[-1]),min(2,463-len(publ)+len(pref[-1])))
   ns['active'].clear() # explicit HOST completion carrier, not model evidence.
  self.assertEqual(publ,list(range(463)));self.assertEqual(sum(map(len,pref)),463)
  self.assertTrue(all(len(v)<=2 for v in pref)) # original token budget, not min4.
  DETAILS['actual_generated_admission']=dict(receive_before_guard=True,full_decode_AST_inverse=True,finite463_no_drop=True,token_budget_small_batch_max2=True,publication_immediate_before_return=True,no_clock_call_while_active=True)

 def test_actual_native_allocator_bank32_and_original_planner_subset(self):
  import prefill197,subset197,next4_197
  path=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_cache.py');ns={};exec(compile(path.read_text(),str(path),'exec'),ns)
  rh=rhythm();pages=[]
  for n in (2,32):
   cache=ns['NativePrefixCache'](544,32,128)
   class Engine:
    is_draft=True;cache_allocation=None;cache_block_tables=None
    def _allocate_cache(self,prompts,**kw):self.cache_allocation=cache.allocate(prompts,False,sequence_capacity=527);self.cache_block_tables=self.cache_allocation.block_tables
    def _activate_cache_sequence(self,i,p,**kw):cache.activate_sequence(i,p,enable_prefix_caching=False)
    def _ensure_cache_capacity(self,ids,pos):cache.ensure_capacity(ids,pos)
   e=Engine();rows=[dict(request_id=str(i),token_ids=[i+1]*v,length=v,original_prompt='HOST')for i,v in enumerate(([127,129,257,59,265,86,300,511]*9)[:65])]
   bank=prefill197.Bank(e,rows,n);guard=bank.guard();self.assertEqual(len(guard['pages']),len(set(guard['pages'])));self.assertLessEqual(len(guard['pages']),544)
   for owner in range(2*n):
    p=bank.prompts[owner%n]
    self.assertTrue(all(bank.engine.cache_allocation.block_tables[owner][j//128]>=0 for j in range(len(p)+4)))
   bank.contexts=[p+[100+i]for i,p in enumerate(bank.prompts)];bank.reference_contexts=copy.deepcopy(bank.contexts)
   bank.runtime_states={i:rh.SpecRhythmRuntimeState(i,i%2,max_gamma=5,output_token_budget=256)for i in range(n)};bank.publication_active=tuple(range(n))
   selected=subset197.prepare(e,rh,bank);wanted=1 if n==2 else 16
   self.assertEqual(selected.n,wanted);self.assertEqual(selected.ids,list(range(0,n,2)));self.assertEqual(selected.reference_ids,[i+n for i in selected.ids])
   wp=next4_197.work_plan(dict(N=wanted,budgets=[4]*wanted,ids=selected.ids),100)
   self.assertEqual(sum(wp.normal_budgets.values()),4*wanted);self.assertEqual(tuple(wp.normal_budgets),tuple(selected.ids))
   for i in selected.ids:selected.controller.publish([selected.controller.new_ticket(i,gamma=4,eager=False)])
   target=selected.controller.build_plan(bank.publication_active,**selected.kwargs)
   self.assertEqual(tuple(target.target_request_indices),tuple(selected.ids));self.assertLessEqual(len(target.target_request_indices),16)
   pages.append(dict(N=n,actual_pages=len(guard['pages']),nextN=wanted,ids=selected.ids,refs=selected.reference_ids,target_ids=list(target.target_request_indices)))
  DETAILS['actual_bank_and_planner']=pages

 def test_actual_production_gate_source_and_parent_single_constructor(self):
  import subset197,next4_197
  self.assertEqual(next4_197.reference_entries(types.SimpleNamespace(n=1,reference_ids=[32],reference_contexts=[[1,2,3]],reference_ledger={32:dict(safe_before=2,tokens={})})),[(32,2,2,[3])])
  with self.assertRaises(RuntimeError):next4_197.work_plan(dict(N=32,budgets=[4]*32,ids=list(range(32))),1)
  runtime=ast.parse((P/'runtime197.py').read_text());fn=next(n for n in runtime.body if isinstance(n,ast.FunctionDef)and n.name=='worker')
  self.assertEqual(sum(isinstance(n,ast.Call)and ast.unparse(n.func)=='native.NativePearlEngine'for n in ast.walk(fn)),1)
  self.assertEqual(sum(isinstance(n,ast.Call)and ast.unparse(n.func)=='capture_original_protocol_before_startup'for n in ast.walk(fn)),1)
  self.assertIn("parent_module('component173').run",(P/'component197.py').read_text())
  self.assertNotIn('replace(', (P/'config197.py').read_text());self.assertIn('from startup_config_v132 import make_config',(P/'config197.py').read_text())
  self.assertIn('import ast,copy,gzip,json,math,statistics',(P/'trace197.py').read_text());self.assertIn('inherited_module,sha',(P/'trace197.py').read_text())
  DETAILS['source_gates']=dict(original_single_capture_constructor=True,parent173_only=True,N32_new_target_rejected=True,no_config_delta=True,metrics_sha_binding=True)

 def test_actual_guard_row_and_storage_scope(self):
  G=B/'budget-diagnostics/v197-root-execution';tree=ast.parse((G/'guard197.py').read_text())
  scope=next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='SCOPE'for t in n.targets));ns={};exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.Assign(targets=[ast.Name(id='SCOPE',ctx=ast.Store())],value=scope)],type_ignores=[])),'HOST::actual-scope','exec'),ns)
  self.assertIs(ns['SCOPE']['empty_active_only'],True);self.assertEqual(ns['SCOPE']['active_windows_requested'],[32])
  # Original supervisor body is source-inherited; statically check exact source
  # argument changes and177 frozen mutable.tmp race fix binding.
  self.assertIn('safe_storage177.py',(G/'guard197.py').read_text());self.assertIn("live_roots=(P/'actual',)",(G/'guard197.py').read_text())
  self.assertIn("f['ROOT_source_reviewed'] is True",(G/'guard197.py').read_text())
  self.assertIn('SOURCE_PEER',(G/'guard197.py').read_text())
  # Execute the actual inherited supervisor receipt constructor, before any
  # child/device operation; catches duplicate **SCOPE keys or schema assumptions.
  import datetime
  original=B/'budget-diagnostics/v154-root-execution/supervise_v154.py'
  unit=ast.parse(original.read_text());execute=next(n for n in unit.body if isinstance(n,ast.FunctionDef)and n.name=='execute_locked')
  row=next(n for n in ast.walk(execute)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='row'for t in n.targets))
  values=dict(ns,datetime=datetime,tag='HOST-only-safe-tag',ACTION='empty-active-cohort-causal80-active32-full463',readiness={'HOST':True},LOCK=B/'.ROOT_NPU_EXECUTION.lock',MAX_SECONDS=7200,MAX_TAG_BYTES=256*1024**2,MAX_AGGREGATE_BYTES=512*1024**2)
  exec(compile(ast.fix_missing_locations(ast.Module(body=[row],type_ignores=[])),'V197::actual-inherited-receipt-constructor','exec'),values)
  self.assertIs(values['row']['empty_active_only'],True);self.assertEqual(values['row']['tag'],'HOST-only-safe-tag')
  DETAILS['guard']=dict(original_selflock_and_ownedcleanup_source_retained=True,frozen177_scoped_temp_race_fix=True,ROOT_freeze_required=True)
