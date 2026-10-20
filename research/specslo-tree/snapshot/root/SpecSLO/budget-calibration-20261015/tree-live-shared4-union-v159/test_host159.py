"""Seven small actual-source/HOST checks; no model, backend or old suite."""
import ast,copy,importlib.util,json,sys,types,unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch
from source159 import P,BASE,PROTO,checked,fast_source,shape_namespace
from scope159 import Capability,native_ready
from lease159 import guarded

def funcs(path,names,namespace):
    nodes=[copy.deepcopy(n) for n in ast.parse(checked(path).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path)+'::V159-HOST-actual-source','exec'),namespace)

class Host(unittest.TestCase):
 def test_actual_shape_allocator_unlocks_only_qualified_shared4_and_preserves_B5(self):
    ns=dict(P=BASE,ModuleType=types.ModuleType,sys=sys,ast=ast,json=json)
    from budget_policy import TreeBudgetPolicy
    from normal_admission_v140 import reset
    ns.update(TreeBudgetPolicy=TreeBudgetPolicy,reset=reset,NATIVE=checked(Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/spec_rhythm.py')))
    funcs(BASE/'test_full_process_v146_host.py',('actual_native_types','actual_live_plan'),ns)
    native=ns['actual_native_types']()
    off,_=shape_namespace(Capability());ns['shape']=off['shape'];old,policy=ns['actual_live_plan'](native,eager=True)
    enabled,_=shape_namespace(Capability(frozenset({(32,8)}),frozenset(),mode='qualifying'))
    ns['shape']=enabled['shape'];new,policy=ns['actual_live_plan'](native,eager=True)
    self.assertEqual(dict(old.eager_budgets),{});self.assertTrue(new.eager_budgets)
    self.assertEqual(dict(old.normal_budgets),dict(new.normal_budgets));self.assertEqual(max(new.normal_budgets.values()),4)
    self.assertEqual(set(new.eager_budgets.values()),{3});self.assertEqual(new.allocated_selected_candidates,sum(new.normal_budgets.values())+24)
    self.assertEqual(new.target_limit.check_actual_call(tuple(new.normal_budgets.values())),128)
    self.assertLessEqual(new.allocated_selected_candidates,192)
    ns2=dict(Counter=Counter);funcs(BASE/'test_candidate_v132_host.py',('State','Policy'),ns2)
    policy=ns2['Policy']();reset(policy,1);kwargs=dict(plan_id=1,normal_request_indices=tuple(range(8)),eager_request_indices=tuple(range(8,24)),
        states={i:ns2['State']() for i in range(24)},projected_wait_ms=0.,context_len=128,batch_size=24,
        draft_token_budget=192,eager_token_cap=3,eager_reserve_tokens=32,verification_roof=96)
    allow,_=shape_namespace(Capability(frozenset({(8,16)}),frozenset(),mode='qualifying'))
    small=allow['shape'](policy,types.SimpleNamespace(min_gamma=1,max_gamma=8),**kwargs)
    self.assertEqual(set(small.normal_budgets.values()),{5});self.assertEqual(dict(small.eager_budgets),{})
    self.assertEqual(enabled['_source_contract']['counts'],[1,2,1])
 def test_single_call_lease_actual_lifecycle_restore_and_stale_negative(self):
    from source159 import module,FUSION
    witness=module('repair_frontier_fusion',FUSION).AllocationWitness
    class E:
      def __init__(self):
        self.cache_allocation=types.SimpleNamespace(block_tables=[[0,1],[2,3]])
        self.cache_block_tables=object();self.config=types.SimpleNamespace(kvcache_block_size=128)
        self._dynamic_tree_known_KV={i:dict(safe_before=128,tokens={128:i}) for i in range(2)}
        self._dynamic_tree_generation=7;self.calls=[]
      def _ensure_cache_capacity(self,ids,positions):self.calls.append('ensure')
      def _release_cache_sequence(self,owner):self.calls.append('release')
      def _release_cache(self):self.calls.append('releaseall')
      def _activate_cache_sequence(self,*args):self.calls.append('activate')
      def _allocate_cache(self,*args):self.calls.append('allocate')
    e=E()
    with guarded(e,[0,1],witness) as lease:
      self.assertEqual(lease.check().block_tables,((0,1),(2,3)))
      e._ensure_cache_capacity([0],[129])
      with self.assertRaises(RuntimeError):e._release_cache_sequence(0)
      self.assertEqual(e.calls,['ensure'])
    self.assertFalse(any(k.startswith('_ensure') or k.startswith('_release') for k in vars(e)))
    with self.assertRaises(RuntimeError):
      with guarded(e,[0,1],witness):e.cache_allocation.block_tables[0][0]=99
    self.assertNotIn('_ensure_cache_capacity',vars(e));self.assertNotIn('_release_cache_sequence',vars(e))
    e.cache_allocation.block_tables[0][0]=0
    with self.assertRaises(RuntimeError):
      with guarded(e,[0,1],witness):e._dynamic_tree_known_KV[0]['tokens'][128]=100
    self.assertNotIn('_ensure_cache_capacity',vars(e))
 def test_exact_FastV5_prelude_inverse_and_runtime_status_AST_inverse(self):
    source,contract=fast_source();self.assertTrue(contract['exact_inverse'])
    self.assertEqual(source.replace(contract['added'],contract['removed']),checked(PROTO/'fast_tree_draft_v5.py').read_text())
    from live_decode_patch import build_live_runtime
    from status159 import rewrite
    actual=build_live_runtime();new,proof=rewrite(actual)
    self.assertEqual(proof['counts'],dict(submit=1,post=1,status=1));self.assertTrue(proof['exact_AST_inverse'])
    self.assertEqual(new.count('_v159_common_draft_status(self)'),1)
    self.assertLess(new.index('_v159_common_draft_status(self)'),new.index("'SpecSLO/DraftToTarget'"))
    self.assertEqual(new.count('_v159_capture_draft_error(self, _v159_error)'),2)
    for token in ('_v75_hooks.fatal_cleanup_all4','_v75_hooks.commit_after_original_finish_cycle','budget_plan.require_work'):
      self.assertIn(token,new)
    with self.assertRaises(RuntimeError):rewrite(actual.replace("with torch.profiler.record_function('SpecSLO/DraftCompute'):","with torch.profiler.record_function('changed'):"))
 def test_common_status_each_role_one_collective_then_all_raise(self):
    from status159 import common,capture
    calls=[]
    class T:
      def __init__(self,v):self.value=v[0]
      def item(self):return self.value
    dist=types.ModuleType('torch.distributed');dist.ReduceOp=types.SimpleNamespace(MAX='MAX')
    dist.all_reduce=lambda v,op,group:(calls.append((op,group,v.value)),setattr(v,'value',1))
    torch=types.ModuleType('torch');torch.int32='int32';torch.tensor=lambda values,**kw:T(values);torch.distributed=dist
    transport=types.ModuleType('host_tree_transport');transport.group=lambda e:'original-Gloo';transport.count=lambda e,k:calls.append((e.rank,k))
    with patch.dict(sys.modules,{'torch':torch,'torch.distributed':dist,'host_tree_transport':transport}):
      for rank in range(4):
        engine=types.SimpleNamespace(rank=rank)
        if rank==0:capture(engine,ValueError('genuine lease failure'))
        with self.assertRaises(RuntimeError):common(engine)
    self.assertEqual(sum(c[0]=='MAX' for c in calls),4)
    self.assertEqual([c for c in calls if len(c)==2],[(i,'draft_status') for i in range(4)])
    self.assertNotIn('torch',sys.modules)
 def test_factory_before_root_seed_preserves_original_normal_fallback(self):
    import live159
    oldfast=sys.modules.pop('fast_tree_draft_v5',None)
    fake_torch=types.ModuleType('torch')
    try:
      factory=live159.prepare_helpers(Capability())
      from source159 import module
      root=module('root_seed',BASE/'root_seed.py')
      class Runner:
        def _execute_draft(self,*a,**kw):return None
      class Engine:
        def draft_tree_forward(self,*a,**kw):return 'old'
        def _allocate_cache(self,*a,**kw):return None
      native=types.SimpleNamespace(NativeACLGraphRunner=Runner,NativePearlEngine=Engine)
      with patch.dict(sys.modules,{'torch':fake_torch}):root.install(native)
      root_closure=dict(zip(Engine.draft_tree_forward.__code__.co_freevars,[c.cell_contents for c in Engine.draft_tree_forward.__closure__]))
      self.assertIs(root_closure['original'].__globals__,factory['fast'].__dict__)
      self.assertEqual(factory['tree_entry']['P'],BASE);self.assertEqual(factory['tree_entry']['OUTPUT'],P)
      self.assertTrue(Path(factory['fast'].__file__).is_file())
      self.assertTrue(Path(factory['tree_entry']['__file__']).is_file())
      controls=factory['controls']['CandidateControls'];self.assertIs(controls.__init__.__globals__['_v159_shape'],factory['shape']['shape'])
      with self.assertRaises(RuntimeError):live159.prepare_helpers(Capability())
    finally:
      sys.modules.pop('fast_tree_draft_v5',None)
      if oldfast is not None:sys.modules['fast_tree_draft_v5']=oldfast
 def test_real_native_key_guard_and_unqualified_pre_model_fallback(self):
    import live159
    cap=Capability(frozenset({(32,32)}),frozenset({(32,1)}),mode='production')
    runner=types.SimpleNamespace(draft_entries={},target_entries={})
    e=types.SimpleNamespace(graph_runner=runner,draft_vocab_size=151936)
    self.assertFalse(native_ready(e,cap,chain_M=64,real_rows=64,union_M=36))
    runner.draft_entries[('draft-greedy:151936|steps:4|paged|confidence',64)]=types.SimpleNamespace(runtime_validated=True,validated_real_row_count=64)
    runner.target_entries[('target-hidden:0|steps:1|paged',(36,))]=types.SimpleNamespace(runtime_validated=True)
    self.assertTrue(native_ready(e,cap,chain_M=64,real_rows=64,union_M=36))
    # Disabled/unsupported paths must return before touching a backend/model or
    # requiring an allocation, and preserve the original B5 true branch path.
    p=lambda b:types.SimpleNamespace(candidate_budget=b)
    old=live159._capability;live159._capability=cap
    try:
      self.assertEqual(live159.fused_prelude(e,None,None,[p(5),p(3)],[0,1],{1:(p(3),[1,2,3])},[(0,128,[9],10)],[(1,131,3)],4),(None,None))
      self.assertEqual(live159.fused_prelude(e,None,None,[p(4),p(3)],[0,1],{1:(p(3),[1,2,3])},[(0,128,[9,8],10)],[(1,131,3)],4),(None,None))
    finally:live159._capability=old
 def test_new_live_fused_prelude_actual_metadata_and_postwrite_failure(self):
    # Reuse the single explicit HOST fixture function, not its tests. It runs
    # the actual Native cache + original metadata/pad AST, never a model.
    import live159
    from scope159 import require
    ns=dict(B=P.parent,checked=checked,copy=copy,sys=sys,types=types,importlib=importlib,Path=Path,ast=ast,gate=types.SimpleNamespace(require=require))
    funcs(P.parent/'tree-union-own-epoch-metadata-fix-local-v161/test_host161.py',('fixture',),ns)
    m,cache,allocate,PA=ns['fixture']();e=allocate()
    e.config.max_num_seqs=64;e.config.enable_prefix_caching=False;e.draft_vocab_size=151936
    e._dynamic_tree_known_KV={i:dict(safe_before=128,tokens={128:10+i}) for i in range(64)}
    m.Tensor.cpu=lambda v:v
    calls=[];corrupt=[False]
    def model_call(*frame):
      calls.append(('model',frame[0].shape[0]))
      if corrupt[0]:e._dynamic_tree_known_KV[0]['tokens'][128]=999
      return m.Tensor([[1,2]]*frame[0].shape[0])
    e.model.compute_greedy_tokens=lambda h,v:(calls.append(('head',h.shape[0])) or m.Tensor(list(range(h.shape[0]))))
    e.graph_runner=types.SimpleNamespace(model=e.model,run_tree_hidden=model_call)
    pad=m.load_originals()[1];runner_cls=types.SimpleNamespace(_pad_inputs=pad)
    native=types.SimpleNamespace(_next_linear_draft_graph_bucket=m.bucket)
    torch=types.ModuleType('torch');torch.tensor=lambda values,**kw:m.Tensor(values);torch.long='int64';torch.cat=m.TensorOps.cat
    p=lambda b:types.SimpleNamespace(candidate_budget=b)
    old=live159._capability
    try:
      for E,F,U in ((1,1,2),(3,4,8)):
        live159._capability=Capability(frozenset({(1,E)}),frozenset({(E,1)}),mode='qualifying')
        ids=[0]+list(range(32,32+E));sources={i:(p(3),[1,2,3]) for i in ids[1:]}
        with patch.dict(sys.modules,{'torch':torch}):
          plan,predicted=live159.fused_prelude(e,native,runner_cls,[p(4)]+[p(3)]*E,ids,sources,[(0,128,[300],400)],[(i,131,1000+i) for i in ids[1:]],4)
        self.assertEqual((plan.frontier_M,plan.union_M),(F,U));self.assertEqual(predicted,list(range(E)))
        self.assertEqual(calls[-2:],[('model',U),('head',F)])
      corrupt[0]=True
      with patch.dict(sys.modules,{'torch':torch}):
        with self.assertRaisesRegex(RuntimeError,'source ledger changed'):
          live159.fused_prelude(e,native,runner_cls,[p(4)]+[p(3)]*3,[0,32,33,34],{i:(p(3),[1,2,3]) for i in (32,33,34)},[(0,128,[300],400)],[(i,131,1000+i) for i in (32,33,34)],4)
      self.assertNotIn('_release_cache_sequence',vars(e));self.assertNotIn('torch',sys.modules)
    finally:live159._capability=old

if __name__=='__main__':unittest.main()
