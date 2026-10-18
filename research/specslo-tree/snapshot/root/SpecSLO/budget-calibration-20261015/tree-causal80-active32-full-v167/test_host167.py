"""Seven new HOST regressions; actual source bodies, no numerical stand-ins."""
import ast,copy,gzip,importlib.util,json,math,shutil,statistics,sys,tempfile,types,unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from base167 import P,B,BASE,ACTIVE,sha,read,require,checked

@contextmanager
def modules(values):
 old={k:sys.modules.get(k) for k in values}
 for k,v in values.items():sys.modules[k]=v
 try:yield
 finally:
  for k,v in old.items():
   if v is None:sys.modules.pop(k,None)
   else:sys.modules[k]=v

def selected(tree,name):
 return next(n for n in ast.parse(tree).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)

class Tests(unittest.TestCase):
 def test01_actual_dimension_adapters_and_metadata_frame(self):
  import draft167 as draft,target167 as target,bank167 as bank
  new,old=bank.cases();self.assertEqual([new['N'],new['Q'],new['M'],new['S']],[16,80,80,16])
  self.assertEqual([old['M'],old['S'],sum(old['dummy_lengths'])],[128,38,48])
  for i in range(16):self.assertEqual(new['owners'][i]['reference_owner'],i+16)
  def plan(*args,**kwargs):return NS(candidate_budget=kwargs['candidate_budget'])
  tree=types.ModuleType('vllm_ascend.spec_decode.pearl.tree');tree.cached_cpu_tree_speculation_plan=plan
  work=dict(plans=[NS(candidate_budget=4)]*16,source=dict(contexts=[dict(token_ids=[123])] *16),rows=[[1,2,3,4]]*16)
  with modules({'vllm_ascend.spec_decode.pearl.tree':tree}):
   for case in (new,old):
    for ref in (False,True):
     pp,rr,dd,ss=target.frame(case,work,reference=ref)
     self.assertEqual(len(pp),case['S']);self.assertEqual(sum(p.candidate_budget+1 for p in pp),case['M'])
     self.assertEqual(ss[:16],list(range(16,32)) if ref else list(range(16)))
     self.assertEqual(ss[16:],list(range(485,507)) if ref and case is old else list(range(463,485)) if case is old else [])
  self.assertEqual(target.qualify.__globals__,vars(target));self.assertIs(draft.generate.__globals__,vars(draft))
  self.assertTrue(any(v.get('cold')==3 for v in target._dimension_adapter['records']))
  actor=ast.parse((P/'actor167.py').read_text())
  for label,method in [('new80-geometry-open','__enter__'),('new80-geometry-close','__exit__')]:
   calls=[n for n in ast.walk(actor) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='c' and n.args and isinstance(n.args[0],ast.Constant) and n.args[0].value==label]
   self.assertEqual(len(calls),1);self.assertIsInstance(calls[0].args[1],ast.Lambda);self.assertEqual(calls[0].args[1].body.func.attr,method)
  self.assertNotIn('range(32)',ast.unparse(selected(checked(BASE/'draft_v146.py').read_text(),'work_plan')).replace('range(32)','range(16)'))

 def test02_actual_cache_and_adapted_dual_bank(self):
  import bank167 as bank
  path=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_cache.py')
  ns={};exec(compile(checked(path).read_text(),str(path),'exec'),ns);Cache=ns['NativePrefixCache']
  class Engine:
   def __init__(self):self.cache=Cache(544,32,128);self.cache_allocation=None;self.is_draft=True;self.config=NS(kvcache_block_size=128)
   def _allocate_cache(self,prompts,**kwargs):self.cache_allocation=self.cache.allocate(prompts,False,sequence_capacity=527);self.cache_block_tables=object()
   def _activate_cache_sequence(self,i,tokens,**kwargs):self.cache.activate_sequence(i,tokens,enable_prefix_caching=False)
   def _ensure_cache_capacity(self,owners,positions):self.cache.ensure_capacity(owners,positions)
  original=bank.tensor_hash;bank.tensor_hash=lambda value:'HOST-table-hash';fake=types.ModuleType('torch')
  try:
   with modules({'torch':fake}):value=bank.DualBank(Engine(),bank.cases()[0])
   self.assertEqual(value.page_count,108);self.assertEqual(len(value.initial_tables),527)
   owners=list(range(32))+list(range(463,507));pages=[]
   for i in owners:
    positions=range(len(value.prompts[i%16])+23) if i<32 else range(8)
    for pos in positions:self.assertGreaterEqual(value.initial_tables[i][pos//128],0)
    pages.extend(p for p in value.initial_tables[i] if p>=0)
   self.assertEqual(len(pages),len(set(pages)));self.assertEqual(len(pages),108)
  finally:bank.tensor_hash=original

 def test03_actual_mapper_qualification_and_current_issuer(self):
  import mapper167 as m
  e=NS(is_draft=False,graph_runner=NS(target_entries={},disabled_entry_keys=set(),graph_cache_sealed=False),cache_allocation=object(),cache_block_tables=object(),_v167_qualified_component=object())
  c=NS(engine=e,original_envelope=lambda *a:(128,38,[3]*4+[2]*18))
  self.assertEqual(m.dispatch(c,*m.DOMAIN)[:2],(128,38))
  with m.qualification_geometry(e):self.assertEqual(m.dispatch(c,*m.DOMAIN),(80,16,[]))
  self.assertFalse(hasattr(e,'_v167_qualification_geometry'));self.assertEqual(m.dispatch(c,*m.DOMAIN)[:2],(128,38))
  proof=dict(audit=dict(math_independent_qualified=True,old_padding_behavior_equal=False))
  e._v167_actual_causal80=proof;e.cache_allocation=e.cache_block_tables=None
  with self.assertRaises(RuntimeError):m.issue(e,proof)
  del e._v167_causal80_permission
  for key in (m.GREEDY,m.HIDDEN):e.graph_runner.target_entries[key]=NS(runtime_validated=True)
  m.issue(e,proof);self.assertEqual(m.dispatch(c,*m.DOMAIN),(80,16,[]))
  self.assertIs(m.current(e)['evidence'],proof)
  e._v167_actual_causal80=dict(proof)
  with self.assertRaises(RuntimeError):m.dispatch(c,*m.DOMAIN)
  self.assertEqual(m.dispatch(c,8,40,'causal'),c.original_envelope(8,40,'causal'))

 def test04_actual_original_call_cold_three_forward_observer(self):
  import target167 as t
  from collections import Counter
  class Tensor:
   shape=(80,)
   def __init__(self,v=1):self.v=v
   def clone(self):return Tensor(self.v)
  fake=types.ModuleType('torch');fake.equal=lambda a,b:a.v==b.v
  case=dict(M=80,Q=80,S=16,budgets=[4]*16);key=t.target_key(case)
  class Runner:
   def __init__(self):self.target_entries={};self.graph_cache_sealed=False;self.seen=None;self.calls=0
   def _record_execution(self,*a,**kw):return None
   def run_target_greedy(self,value):
    self.calls+=1
    if key not in self.target_entries:self.target_entries[key]=NS(runtime_validated=False)
    if self.seen is not None and self.seen!=value:self.target_entries[key].runtime_validated=True
    self.seen=value
    self._record_execution('target',None,None,entry_key=key)
    self._v146_last_target_key=key
    return (Tensor(value),)
  class Engine:
   def __init__(self):self.graph_runner=Runner();self.is_draft=False;self.draft_vocab_size=151936
   def target_tree_forward(self,plans,roots,rows,*a,**kw):
    full=self.graph_runner.run_target_greedy(roots[0])[0]
    return dict(used_aclgraph=True,used_tree_attention=False,query_count=80,logical_query_count=80,
     num_draft_tokens=[4]*16,target_query_token_ids=full)
  ns=dict(Counter=Counter);tree=ast.parse(checked(BASE/'verification_qualification.py').read_text())
  nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
  exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual-original-autoqualifier','exec'),ns)
  ns['install'](NS(NativeACLGraphRunner=Runner,NativePearlEngine=Engine))
  engine=Engine();original=t.frame;t.frame=lambda *a,**kw:([],[7],[],[])
  try:
   with modules({'torch':fake}):
    self.assertEqual(t.original_call(engine,case,{})[1].shape,(80,))
    self.assertEqual(engine.graph_runner.calls,3)
    self.assertEqual(engine._dynamic_verification_qualification_counts,dict(qualified_entries=1,untimed_extra_forwards=2))
    self.assertEqual(t.original_call(engine,case,{})[1].shape,(80,))
    self.assertEqual(engine.graph_runner.calls,4)
  finally:t.frame=original

 def test05_actual_retained_full463_metrics(self):
  import trace167
  fixture=read(P/'METRICS_FIXTURE167.json');seen=[]
  with tempfile.TemporaryDirectory(prefix='actual-metrics-HOST-',dir=P) as temp:
   for source,expected in fixture['files'].items():
    self.assertEqual(sha(source),expected);q=Path(temp)/Path(source).name;shutil.copyfile(source,q)
    info=dict(path=str(q),sha256=sha(q));result=trace167.metrics(info);raw=json.loads(gzip.decompress(q.read_bytes()))
    if raw['generate_results'] is None:self.assertIsNone(result)
    else:
     self.assertEqual(len(raw['generate_results']),463);self.assertEqual(result['TTFT160']['extra_wall_samples'],463)
     self.assertEqual(result['admission_window160'],32);self.assertEqual(result['actual_peak'],32)
     # Execute the exact original clock function on the actual463 rows.
     source_path=checked(B/'budget-diagnostics/analyze_completed_tree_full463_cost.py')
     ns=dict(math=math,statistics=statistics,CLASSES=('tight','normal','loose'))
     nodes=[copy.deepcopy(n) for n in ast.parse(source_path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ('need','clocks')]
     exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(source_path),'exec'),ns)
     original=ns['clocks'](raw)
     self.assertTrue(all(result[k]==v for k,v in original.items()));seen.append(result)
     badns=dict(vars(trace167));badns.pop('sha')
     trace167.functions(ACTIVE/'metrics160.py',badns,names=('_percentiles','full_metrics'))
     with self.assertRaises(NameError):badns['full_metrics'](info)
    with self.assertRaises(RuntimeError):trace167.metrics(dict(path=str(q),sha256='0'*64))
  self.assertEqual(len(seen),1)

 def test06_raw_chain_verdict_and_variant_classification(self):
  from audit80167 import chain_verdict
  rows=[[10,11,12,13]]*16;head=[10,11,12,13,99]*16
  emitted,paths=chain_verdict(rows,head)
  self.assertEqual(emitted,[[10,11,12,13,99]]*16);self.assertEqual(paths,[[0,1,2,3]]*16)
  changed=list(head);changed[2]=77
  em,pa=chain_verdict(rows,changed);self.assertEqual(em[0],[10,11,77]);self.assertEqual(pa[0],[0,1,-1,-1])
  # A head-only difference can preserve output/acceptance and must remain visible.
  changed=list(head);changed[2]=77;changed[0]=88
  em,pa=chain_verdict(rows,changed);self.assertEqual(em[0],[88]);self.assertEqual(pa[0],[-1]*4)
  self.assertNotEqual(head,changed)

 def test07_actual_new_guard_constructors_and_namespace(self):
  g=B/'budget-diagnostics/v167-root-execution/guard167.py';spec=importlib.util.spec_from_file_location('guard167',g)
  guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
  own=ast.parse(g.read_text());parent=ast.parse(checked(B/'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text())
  targets=[(own,'prepare_only','f',dict(state=dict(local={},external={},evidence={},parent_archive={}),root={},peers={})),
   (parent,'execute_locked','row',dict(tag='HOST167',readiness={}))]
  for tree,function,name,extra in targets:
   fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==function)
   node=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
   ns={**vars(guard),**extra,'sha':lambda path:'0'*64 if Path(path)==guard.CPU else sha(path)};exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)],type_ignores=[])),'actual-new-guardian-constructor','exec'),ns)
   self.assertIs(ns[name]['adopted'],False);self.assertIs(ns[name]['formal_Goodput'],False)
   bad=copy.deepcopy(node);bad.value.keywords.insert(0,ast.keyword(arg='cost_measured',value=ast.Constant(False)))
   with self.assertRaises(TypeError):exec(compile(ast.fix_missing_locations(ast.Module(body=[bad],type_ignores=[])),'old-duplicate-negative','exec'),dict(ns))
  for source in (own,parent):
   for call in ast.walk(source):
    if isinstance(call,ast.Call) and any(k.arg is None and isinstance(k.value,ast.Name) and k.value.id=='SCOPE' for k in call.keywords):
     self.assertFalse(set(guard.SCOPE)&{k.arg for k in call.keywords if k.arg is not None})
  self.assertEqual((guard.MAX_SECONDS,guard.MAX_TAG_BYTES,guard.MAX_AGGREGATE_BYTES),(7200,256*1024**2,512*1024**2))
  self.assertEqual(guard.P,P);self.assertEqual(guard.ACTION,'causal80-active32-full463')
  import runtime167,extension167,scope167
  self.assertEqual(runtime167.P,P);self.assertEqual(runtime167.worker.__globals__['P'],P)
  self.assertIs(runtime167.worker.__globals__,vars(runtime167));self.assertIn('sha',vars(trace167) if 'trace167' in globals() else vars(__import__('trace167')))

if __name__=='__main__':unittest.main()
