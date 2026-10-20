"""Execute original adapted bank operations with explicit HOST Native boundaries."""
import ast,builtins,copy,dis,hashlib,importlib.util,json,math,struct,sys,tempfile,types,unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS
from base179 import P,B,checked,sha
class Tensor:
 """Storage/hash/index carrier; never a model or numerical oracle."""
 def __init__(self,values):
  def flat(v):
   for x in v:
    if isinstance(x,(list,tuple)):yield from flat(x)
    else:yield x
  self.values=list(flat(values))
 def contiguous(self):return self
 def view(self,*a):return self
 def cpu(self):return self
 def numpy(self):return self
 def tobytes(self):return b''.join(struct.pack('<d',float(v)) for v in self.values)
 def clone(self):return Tensor(self.values)
 def index_select(self,dim,index):
  assert dim==0;return Tensor([self.values[int(i)] for i in index.values])
class Storage:
 def __init__(self,layer):self.layer=layer;self.values={}
 def flatten(self,*a):return self
 def data_ptr(self):return id(self)
 def value(self,i):return self.values.get(i,float(-1000-self.layer))
 def index_select(self,dim,index):
  assert dim==0;return Tensor([self.value(int(i)) for i in index.values])
 def index_copy_(self,dim,index,value):
  assert dim==0 and len(index.values)==len(value.values)
  self.values.update(zip(map(int,index.values),value.values,strict=True));return self
@contextmanager
def torch_carrier():
 assert 'torch' not in sys.modules;m=types.ModuleType('torch');m.HOST_ONLY_EXPLICIT_CARRIER=True
 m.uint8='uint8';m.long='long';m.tensor=lambda v,**kw:Tensor(v);m.cat=lambda vv:Tensor([v.values for v in vv])
 m.isfinite=lambda v:NS(all=lambda:all(math.isfinite(float(x)) for x in v.values));sys.modules['torch']=m
 try:yield m
 finally:assert sys.modules.pop('torch') is m
def engine():
 path=checked(Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_cache.py'))
 ns={};exec(compile(path.read_text(),str(path),'exec'),ns);Cache=ns['NativePrefixCache']
 class Engine:
  def __init__(self):
   self.cache=Cache(544,32,128);self.is_draft=True;self.device='HOST';self.config=NS(kvcache_block_size=128)
   self.cache_allocation=None;self.cache_block_tables=None;self._cache_storage_sequence_capacity=527
   self._tree_graph_scratch_sequence_ids=tuple(range(463,527));self._mixed_target_scratch_sequence_ids=()
   self.arrays=[Storage(i) for i in range(56)];self.model=NS(layers=[NS(self_attn=NS(key_cache=self.arrays[2*i],value_cache=self.arrays[2*i+1])) for i in range(28)])
   self.prefills=[]
  def tables(self):self.cache_block_tables=Tensor(self.cache_allocation.block_tables)
  def _allocate_cache(self,prompts,**kw):
   assert kw==dict(enable_prefix_caching=False,reserve_sequence_capacity=True)
   self.cache_allocation=self.cache.allocate(prompts,False,sequence_capacity=527);self.tables()
  def _activate_cache_sequence(self,i,tokens,**kw):
   assert kw==dict(enable_prefix_caching=False);self.cache.activate_sequence(i,tokens,enable_prefix_caching=False);self.tables()
  def _ensure_cache_capacity(self,ids,positions):self.cache.ensure_capacity(ids,positions);self.tables()
  def _run_packed_hidden(self,tokens,ids,positions,**kw):
   assert len(tokens)==len(ids)==len(positions)==128 and len(set(ids))==1
   assert positions==list(range(128)) and kw==dict(use_aclgraph=False,use_fused_infer_attention=True)
   owner=ids[0];table=self.cache_allocation.block_tables[owner]
   for token,pos in zip(tokens,positions,strict=True):
    slot=table[pos//128]*128+pos%128
    for i,a in enumerate(self.arrays):a.values[slot]=float(token+i/64)
   self.prefills.append(dict(owner=owner,tokens=list(tokens),positions=list(positions)));return Tensor(tokens)
  def _release_cache(self):self.cache.release();self.cache_allocation=None;self.cache_block_tables=None
 return Engine()
def census(fn):
 names=[]
 def walk(code):
  names.extend(i.argval for i in dis.get_instructions(code) if i.opname=='LOAD_GLOBAL')
  for x in code.co_consts:
   if isinstance(x,types.CodeType):walk(x)
 walk(fn.__code__);return sorted(set(names)-set(fn.__globals__)-set(vars(builtins)))
class Tests(unittest.TestCase):
 def test01_actual_bank_three_endpoints_prefill_hash_guard_and_restore(self):
  import bank179 as b,pure179
  with torch_carrier():
   old=dict(vars(b));old.pop('hashlib');nodes=[copy.deepcopy(n) for n in ast.parse(b.source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
   exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),'actual-old-missing-bank-global','exec'),old)
   bad=engine()
   with self.assertRaisesRegex(NameError,'hashlib'):old['build_bank'](bad,pure179.CASES[0],False,lambda n,f:f())
   self.assertEqual(bad.prefills,[]);bad._release_cache()
   for case in pure179.CASES:
    e=engine();stages=[]
    def stage(name,fn):stages.append(name);return fn()
    bank,proof=b.build_bank(e,case,False,stage)
    self.assertEqual(len(e.prefills),2*case.count);self.assertEqual(len(bank.pages),4*case.count)
    self.assertEqual([x['owner'] for x in e.prefills],list(case.candidate_ids+case.reference_ids))
    for i,row in enumerate(e.prefills):self.assertEqual(row['tokens'],bank.contexts[i%case.count]['token_ids'][:128])
    self.assertEqual(proof['device_table_hash'],hashlib.sha256(e.cache_block_tables.tobytes()).hexdigest())
    self.assertEqual(len(proof['prefix']['KV']),56);self.assertEqual(proof['prefix']['tokens'],b.digest(bank.prefixes))
    self.assertEqual(proof['plan'],b.owner_plan(case));self.assertFalse(proof['target_acceptance_observed'])
    self.assertEqual(stages[0],'allocate-independent-real-banks');self.assertEqual(stages[-1],'complete-independent-prefixes-and-leases')
    bank.capture_initial_queries()
    for offset in (0,64):
     indices=bank.query_indices(offset);before=[v.values[:] for v in bank.query(offset)]
     for a in e.arrays:a.index_copy_(0,indices,Tensor([999999]*case.count))
     bank.restore_queries(offset);self.assertEqual([v.values for v in bank.query(offset)],before)
    bank.protected_histories=copy.deepcopy(bank.prefixes);bank.protected=bank.prefix_hash(bank.protected_histories);bank.guard_full()
    old_table=e.cache_block_tables.values[0];e.cache_block_tables.values[0]+=1
    with self.assertRaisesRegex(RuntimeError,'device tables'):bank.guard_full()
    e.cache_block_tables.values[0]=old_table
    first=bank.kv['slots'](e,case.candidate_ids[0],[0]).values[0];a=e.arrays[0];old_value=a.value(first);a.values[first]+=1
    with self.assertRaisesRegex(RuntimeError,'prefixes'):bank.guard_full()
    a.values[first]=old_value;bank.guard_full();e._release_cache()
   bank,proof=b.build_bank(NS(is_draft=False),pure179.CASES[0],False,lambda n,f:f())
   self.assertIsNone(bank);self.assertIsNone(proof)
 def test02_actual_bound_global_census_and_complete_source_inverse(self):
  import bank179 as b,check179,pure179,runtime179,trace179
  self.assertEqual(check179.verify_parent_projection()['projected_definitions'],29)
  functions=[b.tensor_hash,b.finite,b.build_bank,b.build_parents]+[v for v in vars(b.Bank).values() if isinstance(v,types.FunctionType)]
  functions+=list(b.kv_helpers().values())+list(b.math_helpers().values())
  for fn in functions:
   if isinstance(fn,types.FunctionType):self.assertEqual(census(fn),[],fn.__qualname__)
  old=dict(vars(b));old.pop('hashlib');self.assertEqual(census(types.FunctionType(b.tensor_hash.__code__,old)),['hashlib'])
  self.assertIs(runtime179.worker.__globals__,vars(runtime179));self.assertIs(trace179.metrics.__globals__['sha'],sha)
  self.assertEqual([(c.normal,c.eager) for c in pure179.CASES],[(1,1),(1,3),(32,32)])
  self.assertEqual((P/'FIXTURE_INPUT179.json').read_bytes(),(B/'tree-linear-parent4-child3-local-v169/FIXTURE_INPUT169.json').read_bytes())
 def test03_actual_guard_rows_and_frozen_storage_binding(self):
  G=B/'budget-diagnostics/v179-root-execution';p=G/'guard179.py';spec=importlib.util.spec_from_file_location('_V179_HOST_guard',p)
  guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)
  self.assertEqual(sha(guard._storage_path),json.loads((P/'INPUTS179.json').read_text())['frozen_storage_helper']['sha256'])
  for tree,name,target,extra in [(ast.parse(p.read_text()),'prepare_only','f',dict(state=dict(local={},external={},evidence={},parent_archive={}),root={},peers={})),
   (ast.parse(checked(B/'budget-diagnostics/v154-root-execution/supervise_v154.py').read_text()),'execute_locked','row',dict(tag='HOST179',readiness={}))]:
   fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
   node=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==target for t in n.targets))
   ns={**vars(guard),**extra,'sha':lambda q:'0'*64 if Path(q)==guard.CPU else sha(q)}
   exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)],type_ignores=[])),'actual179-row','exec'),ns)
   self.assertEqual(ns[target]['active_windows_requested'],[32]);self.assertFalse(ns[target]['adopted'])
   bad=copy.deepcopy(node);bad.value.keywords.insert(0,ast.keyword(arg='cost_measured',value=ast.Constant(False)))
   with self.assertRaises(TypeError):exec(compile(ast.fix_missing_locations(ast.Module(body=[bad],type_ignores=[])),'bad-duplicate','exec'),dict(ns))
  root=P/'actual';root.mkdir(exist_ok=True);from unittest.mock import patch
  stat=Path.stat
  with tempfile.TemporaryDirectory(dir=root,prefix='HOST-storage-') as folder:
   folder=Path(folder);stable=folder/'kept.json';stable.write_bytes(b'123');vanish=folder/'entry.tmp';vanish.write_bytes(b'tmp')
   def racing(p,*a,**kw):
    if p==vanish and kw.get('follow_symlinks',True):
     try:p.unlink()
     except FileNotFoundError:pass
    return stat(p,*a,**kw)
   with patch.object(Path,'stat',racing):self.assertEqual(guard.bytes_under(folder),3)
   link=folder/'link.tmp';link.symlink_to(stable)
   with self.assertRaises(RuntimeError):guard.bytes_under(folder)
  self.assertEqual((guard.MAX_SECONDS,guard.MAX_TAG_BYTES,guard.MAX_AGGREGATE_BYTES),(7200,256*1024**2,512*1024**2))
if __name__=='__main__':unittest.main()
