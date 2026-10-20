"""Four new HOST checks using actual allocator/metadata/padding source bodies."""
import ast,copy,importlib.util,sys,types,unittest
from pathlib import Path
from base161 import P,B,checked
import metadata161 as gate

def fixture():
 fusion_path=checked(B/'budget-diagnostics/draft-repair-frontier-fusion-v149-source/repair_frontier_fusion.py')
 if 'repair_frontier_fusion' not in sys.modules:
  spec=importlib.util.spec_from_file_location('repair_frontier_fusion',fusion_path)
  fusion=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fusion;spec.loader.exec_module(fusion)
 else:gate.require(Path(sys.modules['repair_frontier_fusion'].__file__).resolve()==fusion_path,'actual readonly fusion helper origin')
 path=checked(B/'budget-diagnostics/draft-repair-frontier-fusion-v149-source/check_host.py')
 spec=importlib.util.spec_from_file_location('_V161_explicit_tensor_stubs',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 Meta,pad=m.load_originals()
 m.Tensor.to=lambda v,device=None,dtype=None:m.Tensor(v.data,dtype or v.dtype,device or v.device)
 m.Tensor.index_select=lambda v,dim,index:m.Tensor([v.data[i] for i in index.data],v.dtype,v.device)
 m.Tensor.tolist=lambda v:copy.deepcopy(v.data)
 torch=types.SimpleNamespace(Tensor=m.Tensor,tensor=lambda v,dtype='int64',device='host':m.Tensor(v,dtype,device),long='int64',int32='int32')
 model_metadata=m.actual_function('native_model.py','NativeQwen2ForCausalLM','make_attention_metadata',dict(torch=torch,NativeAttentionMetadata=Meta))
 engine_metadata=m.actual_function('native_engine.py','NativePearlEngine','_prepare_attention_metadata',{})
 slot_mapping=m.actual_function('native_engine.py','NativePearlEngine','_cache_slot_mapping',{})
 path=checked(m.ROOT/'native_cache.py');spec=importlib.util.spec_from_file_location('_V161_actual_native_cache',path)
 cache_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=cache_module;spec.loader.exec_module(cache_module)
 cache=cache_module.NativePrefixCache(544,2,128)
 path=checked(B/'tree-union-shared4-protocol-local-v152/draft152.py')
 pa=copy.deepcopy(next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='PA'))
 pa.body=[n for n in pa.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
 ns=dict(torch=torch);exec(compile(ast.fix_missing_locations(ast.Module(body=[pa],type_ignores=[])),'actual-independent-PA-V161-HOST','exec'),ns)
 def allocate():
  allocation=cache.allocate([[10+i]*129 for i in range(64)],enable_prefix_caching=False)
  tables=allocation.block_tables
  model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(block_table=m.Tensor(tables,'int32'),blocks_per_sequence=2,block_size=128))],embed_tokens=types.SimpleNamespace(weight=types.SimpleNamespace(device='host')))
  model.make_attention_metadata=types.MethodType(model_metadata,model)
  e=types.SimpleNamespace(cache_allocation=allocation,cache_block_tables=m.Tensor(tables,'int32'),config=types.SimpleNamespace(kvcache_block_size=128),model=model,device='host',_v151_partial={})
  def ensure(ids,positions):
   if cache.ensure_capacity(ids,positions):raise AssertionError('all tested actual pages already allocated')
  e._ensure_cache_capacity=ensure;e._cache_slot_mapping=types.MethodType(slot_mapping,e);e._prepare_attention_metadata=types.MethodType(engine_metadata,e)
  e.graph_runner=types.SimpleNamespace(_pad_inputs=pad)
  return e
 return m,cache,allocate,ns['PA']

def observed(frame):
 x,p,m=frame
 return dict(inputs=x,positions=p,slot_mapping=m.slot_mapping,context_lens=m.context_lens,block_tables=m.block_tables,
  host_Q=m.actual_seq_lengths_q,host_KV=m.sequence_lens,use_fused_infer_attention=m.use_fused_infer_attention,
  has_attention_mask=m.attention_mask is not None,has_request_tables=m.request_block_tables is not None)

class Host(unittest.TestCase):
 def test_actual_release_reallocate_metadata_same_semantics_different_pages(self):
  m,cache,allocate,PA=fixture();e=allocate()
  from repair_frontier_fusion import AllocationWitness,plan_frame,build_union
  c=types.SimpleNamespace(normal=32,eager=32,prefix=128,repair_count=1,repair_owners=(0,),union_M=36)
  f=dict(sources={i:(None,[800+i,900+i,1000+i]) for i in range(32,64)},catchups=[(300+i,) for i in range(64)])
  owners,positions,tokens=gate.geometry(c,f)
  w=AllocationWitness(id(e.cache_allocation),128,tuple(tuple(t) for t in e.cache_allocation.block_tables))
  plan=plan_frame([(0,128,[300],400)],[(i,131,1000+i) for i in range(32,64)],next_bucket=m.bucket,witness=w)
  actual=build_union(plan,tensor=m.Tensor,prepare=e._prepare_attention_metadata,pad=m.load_originals()[1],cat=m.TensorOps.cat,lease=lambda:w)
  proof=gate.capture_actual(e,c,f,dict(hidden=[dict(metadata=observed(actual))]),PA)
  before=copy.deepcopy(e.cache_allocation.block_tables);cache.release();fresh=allocate()
  self.assertNotEqual(before,fresh.cache_allocation.block_tables)
  reference=PA(fresh,owners,positions,tokens,36)
  closed=gate.compare_fresh(fresh,c,proof,reference)
  self.assertTrue(closed['logical_metadata_exact']);self.assertFalse(closed['raw_page_ID_equality_required'])
  self.assertNotEqual(proof['actual']['slot_mapping'],closed['reference']['actual']['slot_mapping'])
  self.assertNotEqual(proof['actual']['block_tables'],closed['reference']['actual']['block_tables'])
  self.assertTrue(all(proof['field_equal'].values()));self.assertTrue(all(closed['reference']['own_lease'].values()))
  self.assertEqual(fresh._v151_partial['cross_epoch_union_metadata_semantics_equal'],True)
 def test_wrong_owner_slot_table_and_dummy_are_rejected(self):
  m,cache,allocate,PA=fixture();e=allocate();owners=list(range(32,64))+[0];positions=[131]*32+[128]
  raw=gate.snapshot(*PA(e,owners,positions,list(range(33)),36));tables=e.cache_allocation.block_tables
  for field,row,value in (('slot_mapping',0,0),('block_tables',0,tables[1]),('slot_mapping',35,1)):
   wrong=copy.deepcopy(raw);wrong[field][row]=value
   # Even identical corrupted expected metadata must fail own real lease.
   proof=gate.validate_own(wrong,wrong,tables,owners,positions,128)
   self.assertFalse(proof['own_physical_mapping_exact'])
   with self.assertRaises(RuntimeError):gate.semantic(proof)
  proof=gate.validate_own(raw,raw,tables,[33,32]+owners[2:],positions,128)
  self.assertFalse(proof['own_physical_mapping_exact'])
 def test_every_original_field_retained_and_failure_diagnostic_written(self):
  m,cache,allocate,PA=fixture();e=allocate();owners=list(range(32,64))+[0];positions=[131]*32+[128]
  raw=gate.snapshot(*PA(e,owners,positions,list(range(33)),36))
  for field in gate.FIELDS:
   wrong=copy.deepcopy(raw)
   if isinstance(wrong[field],list):wrong[field]=list(reversed(wrong[field]))
   else:wrong[field]=not wrong[field]
   if wrong[field]==raw[field]:wrong[field]=['wrong']
   proof=gate.validate_own(raw,wrong,e.cache_allocation.block_tables,owners,positions,128)
   self.assertFalse(proof['all_fields_exact'],field)
   with self.assertRaises(RuntimeError):gate.semantic(proof)
  c=types.SimpleNamespace(normal=32,eager=32,prefix=128,repair_count=1,repair_owners=(0,),union_M=36)
  f=dict(sources={i:(None,[1,2,1000+i]) for i in range(32,64)},catchups=[(300+i,) for i in range(64)])
  owners,positions,tokens=gate.geometry(c,f);frame=PA(e,owners,positions,tokens,36);bad=observed(frame);bad['slot_mapping']=m.Tensor([0]*36)
  with self.assertRaises(RuntimeError):gate.capture_actual(e,c,f,dict(hidden=[dict(metadata=bad)]),PA)
  diag=e._v151_partial['actual_union_metadata_diagnostic']
  self.assertFalse(diag['field_equal']['slot_mapping']);self.assertEqual(set(diag['field_equal']),set(gate.FIELDS))
 def test_actual_thin_actor_draft_runtime_bindings_and_inverse(self):
  import draft161,actor161,runtime161
  self.assertIs(actor161.run_actor.__globals__['draft'],draft161)
  self.assertIs(draft161.prepare.__globals__['capture_actual'],gate.capture_actual)
  self.assertIs(draft161.compare.__globals__['compare_fresh'],gate.compare_fresh)
  original=checked(B/'tree-union-shared4-protocol-local-v152/draft152.py').read_text();new=original
  edits=draft161._inherited_origin['changes'];self.assertEqual(len(edits),4)
  for old,value,n in edits:self.assertEqual(new.count(old),n);new=new.replace(old,value)
  for old,value,n in reversed(edits):new=new.replace(value,old)
  self.assertEqual(new,original)
  text=ast.unparse(ast.parse((P/'runtime161.py').read_text()));self.assertIn('actor161',text);self.assertIn('check161',text)
  body=ast.unparse(ast.parse(original));self.assertIn('matcher(actual_KV, ownKV)',body)
  self.assertEqual(runtime161.worker.__globals__['P'],P)

if __name__=='__main__':unittest.main()
