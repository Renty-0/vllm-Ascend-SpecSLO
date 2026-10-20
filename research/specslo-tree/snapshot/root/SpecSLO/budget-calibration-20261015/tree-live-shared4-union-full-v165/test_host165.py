"""New bounded full527/metadata/namespace checks, no actual model or backend."""
import ast,copy,importlib.util,json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
from base165 import P,B,V148,V151,V159,V161,checked,require

def function(path,name,namespace):
 node=copy.deepcopy(next(n for n in ast.parse(checked(path).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name))
 exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(path)+'::HOST165','exec'),namespace)
 return namespace[name]

class Host(unittest.TestCase):
 def test_real_native_allocator_full527_lease_and_old_local128_negative(self):
  path=checked(Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/native_cache.py'))
  ns={};exec(compile(path.read_text(),str(path),'exec'),ns)
  cache=ns['NativePrefixCache'](544,32,128)
  allocation=cache.allocate([[1]*128 for _ in range(64)],False,sequence_capacity=527)
  for owner in range(64,102):cache.activate_sequence(owner,[0],enable_prefix_caching=False)
  cache.ensure_capacity(list(range(64)),[136]*64)
  engine=types.SimpleNamespace(cache_allocation=allocation,_cache_storage_sequence_capacity=527,_v151_source_epoch=1)
  import source165
  current=source165.lease(engine)
  self.assertEqual((current['storage'],current['actual_unique_pages']),(527,166))
  scope=dict(require=require,digest=lambda v:None)
  old=function(V151/'source151.py','lease',scope)
  with self.assertRaises(RuntimeError):old(engine)
  self.assertTrue(all(all(v==-1 for v in allocation.block_tables[i]) for i in range(102,527)))
 def test_actual_original_PA_reorder_interior_padding_matches_candidate_all_fields(self):
  import pure165
  ns=dict(B=B,checked=checked,copy=copy,sys=sys,types=types,importlib=importlib,Path=Path,ast=ast,gate=types.SimpleNamespace(require=require))
  make=function(V161/'test_host161.py','fixture',ns)
  m,cache,allocate,PA=make();engine=allocate()
  from repair_frontier_fusion import AllocationWitness,plan_frame,build_union
  engine._v165_partial={}
  frontiers=[(i,131,1000+i) for i in (1,2,3)];repairs=[(0,128,[300],400)]
  bank=types.SimpleNamespace(case=types.SimpleNamespace(eager=3,normal=1),
   parents=dict(frontiers=frontiers,repairs=repairs),math={'PA':PA})
  native=types.SimpleNamespace(_next_linear_draft_graph_bucket=m.bucket)
  torch=types.ModuleType('torch');torch.tensor=lambda values,**kw:m.Tensor(values);torch.long='int64'
  with patch.dict(sys.modules,{'torch':torch}):reference=pure165.independent_frame(engine,bank,native,0)
  w=AllocationWitness(id(engine.cache_allocation),128,tuple(tuple(t) for t in engine.cache_allocation.block_tables))
  plan=plan_frame(repairs,frontiers,next_bucket=m.bucket,witness=w)
  actual=build_union(plan,tensor=m.Tensor,prepare=engine._prepare_attention_metadata,pad=m.load_originals()[1],cat=m.TensorOps.cat,lease=lambda:w)
  for a,b in zip(actual[:2],reference[:2]):self.assertEqual(a.data,b.data)
  for field in ('slot_mapping','context_lens','block_tables'):self.assertEqual(getattr(actual[2],field).data,getattr(reference[2],field).data)
  self.assertEqual(reference[2].actual_seq_lengths_q,(1,2,3,4,5))
  self.assertEqual(reference[2].sequence_lens,(132,132,132,1,129,1,1,1))
  self.assertEqual(reference[2].slot_mapping.data[3],-1);self.assertGreaterEqual(reference[2].slot_mapping.data[4],0)
  record=dict(inputs=actual[0],positions=actual[1],slot_mapping=actual[2].slot_mapping,context_lens=actual[2].context_lens,
   block_tables=actual[2].block_tables,host_Q=actual[2].actual_seq_lengths_q,host_KV=actual[2].sequence_lens,
   use_fused_infer_attention=False,has_attention_mask=False,has_request_tables=False)
  with patch.dict(sys.modules,{'torch':torch}):proof=pure165.actual_metadata_gate(engine,bank,native,record)
  self.assertTrue(all(proof['field_equal'].values()));self.assertTrue(all(proof['own_lease'].values()))
  bad=dict(record,slot_mapping=m.Tensor([0]*8))
  with patch.dict(sys.modules,{'torch':torch}):
   with self.assertRaises(RuntimeError):pure165.actual_metadata_gate(engine,bank,native,bad)
  self.assertFalse(engine._v165_partial['actual_metadata']['own_lease']['dummy_slots'])
 def test_real_isolated_natural_and_next_functions_use_new_full_epoch_everywhere(self):
  import source165,draft165,target165,protocol165,next165,natural165
  self.assertIs(draft165.prepare.__globals__['epoch'],source165)
  self.assertIs(target165.rebuild_history.__globals__['epoch'],source165)
  self.assertIs(protocol165.commit.__globals__['target'],target165)
  self.assertIs(next165.proof.__globals__['draft'],draft165)
  self.assertIs(next165.restore_source.__globals__['epoch'],source165)
  self.assertIs(natural165.run_actor.__globals__['draft'],draft165)
  self.assertIs(natural165.run_actor.__globals__['protocol'],protocol165)
  text=checked(B/'tree-natural-positive-union-discovery-local-v154/actor154.py').read_text()
  for old,new,count in natural165._inherited165['changes']:
   self.assertEqual(text.count(old),count);text=text.replace(old,new)
  self.assertIn('import next165 as next154',text)
  self.assertIn("report['lease_error_status']=lease_error_status",text)
  self.assertEqual(len(draft165._inherited165['changes']),4)
  from audit_component165 import original_verdict
  output,accepted=original_verdict([[10,11,12],[20,21,22]],[10,11,12,13,99,21,22,23])
  self.assertEqual(output,[[10,11,12,13,-1],[99,-1,-1,-1,-1]])
  self.assertEqual(accepted,[[0,1,2,-1],[-1,-1,-1,-1]])
  with self.assertRaises(ValueError):original_verdict([[10,11,12]],[10,11,12])
 def test_actual_observer_and_original_inspector_cover_interior_and_real_repair(self):
  from dataclasses import replace
  import observer165,source165
  from base165 import bind_source_epoch
  ns=dict(B=B,checked=checked,copy=copy,sys=sys,types=types,importlib=importlib,Path=Path,ast=ast,gate=types.SimpleNamespace(require=require))
  m,cache,allocate,PA=function(V161/'test_host161.py','fixture',ns)();e=allocate()
  from repair_frontier_fusion import AllocationWitness,plan_frame,build_union
  T=m.Tensor
  T.__len__=lambda value:len(value.data)
  T.numel=lambda value:len(value.data)
  T.detach=lambda value:value;T.cpu=lambda value:value
  T.clone=lambda value:T(value.data,value.dtype,value.device)
  T.__ge__=lambda value,other:T([v>=other for v in value.data])
  T.sum=lambda value:T(sum(value.data))
  T.item=lambda value:value.data
  T.all=lambda value:T(all(value.data))
  w=AllocationWitness(id(e.cache_allocation),128,tuple(tuple(t) for t in e.cache_allocation.block_tables))
  plan=plan_frame([(0,128,[300],400)],[(i,131,1000+i) for i in (1,2,3)],next_bucket=m.bucket,witness=w)
  x,pos,meta=build_union(plan,tensor=T,prepare=e._prepare_attention_metadata,pad=m.load_originals()[1],cat=m.TensorOps.cat,lease=lambda:w)
  e.graph_runner=types.SimpleNamespace(run_tree_hidden=lambda *args:T([[7]]*8),run_target_greedy=lambda *args:None,run_draft_greedy=lambda *args:None)
  torch=types.ModuleType('torch');torch.tensor=lambda values,**kw:T(values);torch.long='int64';torch.isfinite=lambda value:T([True]*len(value.data))
  epoch=bind_source_epoch();self.assertIs(epoch.read,source165.read)
  def read_actual_lease_prefix0(engine,slots):
   self.assertEqual(slots.data,[meta.block_tables.data[0][0]*128]*4)
   return tuple(T([5]*4) for _ in range(56)) # Explicit HOST values only.
  with patch.dict(sys.modules,{'torch':torch}),patch.object(epoch,'read',read_actual_lease_prefix0):
   result,records=observer165.run(e,lambda:e.graph_runner.run_tree_hidden(x,pos,meta))
   witness=records['hidden'][0]['padding']
   self.assertEqual(witness['actual_physical_real_indices'],[0,1,2,4]);self.assertEqual(witness['actual_physical_dummy_indices'],[3,5,6,7])
   self.assertEqual(witness['diagnostic_row_permutation'],[0,1,2,4,3,5,6,7]);self.assertEqual(len(witness['prefix0_KV']),56)
   self.assertEqual(records['hidden'][0]['metadata']['slot_mapping'].data,meta.slot_mapping.data)
   import padding_reference
   with self.assertRaises(RuntimeError):padding_reference.inspect(e,x,pos,meta,4) # Actual old boundary fails on row4.
   badx=T(x.data);badx._data[3]=99
   with self.assertRaises(RuntimeError):observer165.run(e,lambda:e.graph_runner.run_tree_hidden(badx,pos,meta))
   badslots=T(meta.slot_mapping.data);badslots._data[4]=0
   wrong=replace(meta,slot_mapping=badslots)
   with self.assertRaises(RuntimeError):observer165.run(e,lambda:e.graph_runner.run_tree_hidden(x,pos,wrong))
   self.assertEqual(meta.slot_mapping.data[3],-1);self.assertGreaterEqual(meta.slot_mapping.data[4],0)

 def test_finite_production_domain_and_uncapped64_do_not_reinterpret_budgets(self):
  from component165 import qualifying_capability,production_capability
  from controls165 import window,windows,adapt_source,parse_windows
  q=qualifying_capability();p=production_capability()
  self.assertEqual(q.union_pairs,frozenset({(1,1),(3,1),(32,1),(32,32)}))
  self.assertEqual((len(p.shared4_pairs),len(p.union_pairs)),(1024,1024))
  self.assertEqual(parse_windows('64,32,16'),(64,32,16))
  self.assertEqual(adapt_source('complete reviewed original source',64),'complete reviewed original source')
  for bad in (True,0,8,65):
   with self.assertRaises(RuntimeError):window(bad)
  # Full4 target/root semantics and original parent3/child3 selection stay
  # original; this test establishes only explicit HOST source domains.
  source=(P/'pure165.py').read_text()
  self.assertIn('full_physical_encoded',source);self.assertIn('all56_written_KV',source)
  independent=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='independent_frame')
  self.assertNotIn('build_union',ast.unparse(independent));self.assertIn("bank.math['PA']",ast.unparse(independent))

if __name__=='__main__':unittest.main()
