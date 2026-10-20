"""Five necessary new HOST checks; explicit tensor stubs, no parent suite."""
import ast,copy,importlib.util,json,types,unittest
from pathlib import Path
import reuse152
from scope152 import P,Case,cases,NORMAL_B,target_shape

class Host(unittest.TestCase):
 def test_partial_repair_owners_real_provider_contexts_and_full_query_counts(self):
  import draft152
  self.assertEqual((len(NORMAL_B),sum(NORMAL_B)),(32,96))
  for c in cases():
   cs=draft152.contexts(c)
   self.assertEqual(c.repair_owners,tuple(range(c.repair_count)))
   self.assertEqual(sum(c.repair_vector),c.repair_count)
   self.assertEqual([len(v['token_ids']) for v in cs],[129+r for r in c.repair_vector])
   self.assertEqual(draft152.counts(c),[4+r for r in c.repair_vector[:32]]+[8]*32)
   self.assertEqual(c.work_budgets,(*NORMAL_B,*([3]*32)))
   self.assertEqual(c.union_M,{0:None,1:36,16:48,32:64}[c.repair_count])
   self.assertEqual(target_shape(c.normal_budgets)['real_Q'],128)
  for n in (-1,2,33,True):
   with self.assertRaises(RuntimeError):Case(n)
 def test_actual_new_union_metadata_buckets_and_original_F_head(self):
  from repair_frontier_fusion import AllocationWitness,plan_frame,build_union,run_union
  # Reuse only explicit list tensor stubs/original Native dataclass+pad AST
  # loaders; do not run the retained prototype test suite.
  path=P.parent/'budget-diagnostics/draft-repair-frontier-fusion-v149-source/check_host.py'
  spec=importlib.util.spec_from_file_location('_v152_explicit_host_tensor_stubs',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
  Meta,pad=m.load_originals();w=AllocationWitness(123456,128,tuple((2*i,2*i+1) for i in range(128)))
  # Execute the REAL frozen metadata producer chain with explicit HOST
  # tensor/allocation stubs. No handwritten metadata factory is the oracle.
  m.Tensor.to=lambda v,device=None,dtype=None:m.Tensor(v.data,dtype or v.dtype,device or v.device)
  m.Tensor.index_select=lambda v,dim,index:m.Tensor([v.data[i] for i in index.data],v.dtype,v.device)
  torch_stub=types.SimpleNamespace(Tensor=m.Tensor,tensor=lambda v,dtype='int64',device='host':m.Tensor(v,dtype,device),long='int64',int32='int32')
  model_metadata=m.actual_function('native_model.py','NativeQwen2ForCausalLM','make_attention_metadata',dict(torch=torch_stub,NativeAttentionMetadata=Meta))
  engine_metadata=m.actual_function('native_engine.py','NativePearlEngine','_prepare_attention_metadata',{})
  slot_mapping=m.actual_function('native_engine.py','NativePearlEngine','_cache_slot_mapping',{})
  tables=[list(t) for t in w.block_tables]
  model=types.SimpleNamespace(layers=[types.SimpleNamespace(self_attn=types.SimpleNamespace(block_table=m.Tensor(tables,'int32'),blocks_per_sequence=2,block_size=128))],embed_tokens=types.SimpleNamespace(weight=types.SimpleNamespace(device='host')))
  model.make_attention_metadata=types.MethodType(model_metadata,model)
  engine=types.SimpleNamespace(cache_allocation=types.SimpleNamespace(block_tables=tables),cache_block_tables=m.Tensor(tables,'int32'),config=types.SimpleNamespace(kvcache_block_size=128),model=model,device='host')
  engine._ensure_cache_capacity=lambda ids,pos:self.assertTrue(all(tables[i][p//128]>=0 for i,p in zip(ids,pos)))
  engine._cache_slot_mapping=types.MethodType(slot_mapping,engine)
  engine._prepare_attention_metadata=types.MethodType(engine_metadata,engine)
  engine.graph_runner=types.SimpleNamespace(_pad_inputs=pad)
  prepare=engine._prepare_attention_metadata
  # The actual independent PA function calls one original producer for all
  # frontier+repair owners, without candidate plan_frame/build_union reuse.
  pa=copy.deepcopy(next(n for n in ast.parse((P/'draft152.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='PA'))
  pa.body=[n for n in pa.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
  ns=dict(torch=torch_stub);exec(compile(ast.fix_missing_locations(ast.Module(body=[pa],type_ignores=[])),'actual-P152-reference-PA-HOST-stubs','exec'),ns)
  fronts=[(i,131,800+i) for i in range(32,64)]
  for R,U in ((1,36),(16,48),(32,64)):
   repairs=[(i,128,[300+i],400+i) for i in range(R)]
   plan=plan_frame(repairs,fronts,next_bucket=m.bucket,witness=w)
   self.assertEqual((plan.E,plan.R,plan.frontier_M,plan.union_M),(32,R,32,U))
   x,p,meta=build_union(plan,tensor=m.Tensor,prepare=prepare,pad=pad,cat=m.TensorOps.cat,lease=lambda:w)
   self.assertEqual(meta.actual_seq_lengths_q,tuple(range(1,32+R+1)))
   self.assertEqual(meta.sequence_lens,(132,)*32+(129,)*R+(1,)*(U-32-R))
   self.assertEqual(meta.slot_mapping.data[32+R:],[-1]*(U-32-R))
   ref_x,ref_p,ref_meta=ns['PA'](engine,list(range(32,64))+list(range(R)),[131]*32+[128]*R,[800+i for i in range(32,64)]+[300+i for i in range(R)],U)
   self.assertEqual((x.data,p.data),(ref_x.data,ref_p.data))
   for field in ('slot_mapping','context_lens','block_tables'):self.assertEqual(getattr(meta,field).data,getattr(ref_meta,field).data)
   self.assertEqual((meta.actual_seq_lengths_q,meta.sequence_lens),(ref_meta.actual_seq_lengths_q,ref_meta.sequence_lens))
   calls=[]
   model=types.SimpleNamespace(compute_greedy_tokens=lambda h,v:(calls.append(('head',h.shape[0])) or m.Tensor(list(range(h.shape[0])))))
   runner=types.SimpleNamespace(model=model,run_tree_hidden=lambda *a:(calls.append(('model',a[0].shape[0])) or m.Tensor([[1,2]]*U)))
   self.assertEqual(run_union(plan,(x,p,meta),runner=runner,model=model,vocabulary_size=151936,lease=lambda:w).shape,(32,))
   self.assertEqual(calls,[('model',U),('head',32)])
   with self.assertRaises(ValueError):build_union(plan,tensor=m.Tensor,prepare=prepare,pad=pad,cat=m.TensorOps.cat,lease=lambda:None)
  self.assertIsNone(plan_frame([],fronts,next_bucket=m.bucket,witness=w))
  self.assertIsNone(plan_frame([(0,128,[1,2],3)],fronts,next_bucket=m.bucket,witness=w))
  with self.assertRaises(ValueError):plan_frame([(32,128,[1],2)],fronts,next_bucket=m.bucket,witness=w)
 def test_new_union_reference_and_next_history_actual_namespace(self):
  import draft152,next152,protocol152,target152,source152
  tree=ast.parse((P/'draft152.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='independent')
  text=ast.unparse(fn)
  self.assertIn('union_call = PA(e, ids, positions, tokens, case.union_M)',text)
  self.assertIn('union_hidden = e.model(*union_call).clone()',text)
  self.assertIn('compute_greedy_tokens(union_hidden[:32]',text)
  self.assertNotIn('plan_frame',text.replace('#',''))
  self.assertNotIn('build_union',text)
  self.assertIs(next152.proof.__globals__['draft'],draft152)
  self.assertIs(next152.restore_source.__globals__['epoch'],source152)
  self.assertIs(protocol152.run.__globals__['target'],target152)
  self.assertEqual(next152._inherited_function_origin['isolated_changes'],[['[case.repair]*32+[0]*32','list(case.repair_vector)',2]])
  self.assertEqual(protocol152._inherited_function_origin['isolated_changes'],[['import next151','import next152 as next151',1]])
  parent_tree=ast.parse(reuse152.checked(reuse152.Q/'next151.py').read_text())
  self.assertTrue(any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='draft' and n.func.attr=='independent' for n in ast.walk(parent_tree)))
 def test_actual_protocol_import_binding_and_tail_call(self):
  import sys
  original=reuse152.checked(reuse152.Q/'protocol151.py').read_text()
  actual=original.replace('import next151','import next152 as next151')
  fn=next(n for n in ast.parse(actual).body if isinstance(n,ast.FunctionDef) and n.name=='run')
  imp=next(n for n in fn.body if isinstance(n,ast.Import) and any(a.name=='next152' for a in n.names))
  call=next(n for n in fn.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='nextproof' for t in n.targets))
  def compile_tail(import_node):
   unit=ast.parse('def tail(e,case,trace,source,stage):\n pass\n')
   unit.body[0].body=[copy.deepcopy(import_node),copy.deepcopy(call),ast.Return(ast.Name(id='nextproof',ctx=ast.Load()))]
   ns={};exec(compile(ast.fix_missing_locations(unit),'actual-protocol-run-import-and-nextproof-tail','exec'),ns)
   return ns['tail']
  calls=[];stub=types.ModuleType('next152')
  stub.proof=lambda e,c,t,l,stage:(calls.append((e,c,t,l,stage('proof',lambda:11))) or 'new-union-nextproof')
  missing=object();prior=sys.modules.get('next152',missing);sys.modules['next152']=stub
  try:
   stage=lambda name,fn:fn();source=dict(logical={'actual':1})
   self.assertEqual(compile_tail(imp)('engine','case','trace',source,stage),'new-union-nextproof')
   self.assertEqual(calls,[('engine','case','trace',source['logical'],11)])
   # The exact superseded import without its alias must fail on the same
   # actual next151.proof expression. It is not silently supplied as a global.
   with self.assertRaises(NameError):compile_tail(ast.Import(names=[ast.alias(name='next152')]))('engine','case','trace',source,stage)
  finally:
   if prior is missing:del sys.modules['next152']
   else:sys.modules['next152']=prior
 def test_actual_TTU_snapshot_schema_has_real_families_and_no_dynamic_claim(self):
  from draft152 import _TTU
  entry=types.SimpleNamespace(actual_seq_lengths_q=((1,2),),sequence_lens=((132,129),),tasks=[None]*28,runtime_validated=True)
  runner=types.SimpleNamespace(task_update_skip_replay_count=7,target_entries={('target-hidden:0|steps:1|paged',(36,)):entry},draft_entries={})
  e=types.SimpleNamespace(graph_runner=runner,graph_metrics=lambda:dict(aclgraph_target_task_update_replays=2,aclgraph_target_task_update_tasks=56,aclgraph_task_update_skipped_replays=7,unrelated=3))
  row=_TTU(e);self.assertEqual(row['counters']['actual_task_update_skipped_replays'],7)
  self.assertNotIn('unrelated',row['counters'])
  self.assertEqual(next(iter(row['entries'].values()))['sequence_lens'],[[132,129]])
  self.assertEqual(next(iter(row['entries'].values()))['tasks'],28)
  from scope152 import source_contract
  self.assertFalse(source_contract()['dynamic_prefix_cost_enabled']);self.assertFalse(source_contract()['dynamic_TTU_proved'])
 def test_exact_selected_source_delta_and_no_borrowed_qualification(self):
  from check152 import source_contract
  row=source_contract();self.assertFalse(row['qualification_inherited'])
  self.assertEqual(row['selected_external_files'],174)
  import actor152
  self.assertEqual(actor152._inherited_function_origin['isolated_changes'],[["r['case']['repair']","r['case']['repair_count']",1]])
  text=(P/'runtime152.py').read_text();self.assertIn('cases=4,complete_zeros=2,timings=0',text)
  self.assertIn('require_ready(identity)',text)
  self.assertLess(text.index('require_ready(identity)'),text.index('import torch'))

if __name__=='__main__':unittest.main()
