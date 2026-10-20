"""Only new P4 fixture/verdict/metadata bindings; explicit HOST carriers."""
import ast,copy,gzip,importlib,json,sys,types,unittest
from pathlib import Path
from unittest.mock import patch
from base169 import P,B,V161,checked,require
from scope169 import Case

def function(path,name,namespace):
 n=copy.deepcopy(next(n for n in ast.parse(checked(path).read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name))
 exec(compile(ast.fix_missing_locations(ast.Module(body=[n],type_ignores=[])),str(path)+'::P169-HOST','exec'),namespace)
 return namespace[name]

class Host(unittest.TestCase):
 def test_actual_retained_prefix_only_and_finite_source_binding(self):
  from provider169 import source_contexts,evidence
  import actor169,draft169,protocol169,target169,next169,source169,runtime169
  case=Case();cs=source_contexts(case);proof=evidence(case)
  self.assertEqual(len(cs),64);self.assertEqual(proof['unique_candidate_token_prefixes'],6)
  f=json.loads((P/'FIXTURE_INPUT169.json').read_text());raw=json.loads(gzip.decompress(Path(f['source']).read_bytes()))
  for i,row in enumerate(f['real_source_prefixes']):
   owner=row['request_index'];old=raw['normal_target_calls'][row['source_target_call']];event=raw['stream_events'][row['source_stream_event']]
   j=old['sequence_ids'][:old['N']].index(owner)
   self.assertEqual((old['budgets'][j],old['contexts'][j]),(4,128));self.assertEqual(len(event['token_ids']),5)
   count=129-len(raw['tokenized_inputs'][owner]);expected=raw['tokenized_inputs'][owner]+raw['generate_results'][owner]['completion_token_ids'][:count]
   self.assertEqual(row['source_token_ids_through_current_root'],expected);self.assertEqual(cs[16+i]['token_ids'],expected)
   self.assertEqual(len(expected),129);self.assertNotIn('observed_tokens',row)
  self.assertIs(actor169.warm_source.__globals__['draft'],draft169)
  self.assertIs(protocol169.commit.__globals__['target'],target169)
  self.assertIs(next169.proof.__globals__['draft'],draft169);self.assertIs(next169.proof.__globals__['epoch'],source169)
  self.assertEqual(runtime169.P,P)
  from budget169 import work_plan
  plan=work_plan(case);self.assertEqual(plan.allocated_selected_candidates,112)
  self.assertEqual(plan.target_limit.check_actual_call([4]*16),80)
  with self.assertRaises(ValueError):plan.target_limit.check_actual_call([4]*32)
  self.assertEqual(draft169.counts(case),[5]+[4]*15+[9]*16+[0]*32)
 def test_actual_original_parent4_verdict_and_original_finish_gate(self):
  import actor169
  from audit169 import original_verdict,Value
  rows=[[11,12,13,14] for _ in range(16)];head=[v for _ in rows for v in (11,12,13,14,99)]+[0]*48
  emitted,accepted=original_verdict(rows,head)
  self.assertTrue(all(r==[11,12,13,14,99] for r in emitted));self.assertTrue(all(r==[0,1,2,3] for r in accepted))
  # Execute the actual pure original tree_primary_path with explicit HOST plan.
  tree_path=checked(Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/tree.py'))
  node=copy.deepcopy(next(n for n in ast.parse(tree_path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='tree_primary_path'))
  ns={'TreeSpeculationPlan':object};exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(tree_path),'exec'),ns)
  module=types.ModuleType('vllm_ascend.spec_decode.pearl.tree');module.tree_primary_path=ns['tree_primary_path']
  def ticket(i,pid,eager):return types.SimpleNamespace(request_index=i,proposal_id=pid,home_batch_id=i%2,gamma=3 if eager else 4,required_prefix_epoch=0,eager=eager,lifecycle=types.SimpleNamespace(value='published'))
  parents={i:ticket(i,i,False) for i in range(16,32)};children={i:ticket(i,100+i,True) for i in range(16,32)}
  payloads={}
  for i in range(16,32):
   payloads[i]=dict(plan=types.SimpleNamespace(candidate_budget=4,parent_indices=Value([-1,0,1,2])),row=rows[i-16])
   payloads[100+i]=dict(eager_dependency_length=4,eager_dependency_tokens=(11,12,13,14),eager_frontier_token=99,eager_parent_proposal_id=i)
  source=dict(controller=types.SimpleNamespace(validate_verification=lambda i:parents[i]),staged=children,payloads=payloads)
  with patch.dict(sys.modules,{'vllm_ascend.spec_decode.pearl.tree':module}):
   gates=actor169.gate_rows(source,emitted,accepted)
   self.assertTrue(all(r['predicted_original_finish_eager_valid'] for r in gates))
   payloads[116]['eager_frontier_token']=98
   bad=actor169.gate_rows(source,emitted,accepted)
   self.assertFalse(bad[0]['predicted_original_finish_eager_valid']);self.assertFalse(bad[0]['gates']['frontier_equal'])
  wrong=list(head);wrong[0]=55;out,bad=original_verdict(rows,wrong)
  self.assertEqual(out[0][0],55);self.assertEqual([v for v in bad[0] if v>=0],[])
 def test_actual_original_native_PA_union_metadata_parent4_position_and_negatives(self):
  ns=dict(B=B,checked=checked,copy=copy,sys=sys,types=types,importlib=importlib,Path=Path,ast=ast,gate=types.SimpleNamespace(require=require))
  make=function(V161/'test_host161.py','fixture',ns);m,cache,allocate,PA=make();engine=allocate()
  import metadata169
  from repair_frontier_fusion import AllocationWitness,plan_frame,build_union
  case=Case();f=dict(sources={i:(None,[10,20,30,1000+i]) for i in range(16,32)},catchups=[[300,400]]+[()]*31)
  frontiers=[(i,132,1000+i) for i in range(16,32)];repairs=[(0,128,[300],400)]
  w=AllocationWitness(id(engine.cache_allocation),128,tuple(tuple(t) for t in engine.cache_allocation.block_tables))
  frame=plan_frame(repairs,frontiers,next_bucket=m.bucket,witness=w)
  actual=build_union(frame,tensor=m.Tensor,prepare=engine._prepare_attention_metadata,pad=m.load_originals()[1],cat=m.TensorOps.cat,lease=lambda:w)
  x,pos,meta=actual
  record=dict(inputs=x,positions=pos,slot_mapping=meta.slot_mapping,context_lens=meta.context_lens,block_tables=meta.block_tables,
   host_Q=meta.actual_seq_lengths_q,host_KV=meta.sequence_lens,use_fused_infer_attention=False,has_attention_mask=False,has_request_tables=False)
  engine._v151_partial={};proof=metadata169.capture_actual(engine,case,f,dict(hidden=[dict(metadata=record)]),PA)
  self.assertTrue(proof['all_fields_exact']);self.assertTrue(proof['own_physical_mapping_exact'])
  self.assertEqual(proof['owners'],list(range(16,32))+[0]);self.assertEqual(proof['logical_positions'],[132]*16+[128])
  self.assertEqual(proof['actual']['host_KV'],[133]*16+[129]+[1]*3)
  wrong=dict(record,slot_mapping=m.Tensor([*meta.slot_mapping.data[:16],meta.slot_mapping.data[0],*meta.slot_mapping.data[17:]]))
  with self.assertRaises(RuntimeError):metadata169.capture_actual(engine,case,f,dict(hidden=[dict(metadata=wrong)]),PA)
  wrong=dict(record,positions=m.Tensor([131,*pos.data[1:]]))
  with self.assertRaises(RuntimeError):metadata169.capture_actual(engine,case,f,dict(hidden=[dict(metadata=wrong)]),PA)
  wrong=dict(record,slot_mapping=m.Tensor([*meta.slot_mapping.data[:17],0,*meta.slot_mapping.data[18:]]))
  with self.assertRaises(RuntimeError):metadata169.capture_actual(engine,case,f,dict(hidden=[dict(metadata=wrong)]),PA)
 def test_actual_head_wrapper_and_parent4_shared4_admission_boundaries(self):
  import observer169,live169,trace169,base169
  from audit169 import Value
  self.assertIs(trace169.metrics.__globals__['sha'],base169.sha)
  self.assertIs(trace169.metrics.__globals__['checked'],base169.checked)
  self.assertEqual((P/'fast169.generated.py').read_text(),live169.fast_source())
  class Snapshot(Value):
   def clone(self):return Snapshot(copy.deepcopy(self.values))
  original=lambda hidden,vocab:Snapshot([11]*hidden.shape[0])
  engine=types.SimpleNamespace(model=types.SimpleNamespace(compute_greedy_tokens=original))
  def indexed(e,method):
   result=method();return result,dict(hidden=[],greedy=[],draft=[],root_head=[])
  with patch.object(observer169,'_indexed_run',indexed):
   output,seen=observer169.run(engine,lambda:engine.model.compute_greedy_tokens(types.SimpleNamespace(shape=(16,1024)),151936))
   self.assertEqual(len(seen['union_head']),1);self.assertEqual(seen['union_head'][0]['M'],16)
   self.assertEqual(seen['union_head'][0]['tokens'].tolist(),[11]*16)
   self.assertIs(engine.model.compute_greedy_tokens,original)
  # Compile the actual generated parent filter; real Native primary-path function.
  tree_path=checked(Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/tree.py'))
  ns={'TreeSpeculationPlan':object};function(tree_path,'tree_primary_path',ns)
  tree_module=types.ModuleType('vllm_ascend.spec_decode.pearl.tree');tree_module.tree_primary_path=ns['tree_primary_path']
  namespace=dict(_v169_parent_budgets={},_v169_parent4_enabled=True)
  node=copy.deepcopy(next(n for n in ast.parse((P/'shape169.generated.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='filter_eager_parents'))
  exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),'<actual-P169-filter>','exec'),namespace)
  tickets={i:types.SimpleNamespace(proposal_id=i) for i in (1,2,3,4)}
  payloads={1:dict(plan=types.SimpleNamespace(candidate_budget=3,parent_indices=Value([-1,0,1]))),
   2:dict(plan=types.SimpleNamespace(candidate_budget=4,parent_indices=Value([-1,0,1,2]))),
   3:dict(plan=types.SimpleNamespace(candidate_budget=4,parent_indices=Value([-1,0,-1,2]))),
   4:dict(plan=types.SimpleNamespace(candidate_budget=5,parent_indices=Value([-1,0,1,2,-1])))}
  with patch.dict(sys.modules,{'vllm_ascend.spec_decode.pearl.tree':tree_module}):
   selected=namespace['filter_eager_parents']([1,2,3,4],types.SimpleNamespace(ready=tickets),payloads,{})
  self.assertEqual(selected,[1,2]);self.assertEqual(namespace['_v169_parent_budgets'],{1:3,2:4,3:4,4:5})
  source=(P/'shape169.generated.py').read_text();assignment=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='kept' for t in n.targets))
  def keep(shared4,normal_count=16):
   env=dict(plan=types.SimpleNamespace(normal_budgets={i:4 for i in range(normal_count)},eager_budgets={1:3,2:3}),
    shared_steps4=shared4,shared4_ready=shared4,_v169_parent_budgets={1:3,2:4})
   exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(assignment)],type_ignores=[])),'<actual-P169-kept>','exec'),env)
   return env['kept']
  self.assertEqual(keep(False),{1:3});self.assertEqual(keep(True),{1:3,2:3})
  self.assertEqual(keep(True,17),{1:3})
  # Execute the actual issuer: P3 retains its original shared4/union domain;
  # the actual shape above independently excludes P4 beyond its local limit.
  with patch.object(live169,'_factory',dict(shape={})),patch.object(live169,'_capability',live169._capability):
   current=types.SimpleNamespace(cache_allocation=None,cache_block_tables=None,
    graph_runner=types.SimpleNamespace(graph_cache_sealed=False))
   live169.activate(current,object(),lambda engine,proof:engine is current)
   self.assertTrue(live169._capability.allows_shared4(17,15))
   self.assertTrue(live169._capability.allows_shared4(32,32))
   self.assertTrue(live169._capability.allows_union(32,32))
  self.assertEqual(Case(repair_count=0).union_M,16);self.assertEqual(Case(repair_count=16).union_M,32)
  self.assertEqual(Case(repair_count=16).repair_vector,(1,)*16+(0,)*48)
