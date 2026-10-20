"""Small actual-provider/isolated-binding/original-predicate HOST regressions."""
import ast,copy,json,sys,unittest
from collections import Counter
from types import SimpleNamespace as NS
from unittest.mock import patch
from scope154 import P,P151,P152,case,require
import actor154,draft154,protocol154,next154,source154,target154,runtime154
from provider154 import fixture,source_contexts

TREE='vllm_ascend.spec_decode.pearl.tree'
def source():
 parents={};children={};payloads={}
 for owner in range(32,64):
  def ticket(number,eager):return NS(proposal_id=number,request_index=owner,home_batch_id=owner%2,gamma=3,
    required_prefix_epoch=int(eager),eager=eager,lifecycle=NS(value='staged-eager' if eager else 'available'))
  parent=ticket(2*owner,False);child=ticket(2*owner+1,True);parents[owner]=parent;children[owner]=child
  payloads[parent.proposal_id]=dict(ticket=parent,plan=NS(candidate_budget=3),row=[20,21,22],confidence=.5)
  payloads[child.proposal_id]=dict(ticket=child,eager_parent_proposal_id=parent.proposal_id,
    eager_dependency_length=3,eager_dependency_tokens=(20,21,22),eager_frontier_token=99)
 controller=NS(staged_eager=children,validate_verification=lambda owner:parents[owner])
 return dict(controller=controller,parent_tickets=parents,staged=children,payloads=payloads,logical={})

def original_finish():
 """Extract the actual untouched Native promotion statement interval."""
 from reuse154 import checked,inputs
 path=next(p for p in inputs()['files'] if p.endswith('/pearl/native_engine.py'))
 tree=ast.parse(checked(path).read_text())
 method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_generate_spec_rhythm_tree_decode')
 def assigned(n,name):return isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)
 body=next(n.body for n in ast.walk(method) if isinstance(n,(ast.For,ast.If,ast.Try))
  and any(assigned(v,'accepted_count') for v in n.body) and any(assigned(v,'promoted') for v in n.body))
 first=next(i for i,n in enumerate(body) if assigned(n,'accepted_count'));last=next(i for i,n in enumerate(body) if assigned(n,'promoted'))
 unit=ast.parse('def finish(self,controller,index,tree_plan,target_payloads,payloads,output_rows,accepted_rows,committed_tokens,will_finish,counters):\n row=0\n return None\n')
 unit.body[0].body=unit.body[0].body[:1]+copy.deepcopy(body[first:last+1])+ast.parse('return promoted').body
 ns={};exec(compile(ast.fix_missing_locations(unit),str(path)+'::V154-HOST-original-finish','exec'),ns);return ns['finish']

class Checks(unittest.TestCase):
 def test_actual_prefix_provider_and_no_future_input(self):
  values=source_contexts(case());data=fixture()['real_source_prefixes']
  self.assertEqual(len(values),64)
  for owner in range(32,48):
   row=data[(owner-32)%8]
   self.assertEqual(values[owner]['token_ids'],row['prompt_token_ids']+row['actual_previous_completion_token_ids'])
   self.assertEqual(len(values[owner]['token_ids']),129)
  self.assertEqual(len({tuple(values[i]['token_ids']) for i in range(32,48)}),5)
  self.assertEqual(len({v['request_id'] for v in data}),8)
  self.assertTrue(all(values[i]['V154_source_kind']=='original-P151-negative-context' for i in (*range(32),*range(48,64))))
  self.assertTrue(all(not any('next_delivered' in k or 'available_actual' in k for k in row) for row in data))
 def test_real_partial_repair_contexts(self):
  c=case();rows=draft154.contexts(c)
  self.assertEqual(c.repair_vector,(1,)+(0,)*63)
  self.assertEqual([len(v['token_ids']) for v in rows],[130]+[129]*63)
  self.assertEqual(draft154.counts(c),[5]+[4]*31+[8]*32)
  self.assertEqual((c.union_M,c.selected),(36,192))
 def test_actual_isolated_reference_bindings(self):
  self.assertIs(next154.proof.__globals__['draft'],draft154)
  self.assertIs(next154.restore_source.__globals__['epoch'],source154)
  self.assertIs(protocol154.commit.__globals__['target'],target154)
  self.assertIs(draft154.independent.__globals__['source_contexts'],source_contexts)
  self.assertNotIn('run',vars(protocol154))
  self.assertEqual(next154._inherited_origin['changes'],[['[case.repair]*32+[0]*32','list(case.repair_vector)',2]])
 def test_actual_commit_hook_receives_unmodified_verdict(self):
  s=source();out=[[20,21,22,99,-1] for _ in range(32)];accepted=[[0,1,2,-1] for _ in range(32)];seen=[]
  class StopHook(Exception):pass
  def hook(a,b):seen.append((a,b));raise StopHook()
  s['_v154_before_parent_commit']=hook
  e=NS(rank=0,topology=NS(target_leader_rank=1),_broadcast_spec_rhythm_tree_verdict=lambda *args:(out,accepted,None))
  frame=([NS(candidate_budget=3)]*32,[],[[20,21,22]]*32,list(range(32,64)),32)
  with patch.dict(sys.modules,{'native_protocol':NS()}):
   with self.assertRaises(StopHook):protocol154.commit(e,s,list(range(32,64)),frame,None,None,{},lambda n,f:f())
  self.assertIs(seen[0][0],out);self.assertIs(seen[0][1],accepted)
 def test_actual_original_finish_predicate_matches_gate_detail(self):
  finish=original_finish()
  variants=[{},dict(accepted=[0,1,-1,-1]),dict(dependency=(20,21,23)),dict(frontier=98),dict(parent_id=-1),dict(frontier=None)]
  for variant in variants:
   s=source();o=[20,21,22,99,-1];a=variant.get('accepted',[0,1,2,-1]);owner=32
   for i in range(32,64):
    payload=s['payloads'][s['staged'][i].proposal_id]
    if 'dependency' in variant:payload['eager_dependency_tokens']=variant['dependency']
    if 'frontier' in variant:payload['eager_frontier_token']=variant['frontier']
    if 'parent_id' in variant:payload['eager_parent_proposal_id']=variant['parent_id']
   controller=s['controller'];controller.finish_verification=lambda index,**kw:s['staged'][index] if kw['fully_accepted'] else None
   engine=NS(_tree_primary_path_length=lambda p:3,config=NS(spec_rhythm_acceptance_ema_alpha=.1))
   result=finish(engine,controller,owner,NS(candidate_budget=3),[s['payloads'][s['parent_tickets'][owner].proposal_id]],s['payloads'],[o],[a],o[:4],False,Counter())
   with patch.dict(sys.modules,{TREE:NS(tree_primary_path=lambda plan:(0,1,2))}):
    gates=actor154.gate_rows(s,[o]*32,[a]*32)
   self.assertEqual(result is not None,gates[0]['predicted_original_finish_eager_valid'])
   self.assertEqual(result is not None,not variant)
 def test_actual_runtime_shell_is_new_scope_and_no_dispatch(self):
  origin=runtime154._inherited_origin
  self.assertEqual(origin['path'],str(P152/'runtime152.py'))
  self.assertIs(runtime154.worker.__globals__['P'],P)
  self.assertTrue(any(old=='from actor152 import run_actor,native as snapshot' and new=='from actor154 import run_actor,native as snapshot' for old,new,n in origin['changes']))
  self.assertIn('check154',runtime154.worker.__code__.co_names)
  self.assertIn('actor154',runtime154.worker.__code__.co_names)
  self.assertNotIn('actor152',runtime154.worker.__code__.co_names)
  self.assertFalse((P/'actual').exists())

if __name__=='__main__':unittest.main()
