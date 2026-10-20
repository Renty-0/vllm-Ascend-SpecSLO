"""Five small source/math/boundary regressions; no Torch/model/device imports."""
import ast,copy,json,sys,time,types,unittest
from pathlib import Path
from scope151 import P,Case,NORMAL_B,MC2_ROWS,cases,require,target_shape

class Host(unittest.TestCase):
 def test_exact_scope_and_all_actual_partitions(self):
  self.assertEqual((len(NORMAL_B),sum(NORMAL_B),NORMAL_B.count(4)),(32,96,19))
  for c in cases():
   self.assertEqual(c.selected,192);self.assertEqual(c.work_budgets,(*NORMAL_B,*([3]*32)))
   self.assertEqual(c.parent_ids,tuple(range(32,64)));self.assertEqual(c.deferred_normal,tuple(range(32)))
   for budgets in (NORMAL_B,*[([3]*n) for n in range(1,33)]):
    g=target_shape(budgets);self.assertIn(g['M'],MC2_ROWS);self.assertEqual(g['real_Q']+sum(g['scratch']),g['M'])
    self.assertEqual(g['N']+len(g['scratch']),g['S']);self.assertLessEqual(len(g['scratch']),38)
  for call in (lambda:Case(0,parent_B=4),lambda:Case(0,child_B=4),lambda:target_shape([4]*25),lambda:target_shape([1]*33)):
   with self.assertRaises(RuntimeError):call()
 def test_actual_fast_shared_steps_expression_and_raw8(self):
  from originals151 import checked
  p=checked(P.parent/'tree-s35-qualified-counts-v119/fast_tree_draft_v5.py');tree=ast.parse(p.read_text())
  node=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='steps')
  ctx=dict(depth=4,plans=[types.SimpleNamespace(candidate_budget=b) for b in cases()[0].work_budgets])
  self.assertEqual(eval(compile(ast.Expression(node.value),str(p),'eval'),ctx),4)
  from draft151 import decode,counts
  class Encoded:
   def __getitem__(self,key):return self
   def detach(self):return self
   def cpu(self):return self
   def tolist(self):return [[[11*1000001+900001,21*1000001+90000],[12*1000001+800002,22*1000001+180000],[13*1000001+700003,23*1000001+270000],[14*1000001+600004,24*1000001+360000]]]
  rows,scores=decode(Encoded(),1,4);self.assertEqual(rows,[[11,12,13,14,21,22,23,24]])
  self.assertGreater(scores[0][3],0);self.assertGreater(scores[0][7],0)
  self.assertEqual(counts(Case(0)),[4]*32+[8]*32);self.assertEqual(counts(Case(1)),[5]*32+[8]*32)
  self.assertEqual(64*4*2,512)
 def test_source_closure_and_NO_parent_global_edits(self):
  from check151 import source_contract
  x=source_contract();self.assertEqual(x['frozen_parent_local_and_external_files'],146)
  self.assertFalse(x['qualification_inherited']);self.assertFalse(x['scope']['parent4_allowed'])
  for name in ('originals151.py','protocol151.py','target151.py','next151.py'):
   tree=ast.parse((P/name).read_text())
   for n in ast.walk(tree):
    if isinstance(n,ast.Assign):
     self.assertFalse(any(isinstance(t,ast.Subscript) and ('__globals__' in ast.unparse(t) or '.globals' in ast.unparse(t)) for t in n.targets))
  self.assertNotIn('verification_qualification',json.loads((P/'PARENT_FILES151.json').read_text())['runtime_load_order'])
 def test_common_failure_boundary_and_target_call_lanes(self):
  tree=ast.parse((P/'boundary151.py').read_text());fn=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='stage4'))
  fn.body=[n for n in fn.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
  events=[];gathers=[];sync=[]
  def gather(rows,value,group):
   gathers.append(value)
   for i in range(4):rows[i]=dict(value,rank=i)
  namespace=dict(time=time,require=require,torch=types.SimpleNamespace(npu=types.SimpleNamespace(synchronize=lambda:sync.append(1))),dist=types.SimpleNamespace(all_gather_object=gather))
  exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<actual-boundary-imports-only-inverse>','exec'),namespace)
  e=types.SimpleNamespace(rank=0,groups=types.SimpleNamespace(verification_coordination_group=object()))
  self.assertEqual(namespace['stage4'](e,events,'prepared',lambda:7),7)
  with self.assertRaisesRegex(RuntimeError,'failure-before-next-TP'):
   namespace['stage4'](e,events,'failure-before-next-TP',lambda:(_ for _ in ()).throw(ValueError('real-role-error')))
  self.assertEqual(len(gathers),2);self.assertEqual(len(sync),1);self.assertIn('ValueError',events[-1]['ranks'][0]['error'])
  s=(P/'target151.py').read_text();self.assertIn("'own-current/KV-finite'",s)
  self.assertLess(s.index("'own-current/KV-finite'"),s.index("'own-current/same-M-head-immediate-clone'"))
  self.assertIn("'own-KV-finite'",s);self.assertIn('common-complete-trace-Native-delta',(P/'actor151.py').read_text())
 def test_observed_only_and_true_next_history_source(self):
  p=(P/'protocol151.py').read_text();self.assertNotIn('controlled_successors',p);self.assertNotIn('valid_promotion',p)
  self.assertIn("original['preflight_consensus_compact']",p);self.assertIn("original['finish']",p);self.assertIn("original['cleanup']",p)
  self.assertIn("source['controller'].validate_verification(i) is source['staged'][i]",p)
  n=(P/'next151.py').read_text();self.assertIn("reference=draft.independent(e,case,trace['fixture'])",n)
  self.assertIn("role='draft'",n);self.assertIn('all(len(v)<=1 for v in suffixes)',n)
  a=(P/'actor151.py').read_text();self.assertIn("'passed_component' if full else 'passed_subset'",a)
  self.assertIn("full=all(v['full_scope'] for v in per_scope)",a)
if __name__=='__main__':unittest.main()
