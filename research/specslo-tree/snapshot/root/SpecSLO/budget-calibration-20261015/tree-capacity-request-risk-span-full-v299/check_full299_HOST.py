"""Actual Native CPU metadata/forward/observer; arithmetic model double only."""
import ast,copy,hashlib,inspect,json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
import check299
sys.path.insert(0,str(B/'tree-deferred-FULL-mask-host-projection-v259'))
def main():
 from fixture259 import build
 env=build();engine=env['engine'];native=env['native'];torch=env['torch']
 from static_state299 import bind_checked_prepare
 import branch_tail255
 bind_checked_prepare(engine._branch_tail_state,branch_tail255)
 from padding269 import pad
 native.NativePearlEngine._pad_spec_rhythm_target_tree_graph=pad
 from plan269 import make
 from oracle299 import cases
 original=native.NativePearlEngine.target_tree_forward
 node=copy.deepcopy(next(n for n in ast.walk(ast.parse((P/'extension299.py').read_text()))if isinstance(n,ast.FunctionDef)and n.name=='observed_forward'))
 ns=dict(original_forward=original);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(P/'extension299.py')+'::HOST-actual-observer','exec'),ns)
 forward=ns['observed_forward'];engine._canonical299_phase='qualified';engine._dynamic_tree_generation=1;engine._canonical299_actual_calls={}
 result=[]
 for case in cases():
  budgets=case['budgets'];n=len(budgets);plans=[make(b,127,case['anchor'])for b in budgets]
  roots=list(range(41,41+n));rows=[[103+17*i+29*j for j in range(b)]for i,b in enumerate(budgets)]
  pp,rr,dd,ss,real=pad(engine,plans,roots,rows,list(range(n)),[],[])
  out=forward(engine,pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
  state=engine._branch_tail_state;meta=state.static_current['physical'][2];q=n+sum(budgets)
  assert out['query_count']==120 and out['logical_query_count']==q and out['graph_padding_query_count']==120-q
  assert out['num_draft_tokens']==budgets and out['tree_count']==n
  assert out['target_query_token_ids'].tolist()==[v for root,row in zip(roots,rows,strict=True)for v in [root]+row]
  assert out['target_token_ids'].numel()==sum(budgets)and out['cache_slot_mapping'].numel()==q and out['cache_slot_mapping'].unique().numel()==q
  assert meta.actual_seq_lengths_q==tuple(range(5,121,5))and meta.sequence_lens==(2560,)*24
  assert meta.tree_attention_mask.shape[2]==(11 if any(b>4 for b in budgets)else 10)
  assert state.copy280_bucket in((128,256,512,1024,1792)if any(b>4 for b in budgets)else(0,))
  result.append(dict(case=case,copy_grid=state.copy280_bucket,real_queries=q))
 assert len(result)==len(engine._canonical299_actual_calls['1'])==408
 draft=types.SimpleNamespace(_canonical299_phase='qualified',is_draft=True,_branch_tail_state=types.SimpleNamespace(static_graph_calls=[]))
 ns=dict(original_forward=lambda *a,**kw:None);exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),str(P/'extension299.py')+'::HOST-draft','exec'),ns)
 assert ns['observed_forward'](draft)is None
 from padding231 import root_only
 try:forward(engine,[root_only(4096)],[0],[[]],sequence_ids=[463],return_logits=False,real_tree_count=1)
 except ValueError:pass
 else:raise AssertionError('real zero budget was accepted')
 from runtime299 import project_source
 projected,changes=project_source((B/'tree-causal80-small-cohort-role-aware-audit-full-v209/runtime209.py').read_text())
 assert len(changes)==14;compile(projected,'HOST-runtime299','exec')
 assert 'torch_npu'not in sys.modules
 output=dict(status='HOST_actual_Native408_B1to8_forward_observer_and_copy_grid_passed',cases=result,actual_observer_cases=408,complete_runtime_projection_inverse=True,draft_None_protocol=True,real_zero_budget_rejected=True,NPU_used=False,neural_qualified=False,performance_qualified=False,model='CPU arithmetic protocol double; no actual attention or NPU graph math')
 with(P/'HOST_FULL299_PROTOCOL_RESULT.json').open('x')as f:json.dump(output,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in output.items()if k!='cases'}))
if __name__=='__main__':main()
