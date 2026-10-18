"""Paired synchronous whole-target cost after independent node math passes.

Inputs are teacher-forced qualification fixtures. No actual draft model
generation, service requests, g(n) observations or Goodput are measured.
"""
import time

def run(engine,native,stage,bank,*,rounds=20):
 import torch
 from plan269 import make
 target=not engine.is_draft;records=[]
 before=engine.graph_metrics()
 for n in (4,8,12,16,20,23):
  maximum=min(n,24-n,(96-4*n)//4)
  for forks in sorted({1,maximum}):
   arms={};expected={}
   for budget in (4,5,6,7,8):
    budgets=[budget]*forks+[4]*(n-forks)
    plans=[make(b,len(bank['prefixes'][i]),0)for i,b in enumerate(budgets)]
    rows=[[103+17*i+29*j for j in range(b)]for i,b in enumerate(budgets)]
    roots=bank['roots'][:n]if target else[0]*n
    arms[budget]=(plans,roots,rows,budgets)
    def warm():
     if not target:return None
     pp,rr,dd,ss,real=engine._pad_spec_rhythm_target_tree_graph(plans,bank['roots'][:n],rows,list(range(n)),[],[])
     out=engine.target_tree_forward(pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
     torch.npu.synchronize()
     return out['target_query_token_ids'].cpu().tolist()
    expected[budget]=stage(f'paired-target-cost/N{n}/forks{forks}/B{budget}/warm',warm)
   names=(4,5,6,7,8)
   pairs=[]
   for iteration in range(rounds):
    order=names[iteration%5:]+names[:iteration%5];costs={}
    for budget in order:
     plans,roots,rows,budgets=arms[budget]
     def execute():
      if not target:return None
      torch.npu.synchronize();started=time.perf_counter()
      pp,rr,dd,ss,real=engine._pad_spec_rhythm_target_tree_graph(plans,roots,rows,list(range(n)),[],[])
      out=engine.target_tree_forward(pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
      torch.npu.synchronize();elapsed=(time.perf_counter()-started)*1000
      if out['used_aclgraph']is not True:raise RuntimeError('ROOT280 paired target must use actual ACLGraph')
      if out['target_query_token_ids'].cpu().tolist()!=expected[budget]:
       raise RuntimeError('ROOT280 repeated actual target output changed')
      return elapsed
     costs[str(budget)]=stage(f'paired-target-cost/N{n}/forks{forks}/pair{iteration}/B{budget}',execute)
    pairs.append(dict(order=order,milliseconds=costs))
   records.append(dict(N=n,branch_owners=forks,rounds=rounds,pairs=pairs,
      same_original_committed_prefix=True,original_greedy_output_stable=True if target else None,
      physical_M120=True,q5_groups=24,fixed_KV_extent=2560,
      note='B4 has no side group; B5..8 use same number of side groups with1..4 real nodes'))
 after=engine.graph_metrics()
 checked=['aclgraph_captures','aclgraph_capture_attempts','aclgraph_runtime_validation_replays','aclgraph_failed_captures','aclgraph_capacity_fallbacks','aclgraph_shape_fallbacks']
 if any(k not in before or k not in after or after[k]!=before[k]for k in checked):
  raise RuntimeError('ROOT280 target costs must reuse captured/validated graph inventory')
 return dict(rank=engine.rank,target_math_on_rank=target,rows=records,
    graph_counters_before=before,graph_counters_after=after,no_new_capture_or_validation_in_costs=True,
    scope='Synchronous whole-target pad/metadata/model/sync elapsed, excluding D2H output comparison and stage consensus',
    real_draft_generation=False,empirical_g_measured=False,Goodput=None,performance_qualified=False)
