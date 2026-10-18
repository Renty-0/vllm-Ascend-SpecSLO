"""Actual paired policy and LIVE draft output ABI/charging without NPU."""
import copy,json,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
import check287
sys.path.append(str(B/'tree-high-normal-cap4-native-families-v148'))
def main():
 import torch
 torch.set_num_threads(1)
 from budget_policy import TreeBudgetPolicy
 from live287 import begin,observe,subset,merged
 from paired274 import snapshot
 model=json.loads((B/'tree-high-normal-cap4-native-families-v148/tree-gain-model.json').read_text())
 p=TreeBudgetPolicy(model);e=types.SimpleNamespace(_ROOT287_policy=p,_dynamic_tree_generation=1,_canonical287_phase='qualified',_ROOT287_request_gain_observations={})
 begin(e);cases=[]
 for cls in('tight','normal','loose'):
  for b in range(1,9):
   parents=[-1,0,1,2,-1,4,5,6][:b]
   paths=([list(range(k))for k in range(b+1)]if b<=4 else[list(range(k))for k in range(5)]+[list(range(4,4+k))for k in range(1,b-3)])
   for path in paths:
    req=types.SimpleNamespace(request_index=len(cases),slo_class=cls,output_token_budget=256,delivered_tokens=1)
    plan=types.SimpleNamespace(parent_indices=torch.tensor(parents),candidate_budget=b)
    accepted=path+[-1]*(4-len(path));outputs=list(range(10,11+len(path)))+[-1]*(4-len(path))
    before=copy.deepcopy(p.observed);r=observe(e,req,plan,accepted,outputs)
    side=sum(x>=4 for x in path);g=len(path)+1
    if b>=5:assert r['same_prefix_B4']==g-side and r['accepted_side_nodes']==side
    # The original actual-budget observation remains the original caller's job.
    if b>=4:assert p.observed.get((cls,b))==before.get((cls,b))
    else:assert p.observed==before
    row=e._ROOT287_request_gain_observations['1'][str(req.request_index)]['by_budget'][str(b)]
    assert row['samples']==1 and row['delivered_sum']==g and row['accepted_side_nodes']==side
    assert row['shadow_B4_available']is(b>=4)
    assert row['shadow_B4_sum']==(g-side if b>=4 else 0)
    p.observe(req,b,g);cases.append(dict(B=b,cls=cls,path=path))
 terminal=types.SimpleNamespace(request_index=99999,slo_class='tight',output_token_budget=256,delivered_tokens=252)
 before=copy.deepcopy(p.observed);observe(e,terminal,types.SimpleNamespace(parent_indices=torch.tensor([-1,0,1,2,-1]),candidate_budget=5),[-1]*4,[19,-1,-1,-1,-1])
 assert p.observed==before and '99999'not in e._ROOT287_request_gain_observations['1']
 p.observed.clear();e._dynamic_tree_generation=2;begin(e);assert snapshot(p)['by_class_anchor']=={}
 try:e._dynamic_tree_generation=1;begin(e)
 except ValueError:pass
 else:raise AssertionError('stale actual generation accepted')
 abi=[]
 for n,selected,physical in[(1,[0],1),(8,[0,3,7],4),(16,list(range(12)),16)]:
  budgets=[4]*n
  for index in selected:budgets[index]=6+index%3
  output=dict(draft_token_ids=[list(range(8))for _ in range(n)],root_token_ids=list(range(n)),num_draft_tokens=budgets,node_confidences=[torch.ones(8)*.5 for _ in range(n)],draft_confidence=torch.ones(n)*.5,query_count=4*n,graph_padding_query_count=0,model_calls=1,graph_calls=1)
  initial=copy.deepcopy(output);part=subset(output,selected)
  alt=dict(part,draft_token_ids=[v[:5]+[81,82,83]for v in part['draft_token_ids']],node_confidences=[torch.ones(8)*.7 for _ in selected],draft_confidence=torch.ones(len(selected))*.7,num_draft_tokens=[8]*len(selected))
  result=merged(output,selected,alt,physical)
  assert result['num_draft_tokens']==budgets and result['query_count']==4*n+3*physical
  assert result['graph_padding_query_count']==3*(physical-len(selected))and result['graph_calls']==3 and result['model_calls']==2
  assert output['draft_token_ids']==initial['draft_token_ids']and torch.equal(output['draft_confidence'],initial['draft_confidence'])
  assert all(result['draft_token_ids'][i]==(list(range(5))+[81,82,83]if i in selected else list(range(8)))for i in range(n))
  abi.append(dict(N=n,owners=len(selected),physical=physical,actual_extra_queries=3*physical,selected_B_preserved=True))
 assert 'torch_npu'not in sys.modules
 value=dict(status='HOST_LIVE_paired_observe_generation_and_draft_merge_ABI_passed',actual_policy_cases=len(cases),terminal_excluded=True,original_actual_budget_observe_once=True,request_histograms_independently_checked=True,stale_generation_rejected=True,ABI=abi,NPU_used=False,empirical_g_measured=False,Goodput=None)
 with(P/'HOST_LIVE287_POST_SNAPSHOT_RESULT.json').open('x')as f:json.dump(value,f,indent=2);f.write('\n')
 print(json.dumps(value))
if __name__=='__main__':main()
