"""Check every labeled independent AR history against actual tree ancestors."""
import json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
def main():
 sys.path.insert(0,str(B/'tree-deferred-FULL-mask-host-projection-v259'))
 from fixture259 import build
 build()
 from plan269 import make
 from reference_schedule269 import schedule
 source=json.loads((P/'HOST269_RESULT.json').read_text());assert source['cases']==1252
 labels=0;frames=0;histories=0
 for case in source['rows']:
  n=case['N'];budgets=case['budgets'];mode=case['prefix_mode']
  prefixes=([0]*n if mode=='zero'else[127]*n if mode=='boundary127'else[128]*n if mode=='boundary128'else([1505,2555]+[127]*n)[:n])
  plans=[make(b,p,case['anchor'])for b,p in zip(budgets,prefixes,strict=True)]
  rows=[[103+17*i+29*j for j in range(b)]for i,b in enumerate(budgets)];roots=list(range(41,41+n))
  seen=set();past={}
  for frame in schedule(plans,roots,rows):
   frames+=1
   assert len(frame['tokens'])==120 and frame['actual_seq_lengths_q']==list(range(5,121,5))
   for r in frame['real_queries']:
    owner=r['reference_owner'];history=past.setdefault(owner,[])
    assert r['position']==prefixes[(owner-24)%24]+len(history)
    history.append(r['token'])
    if r['candidate_query']is None:continue
    i,q=r['candidate_query'];key=(i,q);assert key not in seen;seen.add(key)
    chain=[];node=q-1
    while node>=0:chain.append(node);node=int(plans[i].parent_indices[node])
    expected=[roots[i]]+[rows[i][j]for j in reversed(chain)]
    assert history==expected
    assert r['token']==(roots[i]if q==0 else rows[i][q-1])
    assert frame['tokens'][r['head_row']]==r['token']
    labels+=1
  assert len(seen)==sum(budgets)+n
  histories+=len(past)
 assert 'torch_npu'not in sys.modules
 result=dict(status='HOST_all1252_independent_reference_histories_match_original_tree_ancestors',
   cases=1252,labeled_nodes=labels,frames=frames,independent_histories=histories,
   every_real_candidate_node_covered_exactly_once=True,duplicate_shared_prefixes_computed_independently=True,
   no_candidate_KV_or_hidden_inputs=True,NPU_used=False,neural_qualified=False,Goodput=None)
 with(P/'HOST_REFERENCE269_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
