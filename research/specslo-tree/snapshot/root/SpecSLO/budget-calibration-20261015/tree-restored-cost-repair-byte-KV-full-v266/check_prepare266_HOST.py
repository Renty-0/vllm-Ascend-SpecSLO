"""Execute the production private State.prepare on all227 CPU Native layouts."""
import json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
def main():
 sys.path[:0]=[str(B/'tree-deferred-FULL-mask-host-projection-v259'),
               str(B/'tree-fused-branch-KV-row-copy-v262')]
 from fixture259 import build
 env=build();engine=env['engine'];pad=env['pad'];torch=env['torch']
 import branch_tail255
 from static_state266 import bind_checked_prepare
 from oracle255 import cases,prepare
 state=engine._branch_tail_state
 original=state.prepare;bind_checked_prepare(state,branch_tail255)
 def digest(value):
  import hashlib
  return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
 results=[]
 for case in cases():
  prefixes=[[11]*1505,[12]*2555]+[[13+i]*127 for i in range(22)]
  plans,values=prepare(case,prefixes);n=case['N']
  pp,rr,dd,ss,real=pad(engine,plans,list(range(41,41+n)),values,list(range(n)),[],[])
  physical=[p for plan in pp for p in plan.cache_positions.tolist()]
  owners=[owner for owner,plan in zip(ss,pp,strict=True)for _ in range(plan.candidate_budget+1)]
  engine._prepare_attention_metadata(owners,physical,True)
  from host_target_metadata import make_host_tree_attention_metadata
  xs,ps,meta=make_host_tree_attention_metadata(engine,pp,rr,dd,sequence_ids=ss,allow_causal_fast_path=False)
  state.static_real_count=real
  before=original(pp,ss,xs,ps,meta)
  before_physical=state.static_current['physical'];before_src=state.source.clone();before_dst=state.dest.clone()
  after=state.prepare(pp,ss,xs,ps,meta);after_physical=state.static_current['physical']
  from fixture259 import equal
  equal(before_physical,after_physical)
  assert torch.equal(before_src,state.source)and torch.equal(before_dst,state.dest)
  results.append(dict(case=case['name'],enabled=state.enabled,physical_mask_sha256=digest(after_physical[2].tree_attention_mask)))
 assert len(results)==227 and sum(r['enabled']for r in results)==172
 assert state.copy266_layout_checks==172 and 'torch_npu'not in sys.modules
 result=dict(status='HOST_actual_private_State_prepare_all227_equivalent',
   cases=227,actual_checked_branch_layouts=172,all_original_metadata_and_maps_unchanged=True,
   validation_before_index_H2D_in_original_prepare=True,NPU_used=False,
   neural_qualified=False,Goodput=None,rows=results)
 with(P/'HOST_PREPARE266_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items()if k!='rows'}))
if __name__=='__main__':main()
