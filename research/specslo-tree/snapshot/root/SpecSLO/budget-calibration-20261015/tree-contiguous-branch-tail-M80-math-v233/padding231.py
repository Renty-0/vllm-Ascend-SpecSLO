"""Canonical physical padding, with a typed root-only dummy for Q79.

Real proposals remain original B1..5 plans. A root-only dummy is a private
physical query, never an allocated speculative candidate or real request.
"""
from dataclasses import dataclass

@dataclass(frozen=True)
class RootOnlyPaddingPlan:
    parent_indices:object
    positions:object
    cache_positions:object
    attention_mask:object
    candidate_budget:int=0
    prefix_len:int=0
    width:int=1
    depth:int=0

def root_only(max_model_len):
    import torch
    if type(max_model_len)is not int or max_model_len!=4096:raise ValueError('original bounded4096 model domain')
    return RootOnlyPaddingPlan(torch.empty(0,dtype=torch.long,device='cpu'),torch.tensor([0],dtype=torch.long,device='cpu'),
        torch.tensor([0],dtype=torch.long,device='cpu'),(torch.arange(max_model_len,device='cpu')[None,:]!=0))

def allowed_dummy(plan,*,row_id,real_count):
    """Narrow additional check for the private Native target-forward guard."""
    if not isinstance(plan,RootOnlyPaddingPlan)or row_id<real_count:return False
    import torch
    return (plan.candidate_budget==plan.parent_indices.numel()==plan.prefix_len==plan.depth==0
        and plan.width==1 and plan.parent_indices.device.type=='cpu'and plan.parent_indices.dtype==torch.long
        and plan.positions.device.type==plan.cache_positions.device.type=='cpu'
        and plan.positions.dtype==plan.cache_positions.dtype==torch.long
        and plan.positions.tolist()==plan.cache_positions.tolist()==[0]
        and plan.attention_mask.device.type=='cpu'and plan.attention_mask.dtype==torch.bool
        and tuple(plan.attention_mask.shape)==(1,4096)
        and not bool(plan.attention_mask[0,0])and bool(plan.attention_mask[0,1:].all()))

def pad(engine,plans,roots,rows,ids,active,states):
    from mapper231 import envelope
    from target_layout import plan_family
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    n=len(plans)
    if not n==len(roots)==len(rows)==len(ids)or len(set(ids))!=n:raise ValueError('unchanged aligned real proposals')
    m,s,lengths=envelope(n,sum(p.candidate_budget+1 for p in plans),plan_family(plans))
    scratch=engine._tree_graph_scratch_sequence_ids[:len(lengths)]
    if len(scratch)!=len(lengths)or len(set(scratch))!=len(scratch)or set(scratch)&set(ids):raise RuntimeError('independent dedicated padding owners required')
    pp,rr,dd,ss=list(plans),list(roots),list(rows),list(ids)
    for seq,q in zip(scratch,lengths,strict=True):
        pp.append(root_only(engine.config.max_model_len)if q==1 else cached_cpu_tree_speculation_plan(1,q-1,0,engine.config.max_model_len,candidate_budget=q-1))
        rr.append(0);dd.append([0]*(q-1));ss.append(seq)
    if len(pp)!=s or sum(p.parent_indices.numel()+1 for p in pp)!=m:raise RuntimeError('exact canonical physical packing')
    return pp,rr,dd,ss,n
