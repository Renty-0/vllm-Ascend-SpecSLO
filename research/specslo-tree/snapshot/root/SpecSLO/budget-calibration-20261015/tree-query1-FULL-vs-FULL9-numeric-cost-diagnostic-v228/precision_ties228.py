"""Diagnostic bounded FP32 refinement of complete top BF16 ties, no install."""
def local_candidates(head,hidden,vocabulary_size,*,capacity=4):
    import torch
    size=min(head.vocab_end,vocabulary_size)-head.vocab_start
    if size<=0:
        zeros=torch.zeros(hidden.shape[0],device=hidden.device,dtype=torch.float32)
        return torch.stack((zeros-torch.inf,zeros-torch.inf,zeros,zeros,zeros),dim=-1)
    logits=head._project_local(hidden)[:,:size]
    head._track_logits(logits)
    maximum,original=logits.max(-1)
    values,ids=logits.topk(min(capacity,size),dim=-1)
    weights=head.weight.index_select(0,ids.flatten()).reshape(*ids.shape,head.weight.shape[-1]).float()
    dots=(weights*hidden.float()[:,None,:]).sum(-1,dtype=torch.float32)
    eligible=values==maximum[:,None]
    scores=dots.masked_fill(~eligible,-torch.inf)
    best=scores.max(-1).values
    chosen=torch.where(eligible & (scores==best[:,None]),ids,torch.full_like(ids,vocabulary_size)).amin(-1)
    ties=(logits==maximum[:,None]).sum(-1)
    return torch.stack((maximum.float(),best,(chosen+head.vocab_start).float(),
        (original+head.vocab_start).float(),ties.float()),dim=-1)


def merge(candidates,vocabulary_size,*,capacity=4):
    import torch
    values=candidates[...,0];maximum=values.max(-1,keepdim=True).values
    eligible=values==maximum
    overflow=(eligible & (candidates[...,4]>capacity)).any(-1)
    scores=candidates[...,1].masked_fill(~eligible,-torch.inf)
    best=scores.max(-1,keepdim=True).values
    refined=torch.where(eligible & (scores==best),candidates[...,2],torch.full_like(values,vocabulary_size)).amin(-1).long()
    original=torch.where(eligible,candidates[...,3],torch.full_like(values,vocabulary_size)).amin(-1).long()
    return torch.where(overflow,original,refined),dict(original=original,overflow=overflow,
        global_tie_count=torch.where(eligible,candidates[...,4],torch.zeros_like(values)).sum(-1).long())


def greedy(head,hidden,vocabulary_size,*,capacity=4):
    import torch
    import torch.distributed as dist
    own=local_candidates(head,hidden,vocabulary_size,capacity=capacity)
    if head.context.size==1:all_candidates=own[:,None,:]
    else:
        gathered=[torch.empty_like(own)for _ in range(head.context.size)]
        dist.all_gather(gathered,own,group=head.context.group)
        all_candidates=torch.stack(gathered,dim=1)
    return merge(all_candidates,vocabulary_size,capacity=capacity)
