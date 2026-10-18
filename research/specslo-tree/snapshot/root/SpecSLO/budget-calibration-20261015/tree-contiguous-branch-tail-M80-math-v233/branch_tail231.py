"""Contiguous per-branch KV views using private scratch tails, BF16 copies.

Primary queries retain their ordinary table. A side query receives its own
FIA segment and contiguous ancestral KV order. Real KV writes/commit slots
remain the original tree's physical slots. No model weights are changed.
"""
from contextlib import contextmanager
from dataclasses import replace

COPY_SLOTS=1792
BLOCK=128
CONTEXT=4096

def is_fork(plan):
    parents=plan.parent_indices.tolist()
    return any(parents.count(p)>1 for p in set(parents))

def layout(plans,ids,tables,scratch_ids,*,copy_slots=COPY_SLOTS):
    import torch
    if len(plans)!=len(ids)or len(set(ids))!=len(ids):raise ValueError('independent original physical plan owners')
    if sum(p.parent_indices.numel()+1 for p in plans)!=80:raise ValueError('fixed original80 query rows')
    branches=sum(is_fork(p)for p in plans)
    if not branches:return dict(enabled=False,branches=0,physical_queries=80)
    used=set(ids);available=[s for s in scratch_ids if s not in used]
    pages=[tables[s][0]for s in available]
    if len(pages)!=len(set(pages))or any(p<0 or p>=544 for p in pages):raise ValueError('allocated independent unused scratch pages')
    if set(pages)&{p for seq in ids for p in tables[seq]if p>=0}:raise ValueError('scratch cannot alias any current plan pages')
    consumed=0;groups=[];source=[];dest=[];row=0
    for plan,owner in zip(plans,ids,strict=True):
        parents=plan.parent_indices.tolist();physical=plan.cache_positions.tolist();logical=plan.positions.tolist();mask=plan.attention_mask
        q=len(parents)+1;table=list(tables[owner])
        if tuple(mask.shape)!=(q,CONTEXT)or mask.device.type!='cpu':raise ValueError('original complete CPU tree mask')
        if not is_fork(plan):
            groups.append(dict(owner=owner,original_rows=list(range(row,row+q)),tables=table,KV=max(physical)+1,masks=mask));row+=q;continue
        if len(parents)!=5 or parents[:4]!=[-1,0,1,2]or parents[4]not in(-1,0,1,2):raise ValueError('one qualified B5 side branch')
        prefix=physical[0];depth=parents[4]+2
        if physical!=list(range(prefix,prefix+6))or logical!=list(range(prefix,prefix+5))+[prefix+depth]:raise ValueError('original primary/side physical and RoPE layout')
        groups.append(dict(owner=owner,original_rows=list(range(row,row+5)),tables=table,KV=prefix+5,masks=mask[:5]))
        visible=(~mask[5]).nonzero().flatten().tolist()
        if visible[:prefix]!=list(range(prefix))or len(visible)!=logical[5]+1 or visible[-1]!=physical[5]:raise ValueError('exact branch visibility and ancestor order')
        complete=prefix//BLOCK;tail=visible[complete*BLOCK:];n=(len(tail)+BLOCK-1)//BLOCK
        own=pages[consumed:consumed+n]
        if len(own)!=n:raise ValueError('private branch-tail page bound')
        consumed+=n;virtual=list(table)
        if complete+n>len(virtual):raise ValueError('canonical logical KV exceeds model context')
        for j,page in enumerate(own):virtual[complete+j]=page
        for j,pos in enumerate(tail):
            physical_page=table[pos//BLOCK]
            if not 0<=physical_page<544:raise ValueError('branch reads only original allocated KV')
            source.append(physical_page*BLOCK+pos%BLOCK);dest.append(own[j//BLOCK]*BLOCK+j%BLOCK)
        canonical_mask=torch.arange(CONTEXT,device='cpu')[None,:]>=logical[5]+1
        groups.append(dict(owner=owner,original_rows=[row+5],tables=virtual,KV=logical[5]+1,masks=canonical_mask));row+=q
    real_copies=len(source)
    if real_copies>copy_slots:raise ValueError('fixed branch-copy slot bound')
    padding=copy_slots-real_copies;extra=(padding+BLOCK-1)//BLOCK;identity=pages[consumed:consumed+extra]
    if len(identity)!=extra:raise ValueError('private identity-copy padding pages')
    for j in range(padding):
        slot=identity[j//BLOCK]*BLOCK+j%BLOCK;source.append(slot);dest.append(slot)
    if len(dest)!=copy_slots or len(set(dest))!=copy_slots:raise RuntimeError('unique complete fixed copy destinations')
    if any(not 0<=s<544*BLOCK for s in source+dest):raise ValueError('fixed original cache capacity')
    masks=torch.ones((len(groups),1,10,CONTEXT),dtype=torch.bool,device='cpu');ends=[];cursor=0
    for i,g in enumerate(groups):
        q=len(g['original_rows']);masks[i,0,:q]=g['masks'];cursor+=q;ends.append(cursor)
    if cursor!=80 or [r for g in groups for r in g['original_rows']]!=list(range(80)):raise RuntimeError('original query order retained')
    return dict(enabled=True,branches=branches,physical_queries=80,groups=groups,mask=masks,
        actual_seq_lengths_q=tuple(ends),sequence_lens=tuple(g['KV']for g in groups),tables=[g['tables']for g in groups],
        source=source,dest=dest,real_copy_slots=real_copies,fixed_copy_slots=copy_slots,
        branch_tail_pages=consumed,identity_padding_pages=extra,attention_segments=len(groups))

class State:
    def __init__(self,engine):
        import torch
        self.engine=engine;self.enabled=False;self.last=None;self.last_record=None
        self.source=torch.empty(COPY_SLOTS,dtype=torch.long,device=engine.device)
        self.dest=torch.empty(COPY_SLOTS,dtype=torch.long,device=engine.device)

    @contextmanager
    def active(self,enabled):
        prior=self.enabled;self.enabled=bool(enabled)
        try:yield
        finally:self.enabled=prior

    def prepare(self,plans,ids,inputs,pos,meta):
        import torch
        engine=self.engine
        if any(is_fork(p)for p in plans):
            # All scratch owners are private physical padding, not live requests.
            for seq in engine._tree_graph_scratch_sequence_ids:
                if engine.cache_allocation.block_tables[seq][0]<0:engine._activate_cache_sequence(seq,[0],enable_prefix_caching=False)
        desc=layout(plans,ids,engine.cache_allocation.block_tables,engine._tree_graph_scratch_sequence_ids)
        self.enabled=desc['enabled']
        if self.enabled:
            self.source.copy_(torch.tensor(desc['source'],dtype=torch.long,device='cpu'))
            self.dest.copy_(torch.tensor(desc['dest'],dtype=torch.long,device='cpu'))
            tables=torch.tensor(desc['tables'],dtype=torch.int32,device=engine.device)
            meta=replace(meta,block_tables=tables,request_block_tables=tables,tree_attention_mask=desc['mask'].to(device=engine.device),
                actual_seq_lengths_q=desc['actual_seq_lengths_q'],sequence_lens=desc['sequence_lens'])
        self.last=(inputs,pos,meta,self.enabled)
        self.last_record={k:desc[k]for k in('enabled','branches','physical_queries')}
        if self.enabled:self.last_record.update({k:desc[k]for k in('real_copy_slots','fixed_copy_slots','branch_tail_pages','identity_padding_pages','attention_segments')})
        return inputs,pos,meta

    def eager_last(self):
        inputs,pos,meta,enabled=self.last
        with self.active(enabled):return self.engine.model(inputs,pos,meta).clone()

def install(engine,native):
    """Explicit post-constructor diagnostic epoch; production capability pending."""
    from padding231 import pad
    state=State(engine);engine._branch_tail_state=state
    for layer in engine.model.layers:
        attention=layer.self_attn;old=attention._fused_infer_attention
        def execute(query,metadata,*,_attention=attention,_old=old):
            if state.enabled:
                key=_attention.key_cache.flatten(0,1);value=_attention.value_cache.flatten(0,1)
                key.index_copy_(0,state.dest,key.index_select(0,state.source))
                value.index_copy_(0,state.dest,value.index_select(0,state.source))
            return _old(query,metadata)
        attention._fused_infer_attention=execute
    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph=pad
    original_metadata=native._dynamic_tree_make_host_metadata
    def metadata(e,plans,roots,rows,*args,**kwargs):
        result=original_metadata(e,plans,roots,rows,*args,**dict(kwargs,allow_causal_fast_path=False))
        ids=kwargs.get('sequence_ids');ids=list(range(len(plans)))if ids is None else list(ids)
        return state.prepare(plans,ids,*result)
    native._dynamic_tree_make_host_metadata=metadata
    original_forward=native.NativePearlEngine.target_tree_forward
    def forward(e,*args,**kwargs):
        try:return original_forward(e,*args,**kwargs)
        finally:state.enabled=False
    native.NativePearlEngine.target_tree_forward=forward
    return dict(M=80,FULL9_linear=True,FULL10_copy_family=True,fixed_branch_copy_slots=COPY_SLOTS,
        original_weights_and_greedy_sampler=True,branch_KV_view_changed=True,production_capability_issued=False)
