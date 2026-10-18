"""Stabilize only unused FULL-mask query rows; keep real Q/KV lengths exact."""
from dataclasses import replace


def padded(metadata,query_capacity=9):
    if not getattr(metadata,'tree_attention',False) or not metadata.use_fused_infer_attention:
        return metadata
    mask=metadata.tree_attention_mask
    if mask is None or mask.ndim!=4 or mask.shape[2]>query_capacity:
        raise ValueError('FULL mask cannot fit the selected candidate bound')
    if mask.shape[2]==query_capacity:
        return metadata
    new_mask=mask.new_ones((mask.shape[0],mask.shape[1],query_capacity,mask.shape[3]))
    new_mask[:,:,:mask.shape[2],:].copy_(mask)
    return replace(metadata,tree_attention_mask=new_mask)


def install(native):
    original=native._dynamic_tree_make_host_metadata
    def prepare(*args,**kwargs):
        tokens,positions,metadata=original(*args,**kwargs)
        return tokens,positions,padded(metadata)
    native._dynamic_tree_make_host_metadata=prepare
