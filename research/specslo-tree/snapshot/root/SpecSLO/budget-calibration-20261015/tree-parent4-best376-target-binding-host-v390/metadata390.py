"""Independent Q5 M120 metadata for the finite all-B4 parent/child protocol.

This builder does not install callbacks, submit models, copy candidate KV,
or issue a production capability. Real metadata starts with original plans.
"""
from dataclasses import replace


def physical(engine,frame,*,changed=False):
    from fixed_mask_extent import padded
    from projection255 import transform
    from bucket376 import extent
    from static_state255 import ensure_zero
    from originals151 import original_function
    plans,roots,rows,ids,real=frame
    if not(1<=real<=24 and len(plans)==len(roots)==len(rows)==len(ids)==24
            and all(p.candidate_budget==4 for p in plans)
            and len(set(ids))==24):
        raise RuntimeError('ROOT390 finite uniform B4/M120 original frame required')
    # Rebuild from original plan data. No state.static_current candidate tensor
    # is used as an independent reference input.
    independent=[original_function('short_target_v126','independent_plan')(p)for p in plans]
    roots=[(v+1)%engine.draft_vocab_size for v in roots]if changed else list(roots)
    x,pos,meta=engine.model.make_tree_attention_metadata(independent,roots,rows,
        engine.cache_allocation.block_tables,ids,allow_causal_fast_path=False)
    meta=padded(meta,9)
    if not(meta.tree_attention and x.numel()==120 and tuple(meta.actual_seq_lengths_q)==tuple(range(5,121,5))):
        raise RuntimeError('ROOT390 original complete FULL120 metadata required')
    selected=extent(tuple(meta.sequence_lens))
    if not engine._branch_tail_state.uniform_KV_bucket376_enabled:
        raise RuntimeError('ROOT390 actual best376 target extent policy required')
    zero_page=ensure_zero(engine)
    xx,pp,mm,inverse,mapping=transform(x,pos,meta,5*real,zero_page,
        extent=2560,mask_capacity=10)
    mm=replace(mm,sequence_lens=(selected,)*24)
    if mapping['physical_to_original']!=list(range(120)):
        raise RuntimeError('ROOT390 allB4 identity query order required')
    return xx,pp,mm
