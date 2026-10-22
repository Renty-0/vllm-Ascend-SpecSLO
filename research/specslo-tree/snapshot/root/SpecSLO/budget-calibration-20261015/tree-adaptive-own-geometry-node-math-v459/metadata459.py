"""Own-history q5 references retain every original mask and KV bound."""
def pack_q5(engine,frame,*,full):
    from frames459 import pack
    from static_state255 import ensure_zero
    if full is not True:
        raise ValueError('ROOT459 independent FULL reference backend')
    rows=frame['physical_M']
    if rows not in (80,120):
        raise ValueError('ROOT459 declared physical reference rows')
    groups=rows//5;owners=frame['selected']
    if len(owners)!=groups or not len(frame['tokens'])==len(frame['positions'])==len(frame['ids'])==rows:
        raise ValueError('ROOT459 complete independent q5 frame')
    positions=[frame['positions'][5*j]for j in range(groups)]
    for j,owner in enumerate(owners):
        if frame['ids'][5*j:5*j+5]!=[owner]*5 or frame['positions'][5*j:5*j+5]!=list(range(positions[j],positions[j]+5)):
            raise ValueError('ROOT459 all original own-prefix positions and owners')
    ensure_zero(engine)
    return pack(engine,owners,positions,[frame['tokens'][5*j:5*j+5]for j in range(groups)],rows=rows,reference=True)

def checked(meta):
    groups=len(meta.actual_seq_lengths_q)
    if groups not in (16,24)or tuple(meta.actual_seq_lengths_q)!=tuple(range(5,5*groups+1,5)):
        raise RuntimeError('ROOT459 actual independent q5 reference partition')
    if tuple(meta.tree_attention_mask.shape)!=(groups,1,10,4096)or tuple(meta.sequence_lens)!=(2560,)*groups:
        raise RuntimeError('ROOT459 original exact FULL reference mask and extent')
    return meta
