"""Own-history reference:24 q5 groups, first query consumed."""
def pack_q5(engine,frame,*,full):
    from frames255 import pack
    from static_state255 import ensure_zero
    if full is not True:raise ValueError('explicit own-history FULL backend')
    owners=frame['selected']
    if len(owners)!=24 or not len(frame['tokens'])==len(frame['positions'])==120:
        raise ValueError('complete24 q5 reference frame')
    positions=[frame['positions'][5*j]for j in range(24)]
    for j,owner in enumerate(owners):
        if frame['ids'][5*j:5*j+5]!=[owner]*5 or frame['positions'][5*j:5*j+5]!=list(range(positions[j],positions[j]+5)):
            raise ValueError('all original own-prefix group positions/owners')
    ensure_zero(engine)
    return pack(engine,owners,positions,[frame['tokens'][5*j:5*j+5]for j in range(24)],rows=120,reference=True)


def checked(meta):
    if tuple(meta.tree_attention_mask.shape)!=(24,1,10,4096)or tuple(meta.sequence_lens)!=(2560,)*24:
        raise RuntimeError('actual fixedM120 own reference shape')
    return meta
