"""Own-history reference:25 q5 groups plus nonreal3, first query consumed."""
def pack_q5(engine,frame,*,full):
    from frames251 import pack
    from static_state251 import ensure_zero
    if full is not True:raise ValueError('explicit own-history FULL backend')
    owners=frame['selected']
    if len(owners)!=25 or not len(frame['tokens'])==len(frame['positions'])==128:
        raise ValueError('complete25 q5 plus3 reference frame')
    positions=[frame['positions'][5*j]for j in range(25)]
    for j,owner in enumerate(owners):
        if frame['ids'][5*j:5*j+5]!=[owner]*5 or frame['positions'][5*j:5*j+5]!=list(range(positions[j],positions[j]+5)):
            raise ValueError('all original own-prefix group positions/owners')
    if frame['ids'][125:]!=[488]*3 or frame['positions'][125:]!=[0,1,2]:raise ValueError('private always-nonreal three-query tail')
    ensure_zero(engine)
    return pack(engine,owners,positions,[frame['tokens'][5*j:5*j+5]for j in range(25)],rows=128,reference=True,tail_tokens=frame['tokens'][125:])


def checked(meta):
    if tuple(meta.tree_attention_mask.shape)!=(26,1,10,4096)or tuple(meta.sequence_lens)!=(2560,)*26:
        raise RuntimeError('actual fixedM128 own reference shape')
    return meta
