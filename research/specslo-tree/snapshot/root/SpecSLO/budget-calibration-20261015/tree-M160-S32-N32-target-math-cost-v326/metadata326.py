"""Own-history reference:33 q5 groups, first query consumed."""

def pack_q5(engine, frame, *, full):
    from frames326 import pack
    from static_base326 import ensure_zero
    if full is not True:
        raise ValueError('explicit own-history FULL backend')
    owners = frame['selected']
    if len(owners) != 32 or not len(frame['tokens']) == len(frame['positions']) == 160:
        raise ValueError('complete33 q5 reference frame')
    positions = [frame['positions'][5 * j] for j in range(32)]
    for j, owner in enumerate(owners):
        if frame['ids'][5 * j:5 * j + 5] != [owner] * 5 or frame['positions'][5 * j:5 * j + 5] != list(range(positions[j], positions[j] + 5)):
            raise ValueError('all original own-prefix group positions/owners')
    ensure_zero(engine)
    return pack(engine, owners, positions, [frame['tokens'][5 * j:5 * j + 5] for j in range(32)], rows=160, reference=True)

def checked(meta):
    if tuple(meta.tree_attention_mask.shape) != (32, 1, 10, 4096) or tuple(meta.sequence_lens) != (2560,) * 32:
        raise RuntimeError('actual fixedM160 own reference shape')
    return meta
