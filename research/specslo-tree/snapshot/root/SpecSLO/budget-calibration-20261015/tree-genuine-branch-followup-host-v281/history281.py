"""Teacher-forced controls for the original deferred canonical-KV repair.

These are numeric qualification inputs, never real target verdict samples.
"""
def advance(prefix,root,raw,accepted,corrected):
    if not 0<=accepted<=4 or len(raw)!=8 or raw[0]==raw[4]:
        raise ValueError('HOST281 genuine distinct main/rootalternate and accepted0..4')
    values=list(prefix)+[int(root)]+list(map(int,raw[4:4+accepted]))+[int(corrected)]
    p=len(prefix);newp=len(values)-1
    return dict(history=values,old_prefix=p,new_prefix=newp,
        accepted_alternate=list(raw[4:4+accepted]),
        first_missing=p+1,repair_tokens=list(values[p+1:newp]),
        packed_inputs=list(values[p+1:]),packed_positions=list(range(p+1,newp+1)),
        root_offset=accepted,root_seed_required=accepted>0,
        teacher_forced_commit_control=True)
