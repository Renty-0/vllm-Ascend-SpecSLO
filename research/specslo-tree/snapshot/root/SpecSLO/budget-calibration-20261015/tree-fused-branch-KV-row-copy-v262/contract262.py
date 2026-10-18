"""HOST precondition for snapshot-equivalent fused in-place row copies."""
def checked(source,destination,*,rows):
    if type(rows)is not int or rows<=0 or len(source)!=1792 or len(destination)!=1792:
        raise ValueError('fixed1792 complete source/destination map')
    if any(type(index)is not int or not 0<=index<rows for index in (*source,*destination)):
        raise ValueError('every source/destination row must be in bounds')
    if len(set(destination))!=len(destination):
        raise ValueError('unique destination rows required')
    moved={d for s,d in zip(source,destination,strict=True)if s!=d}
    if moved.intersection(source):
        raise ValueError('nonidentity destinations cannot overlap any source row')
    return dict(pairs=len(source),nonidentity=sum(s!=d for s,d in zip(source,destination,strict=True)),
        identity=sum(s==d for s,d in zip(source,destination,strict=True)),rows=rows,
        in_place_snapshot_copy_equivalence=True)
