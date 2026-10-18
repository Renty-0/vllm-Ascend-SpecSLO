"""HOST proof that omitted copy programs are identity rows only."""
BUCKETS=(128,256,512,1024,1792)
def select_grid(source,destination,*,rows=69632):
    from contract278 import checked
    checked(source,destination,rows=rows)
    last=max((i+1 for i,(s,d)in enumerate(zip(source,destination,strict=True))if s!=d),default=0)
    bucket=next(b for b in BUCKETS if b>=last)
    if any(s!=d for s,d in zip(source[bucket:],destination[bucket:],strict=True)):
        raise RuntimeError('ROOT278 omitted nonidentity copy')
    return bucket
