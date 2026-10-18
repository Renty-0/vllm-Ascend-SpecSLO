"""Losslessly compress validated nonidentity prefix row pairs into spans."""
def pack(source,destination,*,rows=69632):
    from contract262 import checked
    checked(source,destination,rows=rows)
    spans=[]
    for src,dst in zip(source,destination,strict=True):
        if src==dst:continue
        if spans and src==spans[-1][0]+spans[-1][2]and dst==spans[-1][1]+spans[-1][2]:
            spans[-1][2]+=1
        else:spans.append([src,dst,1])
    if len(spans)>12:raise ValueError('ROOT292 atmost12 qualified contiguous branch prefix spans')
    count=sum(n for s,d,n in spans)
    expected=[(s,d)for s,d in zip(source,destination)if s!=d]
    actual=[(s+i,d+i)for s,d,n in spans for i in range(n)]
    if actual!=expected:raise RuntimeError('ROOT292 complete row-for-row span inverse')
    return dict(spans=spans+[[0,0,0]for _ in range(12-len(spans))],span_count=len(spans),copied_rows=count)
