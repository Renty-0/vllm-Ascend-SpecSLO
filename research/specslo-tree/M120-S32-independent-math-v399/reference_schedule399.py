"""Stepwise independent spine/side paths for bounded target math qualification.

Each reference owner retains its own neural KV prefix. Returned q5 frames
consume only the first query; future inputs are canaries. No candidate KV
is copied. This is a teacher-forced math oracle, not whole-workload AR.
"""
def schedule(plans,roots,rows,*,future=0,prefix=None):
    n=len(plans)
    if not 1<=n<=32 or len(roots)!=n or len(rows)!=n:raise ValueError('aligned real N<=32 required')
    descriptions=[]
    for i,(plan,root,row)in enumerate(zip(plans,roots,rows,strict=True)):
        parents=plan.parent_indices.tolist();positions=plan.positions.tolist();b=len(parents)
        actual_prefix=positions[0]
        if prefix is not None and actual_prefix!=prefix:raise ValueError('explicit prefix mismatch')
        if b!=plan.candidate_budget or len(row)!=b or not 1<=b<=5 :raise ValueError('original B1..5 plan required')
        spine=min(b,4)
        if parents[:spine]!=[-1,0,1,2][:spine]:raise ValueError('qualified original primary spine order')
        branch_depth=parents[4]+2 if b==5 else None
        if b==5 and not 1<=branch_depth<=4:raise ValueError('original one-side branch anchor')
        if any(positions[j+1]!=actual_prefix+j+1 for j in range(spine))or b==5 and positions[5]!=actual_prefix+branch_depth:raise ValueError('logical ancestry and RoPE positions')
        descriptions.append((int(root),list(map(int,row)),spine,branch_depth,actual_prefix))
    rounds=[]
    for depth in range(5):
        active=[]
        for i,(root,row,spine,branch_depth,actual_prefix)in enumerate(descriptions):
            if depth<=spine:
                active.append(dict(reference_owner=32+i,token=root if depth==0 else row[depth-1],position=actual_prefix+depth,candidate_query=(i,depth)))
            if branch_depth is not None and depth<=branch_depth:
                token=root if depth==0 else row[4]if depth==branch_depth else row[depth-1]
                active.append(dict(reference_owner=64+i,token=token,position=actual_prefix+depth,candidate_query=(i,5)if depth==branch_depth else None))
        for begin in range(0,len(active),24):
            selected=active[begin:begin+24];tokens=[];ids=[];positions=[]
            owners=[r['reference_owner']for r in selected]
            for r in selected:
                tokens.extend([r['token']]+[future]*4);ids.extend([r['reference_owner']]*5);positions.extend(range(r['position'],r['position']+5))
            for j in range(24-len(selected)):
                owner=463+j;owners.append(owner);tokens.extend([0]+[future]*4);ids.extend([owner]*5);positions.extend(range(5))
            if len(owners)!=len(set(owners))or len(tokens)!=120:raise RuntimeError('exact independent unique owner frame')
            rounds.append(dict(depth=depth,chunk=begin//24,tokens=tokens,ids=ids,positions=positions,selected=owners,
                sequence_lens=[positions[5*j+4]+1 for j in range(24)],actual_seq_lengths_q=list(range(5,121,5)),
                real_queries=[dict(head_row=5*j,**r)for j,r in enumerate(selected)]))
    return rounds
