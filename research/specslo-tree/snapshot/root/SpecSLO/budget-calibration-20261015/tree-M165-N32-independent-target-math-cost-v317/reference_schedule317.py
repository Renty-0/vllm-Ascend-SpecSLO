"""Independent primary and alternate AR paths; no candidate KV reuse."""
def schedule(plans,roots,rows,*,future=0,prefix=None):
 from plan285 import plan_family
 n=len(plans)
 if not 1<=n<=32 or len(roots)!=n or len(rows)!=n:
  raise ValueError('HOST269 aligned unique real owners')
 plan_family(plans);paths=[]
 for i,(plan,root,row)in enumerate(zip(plans,roots,rows,strict=True)):
  b=plan.candidate_budget;p=plan.prefix_len
  if len(row)!=b or prefix is not None and p!=prefix:raise ValueError('HOST269 exact prefix and token count')
  main=list(range(min(b,4)))
  paths.append(dict(owner=32+i,prefix=p,tokens=[int(root)]+[int(row[j])for j in main],
                    query_labels=[(i,0)]+[(i,j+1)for j in main]))
  if b>4:
   anchor=int(plan.parent_indices[4])+1
   nodes=list(range(anchor))+list(range(4,b))
   paths.append(dict(owner=64+i,prefix=p,tokens=[int(root)]+[int(row[j])for j in nodes],
                      query_labels=[None]+[None]*anchor+[(i,j+1)for j in range(4,b)]))
 rounds=[]
 for depth in range(5):
  active=[]
  for path in paths:
   if depth<len(path['tokens']):
    active.append(dict(reference_owner=path['owner'],token=path['tokens'][depth],
       position=path['prefix']+depth,candidate_query=path['query_labels'][depth]))
  for begin in range(0,len(active),33):
   selected=active[begin:begin+33];tokens=[];ids=[];positions=[];owners=[]
   for r in selected:
    owners.append(r['reference_owner']);tokens.extend([r['token']]+[future]*4)
    ids.extend([r['reference_owner']]*5);positions.extend(range(r['position'],r['position']+5))
   for j in range(33-len(selected)):
    owner=463+j;owners.append(owner);tokens.extend([0]+[future]*4);ids.extend([owner]*5);positions.extend(range(5))
   if len(set(owners))!=33 or len(tokens)!=165:raise RuntimeError('HOST269 complete q5 frame and owner uniqueness')
   rounds.append(dict(depth=depth,chunk=begin//33,tokens=tokens,ids=ids,positions=positions,selected=owners,
       sequence_lens=[positions[5*j+4]+1 for j in range(33)],actual_seq_lengths_q=list(range(5,166,5)),
       real_queries=[dict(head_row=5*j,**r)for j,r in enumerate(selected)]))
 return rounds
