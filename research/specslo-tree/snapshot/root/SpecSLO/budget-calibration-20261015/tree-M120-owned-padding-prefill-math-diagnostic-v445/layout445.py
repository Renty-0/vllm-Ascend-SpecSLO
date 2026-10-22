"""Explicit known prompt-continuation rows in wholly unused q5 groups.

This is a diagnostic planner; it does not admit requests or issue tickets.
"""
def plan(prefixes,tables,owners,groups,tokens,*,real_groups):
    if not (len(owners)==len(groups)==len(tokens)>0):raise ValueError('ROOT445 complete aligned chunk descriptions')
    if len(set(owners))!=len(owners)or len(set(groups))!=len(groups):raise ValueError('ROOT445 unique owners and groups')
    if type(real_groups)is not int or not 0<=real_groups<=24:raise ValueError('ROOT445 real group count')
    rows=[];used_slots=set()
    for owner,g,chunk in zip(owners,groups,tokens,strict=True):
        if type(owner)is not int or not 0<=owner<len(tables)or type(g)is not int or not real_groups<=g<24:
            raise ValueError('ROOT445 pre-existing diagnostic owner in wholly unused group')
        if not 1<=len(chunk)<=5 or any(type(t)is not int or t<0 for t in chunk):raise ValueError('ROOT445 bounded known-token chunk')
        prefix=prefixes[owner%24]
        if not 0<len(prefix) or len(prefix)+len(chunk)>2560:raise ValueError('ROOT445 complete visible prefix within original extent')
        table=tables[owner]
        for j,token in enumerate(chunk):
            position=len(prefix)+j
            page=table[position//128]
            if type(page)is not int or not 0<=page<544:raise ValueError('ROOT445 allocated actual chunk KV page')
            slot=page*128+position%128
            if slot in used_slots:raise ValueError('ROOT445 disjoint real query writes')
            used_slots.add(slot)
            rows.append(dict(owner=owner,group=g,row=5*g+j,token=token,position=position,
                slot=slot,visible_key_count=position+1))
    return rows
