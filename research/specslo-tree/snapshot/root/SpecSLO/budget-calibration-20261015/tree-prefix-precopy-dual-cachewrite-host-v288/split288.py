"""Separate committed-prefix copies from aliases of current query K/V rows."""
def split(source,destination,query_slots,*,rows=69632):
    from contract262 import checked
    checked(source,destination,rows=rows)
    if len(query_slots)!=120 or any(type(v)is not int or not 0<=v<rows for v in query_slots):
        raise ValueError('ROOT288 exact physical120 cache writers')
    writers={}
    for row,slot in enumerate(query_slots):writers.setdefault(slot,[]).append(row)
    aliases=[-1]*120;prefix_source=list(source);prefix_dest=list(destination)
    for i,(src,dst)in enumerate(zip(source,destination,strict=True)):
        if src==dst:continue
        if src in writers:
            if len(writers[src])!=1:raise ValueError('ROOT288 ambiguous current alias writer')
            row=writers[src][0]
            if aliases[row]!=-1:raise ValueError('ROOT288 one qualified side chain per primary owner')
            aliases[row]=dst
            # Identity retains the original full-map anti-alias contract.
            prefix_source[i]=dst
        elif dst in writers:
            raise ValueError('ROOT288 committed-prefix destination is a current writer')
    checked(prefix_source,prefix_dest,rows=rows)
    assert not set(s for s,d in zip(prefix_source,prefix_dest)if s!=d)&set(writers)
    active_aliases=[v for v in aliases if v>=0]
    if len(set(active_aliases))!=len(active_aliases)or set(active_aliases)&set(writers):
        raise ValueError('ROOT288 alias outputs must be disjoint from canonical writers')
    return dict(prefix_source=prefix_source,prefix_destination=prefix_dest,query_alias_destination=aliases,
        prefix_copies=sum(s!=d for s,d in zip(prefix_source,prefix_dest)),current_aliases=len(active_aliases),
        original_copies=sum(s!=d for s,d in zip(source,destination)),query_rows=120)
