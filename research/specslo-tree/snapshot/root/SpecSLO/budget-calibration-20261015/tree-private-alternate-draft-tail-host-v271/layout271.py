"""HOST-only private PA tables for genuine alternate draft continuations.

The real owner keeps its canonical primary KV and ledger. Complete immutable
prefix pages are shared; its last partial page plus root is copied privately.
This module executes no model, NPU copy, allocation or ledger mutation.
"""
def prepare(tables,owners,prefixes,*,private_owners=tuple(range(500,524))):
    from contract262 import checked
    if len(owners)!=len(prefixes)or not 1<=len(owners)<=12 or len(set(owners))!=len(owners):
        raise ValueError('HOST271 one to12 independent current owners')
    if any(type(i)is not int or not 0<=i<463 for i in owners):
        raise ValueError('HOST271 real request owner indices only')
    if len(private_owners)!=24 or len(set(private_owners))!=24 or any(not 463<=i<526 for i in private_owners):
        raise ValueError('HOST271 exact24 private scratch owners, excluding zero526')
    pages=[tables[i][0]for i in private_owners]
    if len(set(pages))!=24 or any(type(p)is not int or not 0<=p<544 for p in pages):
        raise ValueError('HOST271 allocated unique private pages')
    real_pages={p for row in tables[:463]for p in row if p>=0}
    if real_pages.intersection(pages):
        raise ValueError('HOST271 scratch cannot alias any real owner or reference prefix')
    source=[];destination=[];virtual=[];rows=[]
    for ordinal,(owner,prefix)in enumerate(zip(owners,prefixes,strict=True)):
        if type(prefix)is not int or not 0<=prefix<=2555:
            raise ValueError('HOST271 explicit actual prefix range0..2555')
        table=list(tables[owner]);complete,offset=divmod(prefix,128)
        if len(table)!=32 or any(not 0<=page<544 for page in table[:complete+1]):
            raise ValueError('HOST271 all shared prefix and materialized-root pages allocated')
        first,second=pages[2*ordinal:2*ordinal+2]
        v=list(table);v[complete]=first;v[complete+1]=second
        for j in range(offset+1):
            source.append(table[complete]*128+j);destination.append(first*128+j)
        virtual.append(v)
        query_positions=[prefix+1+step for step in range(3)]
        query_slots=[v[p//128]*128+p%128 for p in query_positions]
        if not all(slot//128 in(first,second)for slot in query_slots):
            raise RuntimeError('HOST271 every alternate write must be private')
        rows.append(dict(owner=owner,prefix=prefix,shared_pages=complete,
                         private_pages=[first,second],positions=query_positions,slots=query_slots,
                         contexts=[p+1 for p in query_positions]))
    real=len(source);moved=set(destination)
    padding=[page*128+j for page in pages for j in range(128)if page*128+j not in moved][:1792-real]
    if len(padding)!=1792-real:raise RuntimeError('HOST271 fixed1792 identity padding capacity')
    source.extend(padding);destination.extend(padding)
    proof=checked(source,destination,rows=69632)
    return dict(owners=list(owners),virtual_tables=virtual,rows=rows,
                source=source,destination=destination,real_copy_rows=real,
                HOST_copy_contract=proof,private_page_pool=pages,
                private_owners=list(private_owners),
                source_prefix_pages=[list(tables[owner][:prefix//128+1])
                                     for owner,prefix in zip(owners,prefixes,strict=True)],
                source_tables_identity=id(tables),
                original_tables_mutated=False,real_owner_or_ledger_mutated=False)
