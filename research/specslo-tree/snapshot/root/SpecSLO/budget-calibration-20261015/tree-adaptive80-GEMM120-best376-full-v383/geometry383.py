"""Reconstruct adaptive physical geometry and checked uniform KV from real metadata."""
def envelope(n,q,family,budgets):
    from mapper255 import envelope as original
    from geometry377 import choose
    original(n,q,family)
    if len(budgets)!=n or n+sum(budgets)!=q or any(type(b)is not int or b not in range(1,6)for b in budgets):
        raise ValueError('ROOT383 actual unmodified logical budgets')
    m=choose(budgets,enabled=True);remaining=m-q;padding=[]
    while remaining:
        size=min(5,remaining);padding.append(size);remaining-=size
    return m,n+len(padding),padding


def valid_record(row):
    from bucket377 import extent
    n,c,branches=row['N'],row['C'],row['branches']
    expected=80 if n+branches<=16 and n+c<=80 else 120
    detail=row['uniform_KV_bucket377']
    try:kv=extent(tuple(detail['original_sequence_lens']))
    except (TypeError,ValueError,KeyError):return False
    return bool(row['M']==expected and row['attention_segments']==expected//5
        and row['primary_side_groups']==n+branches<=expected//5
        and row['actual_q_partition']==list(range(5,expected+1,5))
        and detail['enabled'] is True and detail['extent']==kv
        and row['fixed_KV_extent']==[kv]*(expected//5))
