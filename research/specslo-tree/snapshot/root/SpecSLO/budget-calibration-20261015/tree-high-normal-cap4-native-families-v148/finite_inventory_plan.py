"""CPU-only enumeration of the admitted service domain; no graph flags."""
from target_layout import envelope, legacy_envelope, STATIC_NATIVE_KEYS

VOCAB = 151936
FRONTIER_BUCKETS = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32)
CHAIN_BUCKETS = FRONTIER_BUCKETS + (36, 40, 44, 48, 64)
HEAD_BUCKETS = CHAIN_BUCKETS  # Existing import name now denotes capacity64 chain rows.
PA_BUCKETS = CHAIN_BUCKETS + (128,)
MC2_ROWS = (40,44,48,52,56,60,64,68,72,76,80,84,88,92,96,100,
            116,120,124,128,165,197)


def target_key(family, rows, segments):
    if family == 'full9':
        attention = ('tree-fia-full:exact-length-task-update|buffers:'
                     + repr((((segments,1,9,4096),(segments,32)),)))
    elif family == 'causal':
        attention = ('fia:exact-length-task-update|buffers:'
                     + repr((((segments,32),(segments,32),(2048,2048)),)))
    else:
        raise ValueError('Unknown native target family')
    return (f'target-greedy:{VOCAB}|steps:1|{attention}', (rows,))


def budgets(count, queries, family):
    """Construct a real admitted B1..5 vector, not a padding-only shape."""
    maximum = 4 if family == 'causal' else 5
    result = [1]*count
    remaining = queries - 2*count
    if family == 'full9':
        if remaining < 4:
            raise ValueError('A genuine fork requires a B5 row')
        result[0] = 5
        remaining -= 4
    for index in range(count):
        extra = min(remaining, maximum-result[index])
        result[index] += extra
        remaining -= extra
    if remaining or sum(result)>96 or sum(result)+count != queries:
        raise ValueError('Unreachable admitted budget vector')
    return tuple(result)


def targets():
    witnesses = {}
    for count in range(1,33):
        for queries in range(2*count, min(6*count,count+96,128)+1):
            for family, possible in [('causal',queries<=5*count),
                                     ('full9',queries>=2*count+4)]:
                if not possible:
                    continue
                rows, segments, scratch = envelope(count, queries, family)
                values = budgets(count, queries, family)
                key = target_key(family, rows, segments)
                fixture = dict(family=family, physical_queries=rows,
                    physical_segments=segments, real_count=count,
                    real_queries=queries, budgets=list(values),
                    scratch_query_lengths=scratch,
                    physical_query_lengths=[b+1 for b in values]+list(scratch),
                    physical_KV_lengths=[127+b+1 for b in values]+list(scratch),
                    expected_key=repr(key))
                # Prefer a key's actual real rows rather than gratuitous
                # scratch requests. This remains a real service witness.
                score = (len(scratch),-count,-queries)
                witnesses.setdefault(key,[]).append((score,fixture))
    result=[]
    for key,candidates in sorted(witnesses.items(),key=lambda item:(item[0][1][0],item[0][0])):
        candidates.sort(key=lambda item:item[0])
        primary=dict(candidates[0][1])
        alternatives=[row[1] for row in candidates[1:]]
        primary['reachable_N_Q_pairs']=len(candidates)
        if alternatives:
            # Prefer a different real request count, then a genuinely changed
            # exact query partition; every choice still uses original padding.
            alternate=max(alternatives,key=lambda f:(
                int(f['physical_query_lengths']!=primary['physical_query_lengths']),
                int(f['real_count']!=primary['real_count']),
                abs(f['real_count']-primary['real_count']),
                abs(f['real_queries']-primary['real_queries'])))
            primary['alternate_witness']=alternate
            primary['alternate_changes_query_partition']=(alternate['physical_query_lengths']!=primary['physical_query_lengths'])
            primary['alternate_changes_KV_lengths']=(alternate['physical_KV_lengths']!=primary['physical_KV_lengths'])
            primary['partition_scope']='two_distinct_admitted_N_Q_witnesses'
        else:
            # Preserve explicit exceptions where the bounded mapping admits
            # only one real N/Q and one budget vector for this native key.
            primary['alternate_witness']=None
            primary['alternate_changes_query_partition']=False
            primary['alternate_changes_KV_lengths']=False
            primary['partition_scope']='unique_admitted_N_Q_and_budget_vector'
        result.append(primary)
    return result


def draft_keys():
    return {
        'chain': [(f'draft-greedy:{VOCAB}|steps:{steps}|paged|confidence',rows)
                  for rows in CHAIN_BUCKETS for steps in range(1,5)],
        'hidden_PA': [('target-hidden:0|steps:1|paged',(rows,)) for rows in PA_BUCKETS],
        'frontier_PA': [(f'target-greedy:{VOCAB}|steps:1|paged',(rows,))
                        for rows in FRONTIER_BUCKETS],
    }


def manifest():
    fixtures = targets()
    return dict(target_fixture_count=len(fixtures),
        target_families={family:sum(f['family']==family for f in fixtures)
                         for family in ('full9','causal')},
        target_decode_keys=[f['expected_key'] for f in fixtures],
        target_fixtures=fixtures, draft_keys={k:list(map(repr,v)) for k,v in draft_keys().items()},
        target_keys_with_two_dynamic_partition_witnesses=sum(f['alternate_witness'] is not None for f in fixtures),
        target_keys_with_unique_partition=sum(f['alternate_witness'] is None for f in fixtures),
        draft_explicit_keys=86, draft_native_existing_generic_regression_keys=28,
        expected_target_union_with_201_exact_prefill=201+len(fixtures),
        max_native_entries=512, candidate_roof=96, root_inclusive_limit=128,
        physical_MC2_rows=sorted({f['physical_queries'] for f in fixtures}),
        bounded_padding=True, historical_native_keys=32,
        static_preferred_native_keys=[list(key) for key in sorted(STATIC_NATIVE_KEYS)],
        selection_uses_runtime_cache_contents=False,
        original_native_qualification_transferred=False,
        admitted_family_geometries=sum(1 for count in range(1,33)
            for queries in range(2*count,min(6*count,count+96,128)+1)
            for family,possible in [('causal',queries<=5*count),('full9',queries>=2*count+4)] if possible),
        note='Counts are planned native keys; actual native entry keys and flags must be checked on device')
