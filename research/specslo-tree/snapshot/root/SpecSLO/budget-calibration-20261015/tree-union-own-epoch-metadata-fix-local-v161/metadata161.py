"""Diagnostic exact metadata gates, retaining each allocation's own pages.

No model operation or Native flag changes. Only physical page identifiers are
not required to survive release/reallocate; every other original field remains
exact, and each epoch's pages/slots are checked against its current HOST lease.
"""
from copy import deepcopy

FIELDS=('inputs','positions','slot_mapping','context_lens','block_tables','host_Q','host_KV',
        'use_fused_infer_attention','has_attention_mask','has_request_tables')

def require(ok,message):
    if ok is not True:raise RuntimeError('V161: '+message)

def values(v):
    if hasattr(v,'detach'):v=v.detach()
    if hasattr(v,'cpu'):v=v.cpu()
    if hasattr(v,'tolist'):v=v.tolist()
    return deepcopy(v)

def snapshot(inputs,positions,meta):
    return dict(inputs=values(inputs),positions=values(positions),slot_mapping=values(meta.slot_mapping),
        context_lens=values(meta.context_lens),block_tables=values(meta.block_tables),
        host_Q=list(meta.actual_seq_lengths_q),host_KV=list(meta.sequence_lens),
        use_fused_infer_attention=meta.use_fused_infer_attention,
        has_attention_mask=meta.attention_mask is not None,has_request_tables=meta.request_block_tables is not None)

def observed_snapshot(row):
    require(set(row)==set(FIELDS),'actual observer exact metadata schema')
    return {k:list(row[k]) if k in ('host_Q','host_KV') else values(row[k]) for k in FIELDS}

def owner_geometry(case):
    require(case.normal==case.eager==32 and case.prefix==128 and case.repair_count==1 and case.union_M==36,
        'only unchanged P157 natural fixture R1/E32/F32/U36')
    owners=list(range(32,64))+list(case.repair_owners)
    positions=[case.prefix+3]*32+[case.prefix]*case.repair_count
    return owners,positions

def geometry(case,f):
    owners,positions=owner_geometry(case)
    # Genuine retained parent source, never a fabricated target/frontier token.
    tokens=[int(f['sources'][i][1][2]) for i in range(32,64)]+[int(f['catchups'][i][0]) for i in case.repair_owners]
    return owners,positions,tokens

def validate_own(actual,expected,tables,owners,positions,block_size):
    """All raw fields exact inside this allocation; page ownership is separate."""
    require(set(actual)==set(expected)==set(FIELDS),'complete metadata fields')
    fields={k:actual[k]==expected[k] for k in FIELDS}
    real=len(owners);physical=len(actual['inputs'])
    require(len(positions)==real and len(set(owners))==real,'independent owner geometry')
    slots=[tables[i][p//block_size]*block_size+p%block_size for i,p in zip(owners,positions,strict=True)]
    owner_tables=[list(tables[i]) for i in owners]
    pages=[page for row in owner_tables for page in row if page>=0]
    ownership=dict(real_slots=actual['slot_mapping'][:real]==slots,
        real_tables=actual['block_tables'][:real]==owner_tables,
        distinct_leased_pages=len(pages)==len(set(pages)),
        distinct_write_slots=len(slots)==len(set(slots)),
        finite_dummy_slots=actual['slot_mapping'][real:]==[-1]*(physical-real),
        real_positions=actual['positions'][:real]==positions)
    result=dict(field_equal=fields,own_lease=ownership,owners=list(owners),logical_positions=list(positions),
        actual=actual,independent_original_PA=expected,all_fields_exact=all(fields.values()),
        own_physical_mapping_exact=all(ownership.values()),physical_rows=physical,real_rows=real)
    return result

def semantic(proof):
    """Comparable logical metadata, after strict own-epoch physical validation."""
    require(proof['all_fields_exact'] is proof['own_physical_mapping_exact'] is True,'closed own-epoch exact metadata gates')
    raw=proof['actual'];real=proof['real_rows']
    return {k:raw[k] for k in FIELDS if k not in ('slot_mapping','block_tables')} | dict(
        owners=proof['owners'],logical_positions=proof['logical_positions'],
        table_widths=[len(t) for t in raw['block_tables']],
        write_validity=[True]*real+[False]*(proof['physical_rows']-real))

def capture_actual(e,case,f,seen,PA):
    """Independent original PA producer while the observed lease still exists."""
    require(len(seen['hidden'])==1,'one genuine current union hidden call')
    allocation=e.cache_allocation;device=e.cache_block_tables
    tables=deepcopy(allocation.block_tables);owners,positions,tokens=geometry(case,f)
    # This is the existing independent P152 PA function, which executes original
    # ensure -> prepare_metadata -> Native pad; no candidate union constructor.
    expected=snapshot(*PA(e,owners,positions,tokens,case.union_M))
    require(e.cache_allocation is allocation and e.cache_block_tables is device and allocation.block_tables==tables,
        'metadata-only reference must retain current actual allocation/pages')
    actual=observed_snapshot(seen['hidden'][0]['metadata'])
    proof=validate_own(actual,expected,tables,owners,positions,e.config.kvcache_block_size)
    proof.update(phase='actual-allocation-before-fresh-reference',allocation_id=id(allocation),
        original_PA_producer=True,candidate_union_builder_used_as_reference=False,model_calls_added=0)
    # Save every field before asserting, so unexpected failure retains evidence.
    e._v151_partial['actual_union_metadata_diagnostic']=proof
    require(proof['all_fields_exact'] and proof['own_physical_mapping_exact'],'actual own-lease metadata exact field/owner/slot/table/padding gates')
    return proof

def compare_fresh(e,case,actual_proof,reference_call):
    owners,positions=owner_geometry(case);record=snapshot(*reference_call)
    # The reference_call itself came from independent PA on fresh history. Its
    # physical pages must match THIS allocation, never the former candidate.
    proof=validate_own(record,record,deepcopy(e.cache_allocation.block_tables),owners,positions,e.config.kvcache_block_size)
    proof.update(phase='fresh-independent-reference-allocation',allocation_id=id(e.cache_allocation),
        original_PA_producer=True,candidate_pages_or_KV_copied=False)
    comparable=semantic(actual_proof)==semantic(proof)
    e._v151_partial['reference_union_metadata_diagnostic']=proof
    e._v151_partial['cross_epoch_union_metadata_semantics_equal']=comparable
    require(comparable,'fresh oracle exact logical metadata; physical page IDs independently leased')
    return dict(actual=actual_proof,reference=proof,logical_metadata_exact=True,
        raw_page_ID_equality_required=False,own_physical_mapping_exact=True)
