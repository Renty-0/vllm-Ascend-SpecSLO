"""HOST reconstruction of actual node calls, including natural cold validation."""
def audit_calls(cases,calls):
    from projection249 import plan
    cursor=0;seen=set();extra=0;families=[]
    for case in cases:
        branch=5 in case['budgets'];capacity=11 if branch else 10
        sizes=[]
        for b in case['budgets']:sizes.extend([5,1]if b==5 else[b+1])
        sizes.extend(case['padding']);ends=[];offset=0
        for q in sizes:offset+=q;ends.append(offset)
        physical=plan(ends,case['query_count'])
        cold=capacity not in seen;count=3 if cold else 1
        group=calls[cursor:cursor+count]
        if len(group)!=count:raise RuntimeError('missing real or original cold-validation calls')
        for call in group:
            if not(call['N']==case['N']and call['real_queries']==case['query_count']
                   and call['M']==128 and call['attention_segments']==26
                   and call['actual_seq_lengths_q']==list(range(5,126,5))+[128]
                   and call['sequence_lens']==[2560]*26
                   and call['mask_shape']==[26,1,capacity,4096]
                   and call['copy_enabled']is branch and call['used_aclgraph']is True
                   and call['fallback_reason']is None
                   and call['physical_to_original']==physical['physical_to_original']
                   and call['original_to_physical']==physical['original_to_physical']
                   and call['real_moves']==physical['real_moves']
                   and call['primary_side_groups']==physical['supported_primary_side_groups']):
                raise RuntimeError('actual static dispatch does not equal independently rebuilt original real-node mapping')
        modes=[c['execution_mode']for c in group]
        if modes!=(['capture_replay','replay','replay']if cold else['replay']):
            raise RuntimeError('capture/change/restore or existing replay sequence differs')
        if cold:families.append(capacity);extra+=2
        seen.add(capacity);cursor+=count
    if cursor!=len(calls)or sorted(families)!=[10,11]:
        raise RuntimeError('unexplained extra actual dispatches or missing declared graph family')
    return dict(positive_node_cases=len(cases),actual_graph_dispatches=len(calls),
                actual_cold_families=families,original_changed_input_restore_extra_dispatches=extra,
                all_real_and_cold_calls_reconstructed=True,all128_row_permutations_exact=True,
                Native_flags_assigned=False,whole_workload_AR_qualified=False)
