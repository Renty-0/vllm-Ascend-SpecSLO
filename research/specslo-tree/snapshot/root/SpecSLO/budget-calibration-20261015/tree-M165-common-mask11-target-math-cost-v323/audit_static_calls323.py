"""HOST reconstruction of actual node calls, including natural cold validation."""
def expected_grid(case):
    import json
    from pathlib import Path
    prefixes=json.loads(Path(__file__).with_name('FIXTURE323.json').read_text())['prefixes']
    count=sum(len(prefixes[i])%128+case.get('anchor',0)+(b-4)+1
              for i,b in enumerate(case['budgets'])if b>4)
    return next(b for b in(128,256,512,1024,1792)if b>=count)if count else 0

def audit_calls(cases,calls):
    from projection285 import plan
    cursor=0;seen=set();extra=0;families=[]
    for case in cases:
        branch=any(b>4 for b in case['budgets']);capacity=11
        sizes=[]
        for b in case['budgets']:sizes.extend([5,b-4]if b>4 else[b+1])
        sizes.extend(case['padding']);ends=[];offset=0
        for q in sizes:offset+=q;ends.append(offset)
        physical=plan(ends,case['query_count'])
        grid=expected_grid(case);family=(capacity,grid)
        cold=family not in seen;count=3 if cold else 1
        group=calls[cursor:cursor+count]
        if len(group)!=count:raise RuntimeError('missing real or original cold-validation calls')
        for call in group:
            if not(call['N']==case['N']and call['real_queries']==case['query_count']
                   and call['M']==165 and call['attention_segments']==33
                   and call['actual_seq_lengths_q']==list(range(5,166,5))
                   and call['sequence_lens']==[2560]*33
                   and call['mask_shape']==[33,1,capacity,4096]
                   and call['copy_enabled']is branch and call['used_aclgraph']is True
                   and call['copy_grid']==grid
                   and ((f'|ROOT296:copy-grid:{grid}'in call['actual_capture_key'])if branch else('|ROOT296:copy-grid:'not in call['actual_capture_key']))
                   and call['fallback_reason']is None
                   and call['physical_to_original']==physical['physical_to_original']
                   and call['original_to_physical']==physical['original_to_physical']
                   and call['real_moves']==physical['real_moves']
                   and call['primary_side_groups']==physical['supported_primary_side_groups']):
                raise RuntimeError('actual static dispatch does not equal independently rebuilt original real-node mapping')
        modes=[c['execution_mode']for c in group]
        if modes!=(['capture_replay','replay','replay']if cold else['replay']):
            raise RuntimeError('capture/change/restore or existing replay sequence differs')
        if cold:families.append(dict(mask_capacity=capacity,copy_grid=grid));extra+=2
        seen.add(family);cursor+=count
    if cursor!=len(calls)or seen!={(11,0),(11,128),(11,256),(11,512),(11,1024),(11,1792)}:
        raise RuntimeError('unexplained extra actual dispatches or missing declared graph family')
    return dict(positive_node_cases=len(cases),actual_graph_dispatches=len(calls),
                actual_cold_families=families,original_changed_input_restore_extra_dispatches=extra,
                all_real_and_cold_calls_reconstructed=True,all165_row_permutations_exact=True,
                Native_flags_assigned=False,whole_workload_AR_qualified=False)
