"""Rebuild fabricated HOST dispatch records; no model execution evidence."""
import copy
import json
from pathlib import Path
import runtime344
from domain344 import cases,prelude,short_cases,short_prefixes,mid_prefixes,family
from bucket344 import case_extent
from dispatch344 import audit_calls
from projection255 import plan

P=Path(__file__).resolve().parent


def records(selected,prefixes,enabled,cold):
    result=[];seen=set()
    for case in selected:
        capacity,grid=family(case)
        sizes=[]
        for b in case['budgets']:sizes.extend([5,b-4]if b>4 else[b+1])
        sizes.extend(case['padding']);ends=[];position=0
        for size in sizes:position+=size;ends.append(position)
        mapping=plan(ends,case['query_count'])
        e=case_extent(case,prefixes)if enabled else 2560
        first=cold and(capacity,grid)not in seen
        for index in range(3 if first else 1):
            result.append(dict(N=case['N'],real_queries=case['query_count'],M=120,
                attention_segments=24,actual_seq_lengths_q=list(range(5,121,5)),
                sequence_lens=[e]*24,mask_shape=[24,1,capacity,4096],copy_enabled=bool(grid),copy_grid=grid,
                actual_capture_key='HOST'+(f'|ROOT296:copy-grid:{grid}'if grid else ''),
                execution_mode='capture_replay'if first and index==0 else'replay',
                used_aclgraph=True,fallback_reason=None,
                physical_to_original=mapping['physical_to_original'],original_to_physical=mapping['original_to_physical'],
                real_moves=mapping['real_moves'],primary_side_groups=mapping['supported_primary_side_groups'],
                uniform_KV_bucket344=dict(enabled=enabled,extent=e,
                    original_max_context=max(len(prefixes[i])+min(b,4)+1 for i,b in enumerate(case['budgets'])))))
        seen.add((capacity,grid))
    return result


def main():
    fixture=json.loads((runtime344.original.P/'FIXTURE311.json').read_text())
    all_records=[]
    for selected,prefixes,enabled,cold in[(prelude(),fixture['prefixes'],False,True),
            (cases(),fixture['prefixes'],True,False),
            (short_cases(),mid_prefixes(fixture['prefixes']),True,False),
            (short_cases(),short_prefixes(fixture['prefixes']),True,False)]:
        calls=records(selected,prefixes,enabled,cold)
        value=audit_calls(selected,calls,prefixes,enabled=enabled)
        assert value['positive_node_cases']==len(selected)
        all_records.append(len(calls))
    selected=prelude();prefixes=fixture['prefixes'];good=records(selected,prefixes,False,True)
    mutations=[lambda x:x[0]['sequence_lens'].__setitem__(0,512),
        lambda x:x[0].update(used_aclgraph=False),lambda x:x.pop(),
        lambda x:x[0]['physical_to_original'].__setitem__(0,119),
        lambda x:x[0]['original_to_physical'].__setitem__(0,119),
        lambda x:x[0].update(execution_mode='replay'),
        lambda x:x[3].update(actual_capture_key='missing'),
        lambda x:x[0]['uniform_KV_bucket344'].update(original_max_context=2561)]
    for mutate in mutations:
        bad=copy.deepcopy(good);mutate(bad)
        try:audit_calls(selected,bad,prefixes,enabled=False)
        except RuntimeError:pass
        else:raise AssertionError('corrupt HOST dispatch accepted')
    result=dict(status='passed_HOST_dispatch_schema_only',NPU_used=False,
        fabricated_records_per_phase=all_records,corrupt_records_rejected=len(mutations),
        original120row_bijection_checker_retained=True,no_model_or_graph_qualification=True)
    (P/'HOST344_DISPATCH_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
