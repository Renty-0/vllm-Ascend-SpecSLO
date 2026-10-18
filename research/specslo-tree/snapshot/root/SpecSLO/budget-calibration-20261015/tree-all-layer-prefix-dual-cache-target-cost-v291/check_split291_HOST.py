"""Exercise the installed prepare callback on all408 real target geometries."""
import json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;B=P.parent
import check291
sys.path[:0]=[str(B/'tree-deferred-FULL-mask-host-projection-v259'),str(B/'tree-prefix-precopy-dual-cachewrite-host-v288')]
def main():
    from fixture259 import build
    env=build();e=env['engine'];native=env['native']
    from static_state291 import bind_checked_prepare
    import branch_tail255
    bind_checked_prepare(e._branch_tail_state,branch_tail255)
    from plan269 import make
    from padding269 import pad
    from oracle291 import cases
    from split288 import split
    from spans292 import pack as pack_spans
    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph=pad
    results=[]
    for case in cases():
        budgets=case['budgets'];n=len(budgets)
        plans=[make(b,127,case['anchor'])for b in budgets]
        roots=list(range(41,41+n));rows=[[103+17*i+29*j for j in range(b)]for i,b in enumerate(budgets)]
        pp,rr,dd,ss,real=pad(e,plans,roots,rows,list(range(n)),[],[])
        out=native.NativePearlEngine.target_tree_forward(e,pp,rr,dd,sequence_ids=ss,return_logits=False,real_tree_count=real)
        state=e._branch_tail_state;physical=state.static_current['physical']
        if any(b>4 for b in budgets):
            desc=state.copy291_layout_HOST;slots=physical[2].slot_mapping.tolist()
            expected=split(desc['source'],desc['dest'],slots)
            assert state.split291_last_HOST['query_slots']==slots
            assert state.prefix291_source.tolist()==expected['prefix_source']
            assert state.alias291_destination.tolist()==expected['query_alias_destination']
            assert state.prefix291_spans.tolist()==pack_spans(expected['prefix_source'],expected['prefix_destination'])['spans']
            assert expected['current_aliases']==sum(case['anchor']+b-3 for b in budgets if b>4),(case,expected['current_aliases'],[p.parent_indices.tolist()for p in plans])
        else:assert state.split291_last_HOST is None
        assert out['num_draft_tokens']==budgets and out['query_count']==120
        results.append(dict(name=case['name'],N=n,anchor=case['anchor'],split=state.split291_last_HOST))
    assert len(results)==408 and 'torch_npu'not in sys.modules
    result=dict(status='HOST_actual408_Native_prepare_split_callback_and_physical_writer_bijection_passed',
        actual_cases=408,all_legacy_and_new_branch_anchors_checked=True,
        HOST_derived_query_slots_equal_actual_physical_metadata=True,no_device_metadata_read=True,
        persistent_prefix_and_alias_buffers_exact=True,cases=results,NPU_used=False,kernel_executed=False,Goodput=None)
    with(P/'HOST_SPLIT291_SPANS_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items()if k!='cases'}))
if __name__=='__main__':main()
