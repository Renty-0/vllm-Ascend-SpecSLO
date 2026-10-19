"""Actual Native/305 prepare on CPU; all408 layouts at both legal extents."""
import json
from pathlib import Path
import sys

P=Path(__file__).resolve().parent;B=P.parent
sys.path.insert(0,str(B/'tree-single-prepare-target-math-cost-v311'))
import runtime311
sys.path[:0]=[str(B/'tree-deferred-FULL-mask-host-projection-v259'),
              str(B/'tree-prefix-precopy-dual-cachewrite-host-v288'),
              str(B/'tree-single-validated-prefix-descriptors-host-v304')]


def main():
    from fixture259 import build
    env=build();engine,native=env['engine'],env['native']
    from static_state305 import bind_checked_prepare
    import branch_tail255
    bind_checked_prepare(engine._branch_tail_state,branch_tail255)
    from bucket343 import bind,extent
    from domain343 import cases
    from plan269 import make
    from padding269 import pad
    native.NativePearlEngine._pad_spec_rhythm_target_tree_graph=pad
    # The frozen CPU receiver asserts the old2560 contract. Vary only its
    # expected literal in this test process; actual Native code is untouched.
    from types import FunctionType,MethodType
    pending=[engine.graph_runner.run_target_greedy];seen=set();receivers=[]
    while pending:
        fn=pending.pop()
        if isinstance(fn,MethodType):fn=fn.__func__
        if not isinstance(fn,FunctionType)or id(fn)in seen:continue
        seen.add(id(fn))
        if '::original-fixture-prefix' in fn.__code__.co_filename and fn.__name__=='run_target_greedy':receivers.append(fn)
        pending.extend(cell.cell_contents for cell in fn.__closure__ or ())
    assert len(receivers)==1
    receiver=receivers[0];original_receiver_code=receiver.__code__
    bind(engine)
    results=[]
    for case in cases():
        budgets=case['budgets'];n=len(budgets)
        plans=[make(b,127,case['anchor'])for b in budgets]
        roots=list(range(41,41+n));rows=[[103+17*i+29*j for j in range(b)]for i,b in enumerate(budgets)]
        maps=[]
        for enabled in (False,True):
            receiver.__code__=original_receiver_code.replace(co_consts=tuple(
                ((512 if enabled else 2560),)*24 if type(x)is tuple and x==(2560,)*24 else
                (512 if enabled else 2560)if type(x)is int and x==2560 else x
                for x in original_receiver_code.co_consts))
            state=engine._branch_tail_state;state.uniform_bucket343_enabled=enabled
            pp,rr,dd,ss,real=pad(engine,plans,roots,rows,list(range(n)),[],[])
            out=native.NativePearlEngine.target_tree_forward(engine,pp,rr,dd,
                sequence_ids=ss,return_logits=False,real_tree_count=real)
            physical=state.static_current['physical'];metadata=physical[2]
            assert out['num_draft_tokens']==budgets and out['query_count']==120
            assert tuple(metadata.sequence_lens)==((512 if enabled else 2560),)*24
            assert metadata.actual_seq_lengths_q==tuple(range(5,121,5))
            maps.append(dict(mapping=physical[4],slots=metadata.slot_mapping.tolist(),
                aliases=state.alias305_destination.tolist()if state.enabled else None,
                split=state.split305_last_HOST))
        assert maps[0]==maps[1]
        results.append(case['name'])
    bad=0
    for value in ([],[0],[-1],[2561],[True]):
        try:extent(value)
        except ValueError:bad+=1
        else:raise AssertionError('invalid extent input accepted')
    receiver.__code__=original_receiver_code
    assert len(results)==408 and bad==5 and 'torch_npu'not in sys.modules
    result=dict(status='passed_actual408_Native_prepare_CPU',NPU_used=False,
        model_executed=False,actual_layouts=408,actual_forwards=816,
        physical_mapping_slots_and_checked_copy_splits_identical=True,
        all_original_contexts_fit512=True,only_toy_receiver_expected_extent_varied=True,invalid_inputs_rejected=bad)
    (P/'HOST343_PREPARE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
