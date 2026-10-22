"""Bind the existing fully checked target; never reinstall its backend."""
from contextlib import contextmanager


def install(engine,native):
    if (getattr(engine,'_v173_qualified_component',None) is None
            or getattr(engine,'_uniform_KV_bucket425_proof',{}).get('cases')!=681
            or engine.cache_allocation is not None or engine.graph_runner.graph_cache_sealed):
        raise RuntimeError('ROOT425 original173+227+681 and released current cache before E4 proof')
    state=None if engine.is_draft else engine._branch_tail_state
    if state is not None and not state.uniform_KV_bucket425_enabled:
        raise RuntimeError('ROOT425 current checked uniform target state required')
    descriptor=dict(model_id=id(engine.model),target_state=state,
        actual_pad=native.NativePearlEngine._pad_spec_rhythm_target_tree_graph,
        actual_forward=native.NativePearlEngine.target_tree_forward,
        original173=engine._v173_qualified_component,actual_metadata_matches=0)
    engine._parent4_target391_binding=descriptor
    return dict(actual_best376_target_backend=True,best376_draft_claimed=False,
        original227_and681_run=True,reinstalled_target=False,
        exact_existing_current_model=True,Native_flags_assigned=False)


@contextmanager
def diagnostic(engine,case):
    from vllm_ascend.spec_decode.pearl import native_engine as native
    import live419
    current=getattr(engine,'_parent4_target391_binding',None)
    if not (current is not None and current['model_id']==id(engine.model)
            and current['original173'] is engine._v173_qualified_component
            and current['actual_pad'] is native.NativePearlEngine._pad_spec_rhythm_target_tree_graph
            and current['actual_forward'] is native.NativePearlEngine.target_tree_forward):
        raise RuntimeError('ROOT425 exact existing target callbacks and model binding required')
    if not engine.is_draft and not (current['target_state'] is engine._branch_tail_state
            and current['target_state'].uniform_KV_bucket425_enabled):
        raise RuntimeError('ROOT425 same actual qualified target state')
    with live419.diagnostic(engine,case):yield
