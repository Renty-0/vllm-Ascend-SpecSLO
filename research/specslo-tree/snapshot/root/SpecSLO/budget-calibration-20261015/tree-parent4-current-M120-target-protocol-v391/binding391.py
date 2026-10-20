"""Finite current-state binding; no new production capability or Native flag."""
from contextlib import contextmanager


def install(engine,native):
    if not getattr(engine,'_v173_qualified_component',None):
        raise RuntimeError('ROOT391 fresh original173 baseline required')
    if engine.cache_allocation is not None or engine.graph_runner.graph_cache_sealed:
        raise RuntimeError('ROOT391 independent target epoch before seal with released cache')
    from bucket376 import install as original
    value=original(engine,native)
    if not engine.is_draft:
        state=engine._branch_tail_state
        state.uniform_KV_bucket376_enabled=True
    engine._parent4_target391_binding=dict(model_id=id(engine.model),target_state=None if engine.is_draft else engine._branch_tail_state,
        actual_pad=native.NativePearlEngine._pad_spec_rhythm_target_tree_graph,
        original173=engine._v173_qualified_component,activation=value)
    return dict(activation=value,finite_parent4_child4_scope_only=True,
        actual_best376_target_backend=True,best376_draft_claimed=False,
        original227_and681_replaced=False,original227_and681_run=False,
        Native_flags_assigned=False,production_permission=False,Raw=None,Goodput=None)


@contextmanager
def diagnostic(engine,case):
    from vllm_ascend.spec_decode.pearl import native_engine as native
    import live199
    value=getattr(engine,'_parent4_target391_binding',None)
    if not(value and value['model_id']==id(engine.model)
            and value['original173']is engine._v173_qualified_component
            and value['actual_pad']is native.NativePearlEngine._pad_spec_rhythm_target_tree_graph):
        raise RuntimeError('ROOT391 exact current local binding required')
    if not engine.is_draft:
        if value['target_state']is not engine._branch_tail_state or not value['target_state'].uniform_KV_bucket376_enabled:
            raise RuntimeError('ROOT391 same state and actual current KV bucket policy')
    with live199.diagnostic(engine,case):yield
