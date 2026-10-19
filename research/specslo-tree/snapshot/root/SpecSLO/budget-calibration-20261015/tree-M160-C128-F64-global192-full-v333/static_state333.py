"""Exact326 physical geometry with actual165 MC2 projection adapter."""
from static_state326 import ensure_zero, bind_checked_prepare
from static_state326 import install as original_install
from static_state326 import require_copy_bindings as original_bindings


def install(engine,native):
    activation=original_install(engine,native)
    engine._ROOT333_bridge=None
    engine._ROOT333_bridge_bindings=()
    if not engine.is_draft:
        from bridge334 import install as bridge_install
        from vllm_ascend.spec_decode.pearl import native_model
        bridge=bridge_install(engine.model,native_model)
        engine._ROOT333_bridge=bridge
        bindings=[(engine.model,'forward',engine.model.forward)]
        for layer in engine.model.layers:
            bindings.extend([(layer,'forward',layer.forward),
                             (layer.mlp,'forward_down_mc2',layer.mlp.forward_down_mc2)])
        for module in engine.model.modules():
            if isinstance(module,native_model.NativeRMSNorm):
                bindings.append((module,'forward_mc2',module.forward_mc2))
        engine._ROOT333_bridge_bindings=tuple(bindings)
    return dict(activation,actual_projection_rows165=True,FIA_rows160=True,
                new_profile160_claimed=False,original165_chained_flush_ABI=True,deferred_read_done=False,full_instance_qualified=False)


def require_copy_bindings(engine):
    original_bindings(engine)
    if not engine.is_draft:
        if engine.model._specslo_bridge330 is not engine._ROOT333_bridge:
            raise RuntimeError('ROOT333 actual MC2 bridge instance changed')
        if engine._ROOT333_bridge.pending or engine._ROOT333_bridge.active_layer is not None:
            raise RuntimeError('ROOT333 MC2 bridge scope/ticket not closed')
        for owner,name,value in engine._ROOT333_bridge_bindings:
            if getattr(owner,name)is not value:
                raise RuntimeError('ROOT333 qualified native function binding changed')
    return True
