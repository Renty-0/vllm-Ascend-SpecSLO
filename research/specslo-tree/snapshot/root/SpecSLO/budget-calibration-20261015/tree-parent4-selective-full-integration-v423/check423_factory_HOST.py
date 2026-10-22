"""Actual original installers with explicit class doubles; no Native models."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    import torch
    import torch_npu
    def blocked(*a,**k):raise AssertionError('HOST423 may not initialize an NPU')
    torch.npu._lazy_init=blocked;torch_npu.npu._lazy_init=blocked
    import factory423 as candidate
    assert candidate._FACTORY is None
    prepared=candidate.prepare_helpers()
    import root_seed
    called=[];sentinel=object()
    class Runner:
        def _execute_draft(self,*a,**k):called.append('original_execute');return sentinel
    class Engine:
        def draft_tree_forward(self,*a,**k):called.append('original_native');return sentinel
        def _allocate_cache(self,*a,**k):called.append('original_allocate');return sentinel
    native=SimpleNamespace(NativeACLGraphRunner=Runner,NativePearlEngine=Engine)
    root_seed.install(native)
    current=candidate.current(native)
    assert inspect.getclosurevars(Engine.draft_tree_forward).nonlocals['original'] is current['dispatch']
    assert inspect.getclosurevars(current['fallback']).nonlocals['original'] is current['original_native']
    assert inspect.getclosurevars(Runner._execute_draft).nonlocals['original_execute'].__qualname__.endswith('Runner._execute_draft')
    engine=Engine();engine.is_draft=True
    assert engine._allocate_cache() is sentinel
    assert called==['original_allocate']
    assert engine._dynamic_tree_known_KV=={} and engine._dynamic_tree_fast_profile_calls==0
    assert Runner()._execute_draft(return_confidence=False) is sentinel
    plan=SimpleNamespace(width=1,depth=4,candidate_budget=3)
    assert engine.draft_tree_forward([plan],[7],[0]) is sentinel
    assert called==['original_allocate','original_execute','original_native']
    engine.is_draft=False
    assert engine.draft_tree_forward([plan],[7],[0]) is None
    engine.is_draft=True
    plan=SimpleNamespace(width=2,depth=4,candidate_budget=4)
    before=list(called)
    for forged in (None,(object(),object())):
        if forged is None:
            engine.__dict__.pop('_v199_diagnostic',None)
        else:
            engine._v199_diagnostic=forged;engine._v173_qualified_component=object()
        try:engine.draft_tree_forward([plan],[7],[0],eager_parent_sources={0:(plan,[1,2,3,4])})
        except RuntimeError as error:assert 'current engine/model proof record' in str(error)
        else:raise AssertionError('unissued E4 capability executed')
        assert called==before
    assert candidate.current(native) is current
    value=dict(status='PASS_HOST423_PREINSTALL_ORIGINAL_ROOTSEED_AND_SINGLE_FAST_HOOKS',
        actual_original_RootSeed_and_FastV5_installers_called=True,
        explicit_class_doubles=True,Native_models_constructed=0,
        original_runner_and_cache_wrapped_once=True,
        original_unsupported_topology_fallback_identity=True,
        unissued_or_forged_E4_permission_rejected_before_model=True,
        no_after_install_closure_mutation=True,source=candidate.descriptor(),
        current_production_permission=False,NPU_used=False,Raw=None,Goodput=None)
    p=Path(__file__).resolve().parent
    (p/'HOST423_FACTORY_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
