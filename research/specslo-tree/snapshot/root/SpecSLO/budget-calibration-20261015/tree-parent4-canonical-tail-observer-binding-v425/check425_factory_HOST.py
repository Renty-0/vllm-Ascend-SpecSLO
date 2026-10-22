"""Actual original installers with explicit class doubles; no Native models."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace


def main():
    import torch
    import torch_npu
    def blocked(*a,**k):raise AssertionError('HOST425 may not initialize an NPU')
    torch.npu._lazy_init=blocked;torch_npu.npu._lazy_init=blocked
    import factory425 as candidate
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
    import tail_hidden_observer
    graph=SimpleNamespace(_CAPTURED_TASKS=object())
    original_top2=tail_hidden_observer.install(graph)
    import fast_tree_draft_v5 as canonical
    assert current['fallback'].__globals__ is vars(canonical)
    assert current['runner_execute'].__globals__ is vars(canonical)
    assert canonical.execute_top2 is not original_top2
    class HostModel:
        def __call__(self,inputs,positions,metadata):
            return torch.stack((inputs.float()+positions.float(), torch.ones_like(inputs).float()),dim=1)
        def compute_logits(self,hidden):
            return hidden[:,0:1]+torch.arange(7,dtype=torch.float32)[None,:]
    runner=Runner();runner.model=HostModel();runner._dynamic_tree_top2=True
    key=('draft-greedy:7|steps:4|paged|confidence',2)
    runner.last_draft_entry_key=key
    inputs=torch.tensor([1,2],dtype=torch.long)
    positions=[torch.tensor([j,j+1],dtype=torch.long) for j in range(4)]
    metadata=[object() for j in range(4)]
    captured=runner._execute_draft(inputs,positions,metadata,7,return_confidence=True)
    graph._CAPTURED_TASKS=None
    eager=runner._execute_draft(inputs,positions,metadata,7,return_confidence=True)
    hidden,intervals,eager_intervals=tail_hidden_observer.snapshot(runner,key,4)
    assert len(hidden)==len(intervals)==len(eager_intervals)==4
    assert torch.equal(captured,eager) and tuple(captured.shape)==(2,4,2)
    value=dict(status='PASS_HOST425_PREINSTALL_ORIGINAL_ROOTSEED_AND_SINGLE_FAST_HOOKS',
        actual_original_RootSeed_and_FastV5_installers_called=True,
        explicit_class_doubles=True,Native_models_constructed=0,
        original_runner_and_cache_wrapped_once=True,
        original_unsupported_topology_fallback_identity=True,
        unissued_or_forged_E4_permission_rejected_before_model=True,
        no_after_install_closure_mutation=True,source=candidate.descriptor(),
        actual_original_tail_observer_capture_eager_and_snapshot_called=True,
        CPU_model_double_only_for_observer_binding=True,
        current_production_permission=False,NPU_used=False,Raw=None,Goodput=None)
    p=Path(__file__).resolve().parent
    (p/'HOST425_FACTORY_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
