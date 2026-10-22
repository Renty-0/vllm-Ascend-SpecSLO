"""Reproduce the actual423 observer wiring failure with a CPU model double."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace


def main():
    import torch
    B=Path(__file__).resolve().parent.parent
    sys.path.insert(0,str(B/'tree-parent4-selective-full-integration-v423'))
    import factory423 as previous
    previous.prepare_helpers()
    import root_seed
    class Runner:
        def _execute_draft(self,*a,**k):raise AssertionError('unexpected native fallback')
    class Engine:
        def draft_tree_forward(self,*a,**k):return None
        def _allocate_cache(self,*a,**k):return None
    native=SimpleNamespace(NativeACLGraphRunner=Runner,NativePearlEngine=Engine)
    root_seed.install(native)
    import fast_tree_draft_v5 as canonical
    current=previous.current(native)
    assert current['fallback'].__globals__ is not vars(canonical)
    import tail_hidden_observer
    graph=SimpleNamespace(_CAPTURED_TASKS=object())
    tail_hidden_observer.install(graph)
    class Model:
        def __call__(self,x,pos,meta):return torch.stack((x.float()+pos.float(),torch.ones_like(x).float()),dim=1)
        def compute_logits(self,h):return h[:,0:1]+torch.arange(7,dtype=torch.float32)[None,:]
    runner=Runner();runner.model=Model();runner._dynamic_tree_top2=True
    runner.last_draft_entry_key=('draft-greedy:7|steps:4|paged|confidence',2)
    x=torch.tensor([1,2]);pos=[torch.tensor([j,j+1])for j in range(4)]
    result=runner._execute_draft(x,pos,[object()for j in range(4)],7,return_confidence=True)
    assert tuple(result.shape)==(2,4,2)
    try:tail_hidden_observer.snapshot(runner,runner.last_draft_entry_key,4)
    except RuntimeError as error:
        assert str(error)=='No original captured per-step hidden buffers for this exact tail key'
        observed=str(error)
    else:raise AssertionError('failed423 observer wiring did not reproduce')
    value=dict(status='PASS_HOST425_OLD423_OBSERVER_FAILURE_REPRODUCED',
        actual_original_Fast_RootSeed_and_tail_observer_used=True,
        explicit_CPU_model_double=True,original_snapshot_failure=observed,
        reason='Original installer function globals remained in copied private base module, while observer replaced canonical execute_top2.',
        Native_models_constructed=0,NPU_used=False,NN_math_qualified=False)
    p=Path(__file__).resolve().parent
    (p/'HOST425_OLD423_REGRESSION_RESULT.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value))


if __name__=='__main__':main()
