"""Run actual pre-constructor setup with NPU initialization and models blocked."""
import json
from pathlib import Path
import tempfile


def construct():
    import torch
    import torch_npu
    from vllm_ascend.spec_decode.pearl import native_engine as native
    from vllm_ascend.spec_decode.pearl import native_graph as graph
    from vllm_ascend.spec_decode.pearl import api
    called = []

    def blocked(*args, **kwargs):
        called.append('blocked model or NPU initialization')
        raise AssertionError('HOST419 must not construct models or initialize an NPU')

    native.NativePearlEngine.__init__ = blocked
    api.PEARLEngine.__init__ = blocked
    torch_npu.npu._lazy_init = blocked
    from check419 import ready
    frozen = ready()
    import runtime419
    callbacks = runtime419.build_runtime()
    assert callbacks['worker'].__globals__ is callbacks['main'].__globals__
    assert callbacks['main'].__globals__['worker'] is runtime419.worker
    import extension419
    original = Path('/root/SpecSLO/budget-calibration-20261015/tree-bounded-parent4-shared4-eager-diagnostic-v199')
    output = Path(tempfile.mkdtemp(prefix='HOST419-construction-', dir=original))
    entry, controls = extension419.prepare_install(output)
    policy = entry['install'](fast=True, host=True, transport=True, replay=True, repair=True, prefill=True)
    import live_decode_patch
    overlay = extension419.after_install(native, graph, live_decode_patch)
    origins = extension419.loaded_origins()
    assert origins['fast_tree_draft_v5']['path'].endswith(
        'tree-parent4-normal24-SLO-mixed-protocol-v419/fast419.generated.py')
    from config199 import make_config
    cfg = make_config()
    assert cfg.max_num_seqs == 64
    import actor419
    import component419
    import binding419
    import target391
    import audit419
    # Explicit CPU-only lifecycle replay to test the actual SLO planner.
    # Synthetic verdicts below are HOST fixtures and never NN qualification.
    from types import SimpleNamespace
    from scope419 import Case
    import protocol419
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    c=Case(4,0)
    plan=lambda prefix:cached_cpu_tree_speculation_plan(2,4,prefix,4096,candidate_budget=4)
    parent_sources={i:(plan(128),[2,3,4,5]) for i in range(24,48)}
    fixture=dict(ids=list(range(24))+list(c.eager_ids),sources={i:parent_sources[i] for i in c.eager_ids},
        contexts=[dict(token_ids=[1]*129) for _ in range(64)],
        plans=[plan(128) for _ in range(24)]+[plan(133) for _ in c.eager_ids],
        parent_sources=parent_sources,parent_confidences={i:.5 for i in range(24,48)})
    mixed=dict(draft_token_ids=[[2,3,4,5,6,7,8,9] for _ in fixture['ids']],
        node_confidences=[torch.full((8,),.5) for _ in fixture['ids']],
        eager_frontier_tokens={i:6 for i in c.eager_ids})
    state=protocol419.setup(native,c,dict(fixture=fixture,mixed=mixed))
    ctl=state['controller']
    for i in range(24,48):
        positive=i in (24,25)
        ctl.finish_verification(i,fully_accepted=positive,proposed_tokens=4,
            accepted_tokens=4 if positive else 0,delivered_tokens=5 if positive else 1,
            draft_confidence=.5,ema_alpha=.1)
    import copy
    old=copy.deepcopy(ctl)
    kwargs=dict(active_indices_registered_unique=True,merge_ready_homes=True,
        max_target_requests=24,verification_budget=96,
        ready_candidate_counts={i:ticket.gamma for i,ticket in ctl.ready.items()})
    unchanged_order=old.build_plan(list(range(48)),**kwargs)
    selected=ctl.build_plan(list(range(48)),priority=True,target_slo_first=True,
        projected_wait_ms=20.,**kwargs)
    selected_eager=[i for i in selected.target_request_indices if ctl.ready[i].eager]
    assert len(selected.target_request_indices)==24 and selected_eager==[24,25]
    assert not any(old.ready[i].eager for i in unchanged_order.target_request_indices)
    assert sum(selected.target_candidate_budgets.values())==96
    assert state['diagnostic_slo_inputs'][0]['slo_tpot_ms']==50.
    assert state['diagnostic_slo_inputs'][24]['slo_tpot_ms']==40.
    assert not called
    result = dict(status='PASS_HOST419_REAL_PRECONSTRUCTOR_CANONICAL_ORIGINS_AND_BINDINGS',
                  NPU_used=False, model_constructors_called=0, NPU_lazy_init_called=0,
                  constructors_blocked_in_HOST=True,actual_SLO_controller_CPU_fixture_mixes24=True,synthetic_HOST_verdicts_are_not_NN_evidence=True, original_global_max_num_seqs=64,
                  frozen_source_count=len(frozen['files']), prepared_controls_class=str(controls),
                  actual_install_policy_type=type(policy).__qualname__,
                  actual_install_policy_repr=repr(policy), actual_observer_overlay=overlay,
                  origins=origins, output=str(output),
                  model_qualification=False, production_permission=False,
                  full463=False, Raw=None, Goodput=None)
    target = Path(__file__).with_name('ROOT419_REAL_STARTUP_CONSTRUCTION_HOST_RESULT.json')
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'NPU_used', 'model_constructors_called',
                                          'NPU_lazy_init_called', 'original_global_max_num_seqs')}))


def main():
    from check419 import require_ready
    from base199 import PARENT_ACTION
    from diagnostic_authorization import activate_local_diagnostic
    readiness = require_ready()
    spec = dict(action=PARENT_ACTION, identity=readiness['parent']['identity'])
    with activate_local_diagnostic(spec):
        construct()


if __name__ == '__main__':
    main()
