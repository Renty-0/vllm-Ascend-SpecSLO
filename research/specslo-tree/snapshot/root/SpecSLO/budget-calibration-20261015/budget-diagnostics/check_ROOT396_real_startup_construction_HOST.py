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
        raise AssertionError('HOST396 must not construct models or initialize an NPU')

    native.NativePearlEngine.__init__ = blocked
    api.PEARLEngine.__init__ = blocked
    torch_npu.npu._lazy_init = blocked
    from check396 import ready
    frozen = ready()
    import runtime396
    callbacks = runtime396.build_runtime()
    assert callbacks['worker'].__globals__ is callbacks['main'].__globals__
    assert callbacks['main'].__globals__['worker'] is runtime396.worker
    import extension396
    original = Path('/root/SpecSLO/budget-calibration-20261015/tree-bounded-parent4-shared4-eager-diagnostic-v199')
    output = Path(tempfile.mkdtemp(prefix='HOST396-construction-', dir=original))
    entry, controls = extension396.prepare_install(output)
    policy = entry['install'](fast=True, host=True, transport=True, replay=True, repair=True, prefill=True)
    import live_decode_patch
    overlay = extension396.after_install(native, graph, live_decode_patch)
    origins = extension396.loaded_origins()
    assert origins['fast_tree_draft_v5']['path'].endswith(
        'tree-parent4-normal24-repair-load-protocol-v396/fast396.generated.py')
    from config199 import make_config
    cfg = make_config()
    assert cfg.max_num_seqs == 64
    import actor396
    import component396
    import binding396
    import target391
    import audit396
    assert not called
    result = dict(status='PASS_HOST396_REAL_PRECONSTRUCTOR_CANONICAL_ORIGINS_AND_BINDINGS',
                  NPU_used=False, model_constructors_called=0, NPU_lazy_init_called=0,
                  constructors_blocked_in_HOST=True, original_global_max_num_seqs=64,
                  frozen_source_count=len(frozen['files']), prepared_controls_class=str(controls),
                  actual_install_policy_type=type(policy).__qualname__,
                  actual_install_policy_repr=repr(policy), actual_observer_overlay=overlay,
                  origins=origins, output=str(output),
                  model_qualification=False, production_permission=False,
                  full463=False, Raw=None, Goodput=None)
    target = Path(__file__).with_name('ROOT396_REAL_STARTUP_CONSTRUCTION_HOST_RESULT.json')
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'NPU_used', 'model_constructors_called',
                                          'NPU_lazy_init_called', 'original_global_max_num_seqs')}))


def main():
    from check396 import require_ready
    from base199 import PARENT_ACTION
    from diagnostic_authorization import activate_local_diagnostic
    readiness = require_ready()
    spec = dict(action=PARENT_ACTION, identity=readiness['parent']['identity'])
    with activate_local_diagnostic(spec):
        construct()


if __name__ == '__main__':
    main()
