"""Frozen actual install/origin check, stopped before any Native constructor."""
import json
from pathlib import Path
import tempfile


def main():
    import torch
    import torch_npu
    attempted = []
    def blocked(*args, **kwargs):
        attempted.append(True)
        raise AssertionError('ROOT425 preconstructor check cannot initialize models/NPU')
    torch.npu._lazy_init = blocked
    torch_npu.npu._lazy_init = blocked
    from check425 import ready
    frozen = ready()
    from check209 import require_ready
    from base209 import PARENT_ACTION
    from diagnostic_authorization import activate_local_diagnostic
    original = require_ready()
    spec = dict(action=PARENT_ACTION, identity=original['parent']['identity'])
    B = Path(__file__).resolve().parent.parent
    with activate_local_diagnostic(spec):
        import extension425
        output = Path(tempfile.mkdtemp(prefix='HOST425-preconstructor-',
            dir=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'))
        entry, controls = extension425.prepare_install(output)
        from vllm_ascend.spec_decode.pearl import native_engine as native, native_graph as graph
        native.NativePearlEngine.__init__ = blocked
        policy = extension425.install_entry(entry, native)
        import live_decode_patch
        overlay = extension425.after_install(native, graph, live_decode_patch)
        origins = extension425.loaded_origins()
        from factory425 import current
        callbacks = current(native)
        from bounded425 import actor
        import actor419
        actual_actor = actor()
        assert actual_actor.__globals__['run_case'] is actor419.run_case
        from base_component425 import construct209, parent_component
        baseline = construct209()
        parent = parent_component()
        assert baseline.__code__.co_filename.endswith('component209.py')
        assert parent.run.__code__.co_filename.endswith('component173.py')
        assert callable(controls) and policy.__class__.__name__ == 'TreeBudgetPolicy'
        assert overlay['frontier_observer392_before_constructor'] is True
        assert callbacks['e4'].__code__.co_filename.endswith('retained419-E4-body-new-explicit-gate')
        assert not attempted
        result = dict(status='PASS_HOST425_FROZEN_REAL_INSTALL_ORIGINS_BEFORE_NATIVE_CONSTRUCTOR',
            frozen_members=len(frozen['files']), actual_helper_origins=origins,
            actual_original_installers_and_early_observer=True,
            single_original_runner_cache_hooks=True,
            original173_209_math_and419_case_bodies_retained=True,
            explicit_current_dispatcher_without_E4_permission=True,
            Native_constructor_calls=0, NPU_init_calls=0, NPU_used=False,
            Raw=None, Goodput=None)
        destination = B/'budget-diagnostics/ROOT425_REAL_PRECONSTRUCTOR_RESULT.json'
        with destination.open('x') as stream:
            json.dump(result, stream, indent=2)
            stream.write('\n')
        print(json.dumps({k: result[k] for k in ('status','frozen_members','NPU_used')}))


if __name__ == '__main__':
    main()
