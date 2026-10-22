"""Frozen exact full installer, capture order, worker and source origins."""
import json
from pathlib import Path
import tempfile


def main():
    import torch, torch_npu
    attempts = []
    def forbidden(*a, **k):
        attempts.append(True)
        raise AssertionError('HOST434 frozen no model or NPU init')
    torch.npu._lazy_init = forbidden
    torch_npu.npu._lazy_init = forbidden
    from check434 import ready
    ready()
    from check209 import require_ready
    from base209 import PARENT_ACTION
    from diagnostic_authorization import activate_local_diagnostic
    parent = require_ready()
    spec = dict(action=PARENT_ACTION, identity=parent['parent']['identity'])
    B = Path(__file__).resolve().parent.parent
    with activate_local_diagnostic(spec):
        import extension434
        output = Path(tempfile.mkdtemp(prefix='HOST434-frozen-', dir=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'))
        entry, _ = extension434.prepare_install(output)
        from vllm_ascend.spec_decode.pearl import native_engine as native, native_graph as graph
        native.NativePearlEngine.__init__ = forbidden
        from original_tree_cycle_v75 import capture_original_protocol_before_startup
        from lifecycle430 import capture, bind
        protocol = capture_original_protocol_before_startup(native)
        captured = capture(protocol, native)
        extension434.install_entry(entry, native)
        import tail_hidden_observer, live_decode_patch
        tail_hidden_observer.install(graph)
        extension434.after_install(native, graph, live_decode_patch)
        engine = object.__new__(native.NativePearlEngine)
        engine._v75_original_protocol = protocol
        binding = bind(engine, captured)
        origins = extension434.loaded_origins()
        from factory425 import current
        installed = current(native)
        import fast_tree_draft_v5 as canonical
        assert installed['fallback'].__globals__ is vars(canonical)
        from bounded428 import actor
        import target428
        assert actor().__globals__['run_case'].__globals__['target'] is target428
        import runtime434
        import slot_preflight218
        assert slot_preflight218.snapshot()['shadow'] is True
        import extension432
        qualifier=extension432.qualifier()
        names=qualifier.__code__.co_names
        assert names.index('bounded428')<names.index('incumbents427')<names.index('end_shadow')<names.index('issue')
        runtime = runtime434.build_runtime()
        assert runtime['worker'].__globals__['worker'] is runtime434.worker
        assert not attempts
        value = dict(status='PASS_HOST434_FROZEN_ACTUAL_INSTALL_WORKER_AND_ORIGINAL_COMMIT_CAPTURE',
            actual_origins=origins, original_commit_binding=binding,
            original425_factory_and428_reference_retained=True,
            actual419_case_protocol_and_target_unchanged=True,
            Native_constructor_calls=0, NPU_init_calls=0, NPU_used=False,
            NN_qualification_inherited=False)
        path = B/'budget-diagnostics/ROOT434_REAL_PRECONSTRUCTOR_RESULT.json'
        with path.open('x') as stream:
            json.dump(value, stream, indent=2); stream.write('\n')
        print(json.dumps(dict(status=value['status'], NPU_used=False)))


if __name__ == '__main__':
    main()
