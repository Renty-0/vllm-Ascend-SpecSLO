"""Actual installers and original419 commit entry, without a Native model."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace


def main():
    import torch
    import torch_npu
    attempted = []
    def forbidden(*a, **k):
        attempted.append(True)
        raise AssertionError('HOST430 cannot construct models or initialize an NPU')
    torch.npu._lazy_init = forbidden
    torch_npu.npu._lazy_init = forbidden
    from check209 import require_ready
    from base209 import PARENT_ACTION
    from diagnostic_authorization import activate_local_diagnostic
    parent = require_ready()
    spec = dict(action=PARENT_ACTION, identity=parent['parent']['identity'])
    with activate_local_diagnostic(spec):
        import extension428
        B = Path(__file__).resolve().parent.parent
        output = Path(tempfile.mkdtemp(prefix='HOST430-prestartup-', dir=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'))
        entry, _ = extension428.prepare_install(output)
        from vllm_ascend.spec_decode.pearl import native_engine as native, native_graph as graph
        native.NativePearlEngine.__init__ = forbidden
        from original_tree_cycle_v75 import capture_original_protocol_before_startup
        protocol = capture_original_protocol_before_startup(native)
        import lifecycle430
        captured = lifecycle430.capture(protocol, native)
        extension428.install_entry(entry, native)
        import tail_hidden_observer, live_decode_patch
        tail_hidden_observer.install(graph)
        extension428.after_install(native, graph, live_decode_patch)
        from factory425 import current
        current(native)
        # Allocate only a Python object. No constructor, model, graph, cache or
        # device is used. The original419 entry rejects the actual old omission.
        engine = object.__new__(native.NativePearlEngine)
        engine._v75_original_protocol = protocol
        import protocol419
        source = dict(controller=object(), payloads={}, logical={})
        class StopBeforeFirstNativeStage(Exception):
            pass
        def stop(label, callback):
            assert label == 'pre-verdict-ticket-identity'
            raise StopBeforeFirstNativeStage
        rejected = []
        try:
            protocol419.commit(engine, source, [], [[], [], [], [], 0], None, None, protocol, stop)
        except AttributeError as error:
            assert '_v199_commit_states' in str(error)
            rejected.append('old428_missing_original_binding')
        else:
            raise AssertionError('old binding omission did not reproduce')
        evidence = lifecycle430.bind(engine, captured)
        try:
            protocol419.commit(engine, source, [], [[], [], [], [], 0], None, None, protocol, stop)
        except StopBeforeFirstNativeStage:
            pass
        else:
            raise AssertionError('original419 must reach its unmodified first Native stage')
        original_function = engine._v199_commit_states
        engine._v199_commit_states = lambda *args: None
        try:
            lifecycle430.snapshot(engine)
        except RuntimeError:
            rejected.append('different_commit_function_identity')
        else:
            raise AssertionError('wrong commit function was accepted')
        engine._v199_commit_states = original_function
        assert lifecycle430.snapshot(engine) == evidence
        assert not attempted
        result = dict(status='PASS_HOST430_REAL_PRESTARTUP_CAPTURE_INSTALL_AND_ORIGINAL419_COMMIT_ENTRY',
            binding=evidence, controlled_negatives=rejected,
            original428_factory_and_tail_observer_installed=True,
            original419_commit_code_unchanged=True,
            positive_stopped_before_first_Native_stage=True,
            only_uninitialized_Python_engine_object_used=True,
            Native_models_constructed=0, NPU_used=False,
            NN_qualification_inherited=False)
        (Path(__file__).resolve().parent / 'HOST430_BINDING_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
