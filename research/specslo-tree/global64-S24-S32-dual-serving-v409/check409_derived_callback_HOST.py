"""Construct real derived controls and invoke its actual installed callback.

No Native model is constructed. A CPU context transition simulates only the
constructor's existing CPU context, so real control selection can be tested.
"""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace


def main():
    import torch
    import torch_npu
    calls = []
    def blocked(*args, **kwargs):
        calls.append('forbidden_NPU_or_model_constructor')
        raise AssertionError('HOST409 cannot construct models or initialize an NPU')
    torch.npu._lazy_init = blocked
    torch_npu.npu._lazy_init = blocked
    torch.set_num_threads(1)
    from check209 import require_ready
    from base209 import PARENT_ACTION
    from diagnostic_authorization import activate_local_diagnostic
    readiness = require_ready()
    spec = dict(action=PARENT_ACTION, identity=readiness['parent']['identity'])
    with activate_local_diagnostic(spec):
        from extension409 import prepare_install, install_entry
        donor = Path('/root/SpecSLO/budget-calibration-20261015/tree-causal80-small-cohort-role-aware-audit-full-v209')
        output = Path(tempfile.mkdtemp(prefix='HOST409-derived-', dir=donor))
        entry, Controls = prepare_install(output)
        from vllm_ascend.spec_decode.pearl import native_engine as native, spec_rhythm as rhythm
        policy = install_entry(entry, native)
        from vllm_ascend.spec_decode.pearl import native_graph as graph
        import tail_hidden_observer, live_decode_patch
        tail_hidden_observer.install(graph)
        from extension409 import after_install
        overlay = after_install(native, graph, live_decode_patch)
        native.NativePearlEngine.__init__ = blocked
        from cpu_default_device_cleanup_host import capture_before_native_constructor
        constructor_state = capture_before_native_constructor(torch)
        assert constructor_state.eligible
        torch.set_default_device('cpu')
        from startup_config_v132 import make_config
        engine = SimpleNamespace(config=make_config().to_native(), cache_allocation=None, cache_block_tables=None,
                                 _dynamic_tree_generation=1)
        combination = json.loads((Path('/root/SpecSLO/budget-calibration-20261015/tree-high-normal-cap4-native-families-v148') / 'CANDIDATE_V132.json').read_text())
        controls = Controls(native=native, rhythm=rhythm, policy=policy, torch=torch,
                            constructor_state=constructor_state, is_draft=True,
                            configuration=combination['controls'])
        controls.select('candidate', engine=engine)
        engine._root409_controls = controls
        shaper = rhythm.SpecRhythmBudgetShaper(min_gamma=1, max_gamma=8, verification_budget=96)
        assert shaper.shape.__func__ is controls.candidate_shape
        from callback409 import inspect_actual
        method, actual_policy, original_shape = inspect_actual(engine, shaper)
        assert actual_policy is controls.policy is policy
        from normal_admission_v140 import reset
        reset(policy, 1)
        states = {i: rhythm.SpecRhythmRuntimeState(
            i, i % 2, slo_class='tight' if i % 5 < 3 else 'normal' if i % 5 == 3 else 'loose',
            slo_tpot_ms=40. if i % 5 < 3 else 50. if i % 5 == 3 else 150.,
            max_gamma=5, delivered_tokens=1, decode_elapsed_ms=10.,
            decode_start_elapsed_ms=0., output_token_budget=256) for i in range(64)}
        before = copy.deepcopy(states)
        kwargs = dict(plan_id=0, normal_request_indices=tuple(range(64)), eager_request_indices=(),
                      states=states, projected_wait_ms=20., context_len=2048, batch_size=64,
                      draft_token_budget=192, eager_token_cap=3, eager_reserve_tokens=32,
                      verification_roof=96)
        import shape409
        result = shape409.shape(engine, shaper, **kwargs)
        assert len(result.requested_normal) == 32 and states == before
        assert engine._typed404_binding.policy is policy
        assert controls.observe()['mode'] == 'candidate'
        assert len(result.normal_budgets) <= 32 and sum(result.normal_budgets.values()) <= 88
        from audit_normal409 import verify_snapshot
        verify_snapshot(shape409.snapshot(engine), 1)
        # Reproduce the exact402 failure with this real constructed callback.
        assert method.__code__.co_filename.endswith('source173.py::candidate_controls_v132.py')
        old_rejects_derived = method.__code__.co_filename != '/root/SpecSLO/budget-calibration-20261015/tree-high-normal-cap4-native-families-v148/candidate_controls_v132.py'
        assert old_rejects_derived
        # The pending S32 policy has the same provenance assumption in frozen
        #404. Exercise its exact body with the repaired engine-owned binding.
        from wide409 import shape as wide_shape
        reset(policy, 2)
        engine._dynamic_tree_generation = 2
        wide = wide_shape(engine, shaper, **dict(kwargs,
            normal_request_indices=tuple(range(56)), eager_request_indices=tuple(range(56, 64))))
        assert len(wide.requested_normal) == 32
        assert sum(wide.normal_budgets.values()) <= 88
        assert sum(b == 3 for b in wide.normal_budgets.values()) <= 24
        assert all(1 <= b <= 3 for b in wide.normal_budgets.values())
        assert all(b == 3 for b in wide.eager_budgets.values())
        assert engine._typed404_binding.policy is policy and states == before
        assert controls.observe()['mode'] == 'candidate'
        # Short normal groups under global active64 also stay B<=3, keeping
        #the conservative whole-ticket router eligible for the wide backend.
        reset(policy, 3)
        engine._dynamic_tree_generation = 3
        short = shape409.shape(engine, shaper, **dict(kwargs,
            normal_request_indices=tuple(range(8)), eager_request_indices=()))
        assert len(short.normal_budgets) == 8
        assert all(b <= 3 for b in short.normal_budgets.values())
        assert shape409.snapshot(engine)['policy_projection']['high_active_threshold'] == 48
        original_code = method.__code__
        method.__code__ = original_code.replace(co_filename='/invalid/forged_callback.py')
        try:
            try:
                inspect_actual(engine, shaper)
            except RuntimeError:
                pass
            else:
                raise AssertionError('forged callback provenance accepted')
        finally:
            method.__code__ = original_code
        controls.select('original', engine=engine)
        torch.set_default_device(None)
        assert not calls
        result = dict(status='PASS_HOST409_REAL_DERIVED_CONTROLS_SELECT_AND_TYPED_SHAPE',
                      actual_controls_constructor_and_select_called=True,
                      actual_full_after_install_overlay_executed=True,
                      high_load_short_normal_groups_also_cap3=True,
                      actual_shape_callback_filename=method.__code__.co_filename,
                      old402_filename_gate_failure_reproduced=True,
                      repaired_exact_code_filename_and_original_closure_checked=True,
                      forged_callback_filename_rejected=True,
                      actual_policy_object_preserved=True,
                      real_normal64_to32_plan_and_defer_age_audited=True,
                      real_derived_wide32_C88_and_B3_quota_checked=True,
                      wide_policy_not_installed_into_serving=True,
                      controls_original_restored=True,
                      HOST_CPU_context_transition_explicit_not_a_Native_constructor=True,
                      Native_constructor_calls=0, NPU_init_calls=0, NPU_used=False,
                      model_qualification=False, production_permission=False,
                      Raw=None, Goodput=None)
        p = Path(__file__).resolve().parent
        (p / 'HOST409_DERIVED_CALLBACK_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result))


if __name__ == '__main__':
    main()
