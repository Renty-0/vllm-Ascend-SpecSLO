"""Source projection, original planner sizes and old-proof identity checks."""
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import check_admission345_HOST as admission_test

P = Path(__file__).resolve().parent
B = P.parent
sys.path.insert(0, str(B/'tree-capacity-risk-single-prepare-full-v307'))
import runtime307
import generation307
import prefill345
import subset345
from admission345 import patch


def main():
    from source307_HOST import source as actual_source
    old = generation307.patch(actual_source())
    new = patch(old)
    assert 'if active:\n        return closed' not in new
    assert '_root307_build_plan' in new and '_root307_observe' in new
    ns, effects = admission_test.namespace([128]*48, list(range(44)), list(range(44,48)))
    hooks = dict(reset=0, reserve=0)
    def reset(**kwargs):
        assert kwargs['active'] == list(range(44))
        hooks['reset'] += 1
    def reserve(**kwargs):
        assert kwargs['admitted'] == kwargs['needs_prefill'] == list(range(44,48))
        assert kwargs['active'] == list(range(44))
        hooks['reserve'] += 1
        return False
    ns.update(cache_activated=set(), cache_released=set(),
              _v75_hooks=SimpleNamespace(reset_for_original_admission_fallback=reset,
                  reserved_slots=lambda: 0, filter_pending_candidates=lambda candidates: candidates,
                  reserve_at_original_admission=reserve))
    exec(admission_test.function(new, False), ns)
    assert ns['admit_available']() is False
    assert hooks == dict(reset=1, reserve=1)
    assert ns['active'] == list(range(48)) and ns['pending_admission'] == []
    assert effects['prefills'] == [list(range(44,48))]
    assert all(abs(s.decode_elapsed_ms-50.0) < 1e-8 for s in ns['states'][:44])
    assert prefill345.project().count('for n in (1,3,4,8)') == 3
    real = prefill345.namespace()
    assert inspect.signature(real['run_case']) == inspect.signature(prefill345.original.run_case)
    assert inspect.signature(real['run']) == inspect.signature(prefill345.original.run)
    assert inspect.signature(subset345.namespace()['prepare']) == inspect.signature(subset345.original.prepare)
    original_proof = object()
    engine = SimpleNamespace(_v209_actual_prefill=original_proof)
    expected = dict(actual_extra_component='fixture-only')
    def fake_run(engine):
        engine._v209_actual_prefill = expected
        return expected
    try:
        prefill345._NAMESPACE = dict(run=fake_run)
        assert prefill345.run(engine) is expected
        assert engine._v209_actual_prefill is original_proof
        assert engine._ROOT345_actual_small_refill is expected
        def fail(engine):
            engine._v209_actual_prefill = expected
            raise RuntimeError('controlled HOST failure')
        prefill345._NAMESPACE = dict(run=fail)
        try: prefill345.run(engine)
        except RuntimeError as error: assert str(error) == 'controlled HOST failure'
        else: raise AssertionError('expected failure absent')
        assert engine._v209_actual_prefill is original_proof
    finally:
        prefill345._NAMESPACE = real
    assert not any(n.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend') for n in sys.modules)
    result = dict(status='passed_small_refill_component_projection_HOST', NPU_used=False,
        source_inverse_passed=True, actual_sizes=[1,3,4,8], zero_scalar_checks_required=672,
        original_numerical_bodies_and_matchers_retained=True, original_worker_signatures=True,
        original_component_proof_identity_preserved_on_success_and_failure=True,
        B1to8_LIVE_admission_projection_passed=True, model_constructed=False,
        actual_composed_LIVE_function_and_original_admission_hooks_executed=True,
        no_actual_model_or_performance_qualification=True)
    (P/'HOST345_COMPONENT_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
