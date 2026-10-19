"""Check real additive delegates, whole-source inverses and launch pins on HOST."""
import ast
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


P = Path(__file__).resolve().parent
B = P.parent
sys.path.insert(0, str(P))
import runtime356
import runtime348
import generation348
import generation356
import trace348
import trace356
import extension348
import extension356
import audit356
import guard356
from source348_HOST import source
from check_admission345_HOST import function, namespace


def main():
    output = P / 'HOST356_RESULT.json'
    assert not output.exists()
    frozen_parent_generation = generation348.patch
    original_source = (B / 'tree-causal80-small-cohort-role-aware-audit-full-v209/runtime209.py').read_text()
    original_projection, _ = runtime348.project_source(original_source)
    projected, worker_changes = runtime356.project_source(original_projection)
    ast.parse(projected)
    audit356.projected_source()  # Executes the complete reversible audit projection.
    original_namespace = trace348.namespace()
    original_patch = original_namespace['canonical_patch']
    private = trace356.namespace()
    assert private is not original_namespace
    assert private['canonical_patch'] is generation356.patch
    for name in ('make_trace', 'warm_to_fixedpoint', 'run_windows'):
        assert private[name].__code__ is original_namespace[name].__code__
        assert private[name].__globals__ is private
    assert original_namespace['canonical_patch'] is original_patch
    assert generation348.patch is frozen_parent_generation
    for name in ('prepare_install', 'install_entry', 'after_install', 'qualify_component',
                 'snapshot', 'cache_snapshot', 'postfull_AR'):
        assert getattr(extension356, name) is getattr(extension348, name)
    base = source()
    projected_live = generation356.patch(base)
    descriptor = generation356.descriptor(base)
    digest = hashlib.sha256(projected_live.encode()).hexdigest()
    assert descriptor['projected_source_sha256'] == descriptor['ROOT348_LIVE_source_sha256'] == digest
    assert descriptor['ROOT356_admission_mode'] == 'protect_tight_normal'
    code = function(projected_live, False)
    fixtures = []
    for cls in ('tight', 'normal'):
        ns, effects = namespace([128] * 48, list(range(44)), list(range(44, 48)))
        for state in ns['states']:
            state.slo_class = cls
        ns.update(cache_activated=set(), cache_released=set(), _v75_hooks=SimpleNamespace(
            reset_for_original_admission_fallback=lambda **kw: None,
            reserved_slots=lambda: 0, filter_pending_candidates=lambda values: values,
            reserve_at_original_admission=lambda **kw: False))
        exec(code, ns)
        assert ns['admit_available']() is False
        assert not any(effects.values())
        assert ns['pending_admission'] == list(range(44, 48))
        assert ns['active'] == list(range(44))
        fixtures.append(dict(incumbent_class=cls, refill_blocked=True,
                             no_prefill_clock_or_root_publication=True))
    # Execute the entire allowed loose-incumbent path, including original
    # root publication, home assignment and50ms incumbent time charge.
    ns, effects = namespace([128] * 48, list(range(44)), list(range(44, 48)))
    for state in ns['states']:
        state.slo_class = 'loose'
    ns.update(cache_activated=set(), cache_released=set(), _v75_hooks=SimpleNamespace(
        reset_for_original_admission_fallback=lambda **kw: None,
        reserved_slots=lambda: 0, filter_pending_candidates=lambda values: values,
        reserve_at_original_admission=lambda **kw: False))
    exec(code, ns)
    assert ns['admit_available']() is False
    assert ns['active'] == list(range(48)) and ns['pending_admission'] == []
    assert effects['prefills'] == [[44,45,46,47]]
    assert effects['deliveries'] == [44,45,46,47]
    assert effects['clocks'] == ['arrival_clock','prefill_clock']
    assert all(abs(state.decode_elapsed_ms-50.) < 1e-8 for state in ns['states'][:44])
    fixtures.append(dict(incumbent_class='loose', refill_blocked=False,
                         original_root_publication_and50ms_charge=True))
    parent = B / 'tree-coalesced-live-refill-M120-singleton-capture-full-v348'
    original_env = [line for line in (parent / 'startup348.sh').read_text().splitlines()
                    if line.startswith(('export VLLM_', 'unset VLLM_'))]
    actual_env = [line for line in (P / 'startup356.sh').read_text().splitlines()
                  if line.startswith(('export VLLM_', 'unset VLLM_'))]
    assert actual_env == original_env
    saved = guard356.EXPECTED_FREEZE
    guard356.EXPECTED_FREEZE = 'a' * 64
    try:
        argv = guard356.launch_argv('HOST-no-execution')
    finally:
        guard356.EXPECTED_FREEZE = saved
    expected_parent = 'SPECSLO_V348_ROOT_FREEZE_SHA=09119c918ad6d28d137a731cf041e69f7a0135dc302169455fa944df6caab5c8'
    assert argv.count(expected_parent) == 1
    assert argv.count('SPECSLO_V356_ROOT_FREEZE_SHA=' + 'a' * 64) == 1
    original218 = 'fa58f2866e8fed383e2d1ed1103b2fcb72a731aab3d65060e3a7c3e350a4d260'
    assert 'SPECSLO_V218_ROOT_FREEZE_SHA=' + original218 in (P / 'launch356.sh').read_text()
    assert not any(name.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend')
                   for name in sys.modules)
    result = dict(status='PASS_HOST_additive_protected_refill_delegates_and_source_inverses',
                  actual_worker_projection_changes=len(worker_changes),
                  whole_worker_and_audit_source_inverse=True,
                  frozen348_modules_and_function_globals_not_modified=True,
                  actual_trace_code_objects_identical_with_private_globals=True,
                  fresh_numerical_qualification_delegates_identical=True,
                  model_kernel_and_VLLM_environment_lines_identical=True,
                  separate356_and_original348_source_pins=True,
                  actual_original218_launch_pin_preserved=True,
                  actual_live_source_sha256=digest, fixtures=fixtures,
                  NPU_used=False, model_constructed=False, frozen=False,
                  whole463_AR_qualified=False, Goodput=None, goal_achieved=False)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
