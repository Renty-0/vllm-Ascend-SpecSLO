"""Exercise actual composed admission, original publication and protected polls."""
import copy
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path[:0] = [str(HERE), str(BASE / 'tree-coalesced-live-refill-host-v345'),
               str(BASE / 'tree-coalesced-live-refill-M120-singleton-capture-full-v348')]
import runtime348  # Only composes source lookup paths; no framework import.
from admission354 import MODES, allowed_classes, descriptor, patch
from admission345 import patch as previous_patch
from check_admission345_HOST import function, namespace
from generation348 import _original_tree_patch
from source348_HOST import source


def world(lengths, active, pending, classes, **kwargs):
    ns, effects = namespace(lengths, active, pending, **kwargs)
    for state, cls in zip(ns['states'], classes, strict=True):
        state.slo_class = cls
    ns.update(cache_activated=set(), cache_released=set(), _v75_hooks=SimpleNamespace(
        reset_for_original_admission_fallback=lambda **kwargs: None,
        reserved_slots=lambda: 0,
        filter_pending_candidates=lambda values: values,
        reserve_at_original_admission=lambda **kwargs: False))
    return ns, effects


def compare_worlds(actual, expected):
    ns, effects = actual
    other, other_effects = expected
    assert effects == other_effects
    assert ns['active'] == other['active']
    assert ns['pending_admission'] == other['pending_admission']
    assert ns['counters'] == other['counters']
    for key in ('states', 'local_states', 'draft_states', 'target_states'):
        for state, reference in zip(ns[key], other[key], strict=True):
            left = {k: v for k, v in vars(state).items() if not callable(v)}
            right = {k: v for k, v in vars(reference).items() if not callable(v)}
            assert left == right


def execute(code, args, classes, **kwargs):
    ns, effects = world(*args, classes, **kwargs)
    exec(code, ns)
    result = ns['admit_available']()
    return result, (ns, effects)


def main():
    output = HERE / 'HOST354_RESULT.json'
    assert not output.exists()
    base = _original_tree_patch(source())
    reference = function(previous_patch(base), False)
    codes = {mode: function(patch(base, mode), False) for mode in MODES}
    explicit = []
    for mode in MODES:
        for cls in ('tight', 'normal', 'loose', 'unknown'):
            args = ([128] * 48, list(range(44)), list(range(44, 48)))
            classes = [cls] * 44 + ['tight'] * 4
            result, actual = execute(codes[mode], args, classes)
            ns, effects = actual
            blocked = cls not in allowed_classes(mode)
            assert result is False
            if blocked:
                assert ns['active'] == args[1] and ns['pending_admission'] == args[2]
                assert not any(effects.values())
                assert ns['counters']['spec_rhythm_ROOT354_priority_guard_polls'] == 1
                assert all(state.decode_elapsed_ms == 0 for state in ns['states'])
            else:
                _, expected = execute(reference, args, classes)
                compare_worlds(actual, expected)
                assert effects['deliveries'] == list(range(44, 48))
                assert effects['prefills'] == [[44, 45, 46, 47]]
                assert all(abs(s.decode_elapsed_ms - 50.) < 1e-8 for s in ns['states'][:44])
            explicit.append(dict(mode=mode, incumbent_class=cls, blocked=blocked,
                                 published_roots=effects['deliveries']))
        # Empty-active initial admission is identical for every mode.
        args = ([128] * 48, [], list(range(48)))
        classes = ['tight', 'normal', 'loose'] * 16
        _, actual = execute(codes[mode], args, classes)
        _, expected = execute(reference, args, classes)
        compare_worlds(actual, expected)
        assert actual[1]['deliveries'] == list(range(48))

    rng = random.Random(354)
    blocked_cases = allowed_cases = 0
    for case in range(240):
        lengths = [rng.choice((59, 128, 609, 2301)) for _ in range(60)]
        count = rng.randrange(21)
        active = list(range(count))
        pending = list(range(count, 60))
        rng.shuffle(pending)
        classes = [rng.choice(('tight', 'normal', 'loose')) for _ in lengths]
        if case % 2 == 0:
            classes[:count] = ['loose'] * count
        args = (lengths, active, pending)
        options = dict(arrivals=[rng.choice((0, 99, 101)) for _ in lengths])
        for mode in MODES:
            _, actual = execute(codes[mode], args, classes, **options)
            blocked = bool(active) and any(classes[i] not in allowed_classes(mode) for i in active)
            if blocked:
                ns, effects = actual
                assert ns['active'] == active and ns['pending_admission'] == pending
                assert not any(effects.values())
                blocked_cases += 1
            else:
                _, expected = execute(reference, args, classes, **options)
                compare_worlds(actual, expected)
                allowed_cases += 1
    assert allowed_cases and blocked_cases
    for invalid in ('all', '', True):
        try:
            patch(base, invalid)
        except ValueError:
            pass
        else:
            raise AssertionError('undeclared mode accepted')
    assert not any(name.split('.')[0] in ('torch', 'torch_npu', 'vllm', 'vllm_ascend')
                   for name in sys.modules)
    result = dict(status='PASS_HOST_actual_composed_priority_guard_and_original_admission',
                  modes={mode: descriptor(base, mode) for mode in MODES},
                  explicit_cases=explicit, random_mode_cases=720,
                  random_blocked=blocked_cases, random_allowed=allowed_cases,
                  allowed_original_outputs_effects_and_accounting_identical=True,
                  protected_pending_and_incumbent_state_unchanged=True,
                  no_root_computed_and_held=True,
                  original50ms_refill_charge_for_allowed_incumbents=True,
                  unknown_incumbent_class_blocks_refill=True,
                  NPU_used=False, serving_installed=False, full463_qualified=False,
                  Goodput=None, goal_achieved=False)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
