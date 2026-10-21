"""Real controller with CPU protocol metadata, no model backend imports."""
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

P = Path(__file__).resolve().parent
B = P.parent


def main():
    sys.path.insert(0, str(B / 'tree-frugal-S32-paid-budget-control-host-v401'))
    from check401_HOST import BACKENDS, original_control_sources, state_set
    rhythm = original_control_sources()
    sys.path[:0] = [str(B / 'tree-active64-whole-proposal-segment-limit-v398'),
                    str(B / 'tree-frugal-S32-paid-budget-control-host-v401')]
    import packing407
    import packing398
    import packing401
    original = rhythm.SpecRhythmPipelineController.build_plan
    rng = random.Random(0)

    def fixture(budgets):
        states = state_set(rhythm, len(budgets), rng)
        controller = rhythm.SpecRhythmPipelineController(states)
        payloads = {}
        for i, b in enumerate(budgets):
            ticket = controller.new_ticket(i, gamma=b, eager=False)
            controller.publish([ticket])
            parents = [-1, 0, 1, 2][:b] + ([-1] if b == 5 else [])
            vector = SimpleNamespace(device=SimpleNamespace(type='cpu'),
                                     tolist=lambda parents=parents: list(parents))
            payloads[ticket.proposal_id] = dict(plan=SimpleNamespace(
                candidate_budget=b, parent_indices=vector))
            controller._ready_wait_cycles[i] = rng.randrange(8)
        return controller, payloads

    narrow = wide = actual_wide = 0
    for trial in range(1200):
        n = rng.randrange(1, 65)
        budgets = [rng.randrange(1, 4 if trial % 2 else 6) for _ in range(n)]
        ctl, payloads = fixture(budgets)
        resources = packing407.collect(ctl, payloads, range(n))
        expected = copy.deepcopy(ctl)
        is_wide = n > 24 and max(budgets) <= 3
        kwargs = dict(active_indices_registered_unique=True, merge_ready_homes=True,
                      priority=bool(trial % 2), target_slo_first=bool(trial % 3),
                      projected_wait_ms=20., goodput_edf=bool(trial % 5),
                      max_target_requests=32, verification_budget=96)
        if is_wide:
            truth = packing401.build_plan(expected, range(n), **dict(kwargs,
                verification_budget=88, ready_candidate_counts=dict(enumerate(budgets))),
                max_large_groups401=24,
                ready_large_groups401={i: int(b == 3) for i, b in enumerate(budgets)})
            wide += 1
        else:
            truth = packing398.build_plan(expected, range(n), **dict(kwargs,
                max_target_requests=24, ready_candidate_counts=dict(enumerate(budgets))),
                max_target_segments=24, ready_segment_counts=resources.segments)
            narrow += 1
        tickets = dict(ctl.ready)
        result = packing407.build_plan(ctl, range(n), ready_resources=resources, **kwargs)
        assert result == truth and ctl.__dict__ == expected.__dict__
        selected = result.target_request_indices
        assert all(ctl.ready[i] is ticket for i, ticket in tickets.items())
        assert set(selected) | set(result.deferred_target_request_indices) == set(range(n))
        if len(selected) > 24:
            assert all(budgets[i] <= 3 for i in selected)
            assert sum(budgets[i] == 3 for i in selected) <= 24
            assert sum(budgets[i] for i in selected) <= 88
            actual_wide += 1
    # Independent quota:25 B3 plus7 B1 fit C88 but cannot fit24 q4 groups.
    ctl, payloads = fixture([3] * 25 + [1] * 7)
    resources = packing407.collect(ctl, payloads, range(32))
    kwargs = dict(active_indices_registered_unique=True, merge_ready_homes=True,
                  verification_budget=96, max_target_requests=32)
    got = packing407.build_plan(ctl, range(32), ready_resources=resources, **kwargs)
    assert len(got.target_request_indices) == 31
    try:
        packing407.build_plan(copy.deepcopy(ctl), range(32), ready_resources=resources,
            **kwargs, reserved_target_request_indices=tuple(range(32)), reserved_target_home_batch_id=0)
    except ValueError:
        pass
    else:
        raise AssertionError('unsafe wide replay reservation accepted')
    # A deep leftover retains old S24 routing until actually consumed; no trim.
    ctl, payloads = fixture([5] + [3] * 31)
    resources = packing407.collect(ctl, payloads, range(32))
    got = packing407.build_plan(ctl, range(32), ready_resources=resources, **kwargs)
    assert len(got.target_request_indices) <= 24
    assert sum(resources.segments[i] for i in got.target_request_indices) <= 24
    try:
        packing407.build_plan(ctl, range(65), ready_resources=resources, **kwargs)
    except ValueError:
        pass
    else:
        raise AssertionError('global65 accepted')
    assert narrow and wide and actual_wide
    assert rhythm.SpecRhythmPipelineController.build_plan is original
    assert not any(name.split('.')[0] in BACKENDS for name in sys.modules)
    result = dict(status='PASS_HOST407_DUAL_WHOLE_TICKET_ROUTES_AND_AGE_EQUIVALENCE',
                  exact_narrow_route_cases=narrow, exact_wide_route_cases=wide,
                  actual_selected_N_gt24_cases=actual_wide,
                  original_priority_and_wait_age_and_ticket_identity=True,
                  independent_large_group_quota_and_unsafe_reservation=True,
                  deep_leftover_narrow_fallback=True, global65_rejected=True,
                  resources=packing407.descriptor(), NPU_used=False,
                  model_backends_imported=False, LIVE_installed=False,
                  production_permission=False, Raw=None, Goodput=None)
    (P / 'HOST407_RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
