"""Exercise real controller tickets, original priorities and the failed geometry."""
import copy
import json
from pathlib import Path
import random

from vllm_ascend.spec_decode.pearl import spec_rhythm as rhythm
from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan, pack_selected_tree_plan
import packing405


def make(budgets, *, one_home=False, seed=0):
    rng = random.Random(seed)
    classes = [('tight', 40.), ('normal', 50.), ('loose', 150.)]
    states = {}
    for i, budget in enumerate(budgets):
        category, slo = classes[i % 3]
        states[i] = rhythm.SpecRhythmRuntimeState(
            i, 0 if one_home else i % 2, slo_tpot_ms=slo, slo_class=category,
            max_gamma=5, output_token_budget=256, delivered_tokens=rng.randrange(1, 64),
            decode_elapsed_ms=rng.randrange(3000), decode_start_elapsed_ms=0.,
            verification_rounds=rng.randrange(20))
    controller = rhythm.SpecRhythmPipelineController(states)
    payloads = {}
    for i, budget in enumerate(budgets):
        raw = cached_cpu_tree_speculation_plan(2, 4, 127, 4096, candidate_budget=8)
        plan = pack_selected_tree_plan(raw, range(budget))
        ticket = controller.new_ticket(i, gamma=budget, eager=False)
        controller.publish([ticket])
        payloads[ticket.proposal_id] = dict(ticket=ticket, plan=plan)
        controller._ready_wait_cycles[i] = rng.randrange(8)
    return controller, payloads


def call(controller, budgets, costs=None, **extra):
    kwargs = dict(active_indices_registered_unique=True, merge_ready_homes=True,
                  max_target_requests=24, verification_budget=96,
                  ready_candidate_counts=dict(enumerate(budgets)))
    kwargs.update(extra)
    if costs is None:
        return packing405.build_plan(controller, list(range(len(budgets))), **kwargs)
    return packing405.build_plan(controller, list(range(len(budgets))), max_target_segments=24,
                                 ready_segment_counts=costs, **kwargs)


def main():
    unchanged = rhythm.SpecRhythmPipelineController.build_plan
    rng = random.Random(0)
    defaults_equal = 0
    constrained_cases = 0
    for trial in range(500):
        n = rng.randrange(1, 65)
        budgets = [rng.randrange(1, 6) for _ in range(n)]
        ctl, payloads = make(budgets, seed=trial)
        actual = copy.deepcopy(ctl)
        flags = dict(priority=bool(trial % 2), target_slo_first=bool(trial % 3),
                     goodput_edf=bool(trial % 5), projected_wait_ms=17.,
                     residual_remaining_service_target_first=bool(trial % 7 == 0))
        expected = unchanged(ctl, list(range(n)), active_indices_registered_unique=True,
                             merge_ready_homes=True, max_target_requests=24,
                             verification_budget=96, ready_candidate_counts=dict(enumerate(budgets)), **flags)
        got = call(actual, budgets, **flags)
        assert got == expected
        assert actual.__dict__ == ctl.__dict__
        defaults_equal += 1
        ctl, payloads = make(budgets, seed=trial)
        costs = packing405.ready_costs(ctl, payloads, range(n))
        identities = dict(ctl.ready)
        got = call(ctl, budgets, costs, **flags)
        ids = got.target_request_indices
        assert len(ids) <= 24 and sum(budgets[i] for i in ids) <= 96
        assert sum(costs[i] for i in ids) <= 24
        assert set(ids).isdisjoint(got.deferred_target_request_indices)
        assert set(ids) | set(got.deferred_target_request_indices) == set(range(n))
        assert all(ctl.ready[i] is ticket for i, ticket in identities.items())
        assert got.target_candidate_budgets == {i: budgets[i] for i in ids}
        assert all(ctl._ready_wait_cycles[i] == 0 for i in ids)
        constrained_cases += 1
    # The observed failed HOST snapshot had24 tickets:four B5 andtwenty B3.
    # It is retained as possibly-stale context, not relabelled as an exact
    # synchronized failure-round trace. This exact legal input reproduces
    # the missing resource independently of that uncertainty.
    budgets = [5] * 4 + [3] * 20
    ctl, payloads = make(budgets, one_home=True)
    old = unchanged(copy.deepcopy(ctl), range(24), max_target_requests=24,
                    verification_budget=96, ready_candidate_counts=dict(enumerate(budgets)))
    costs = packing405.ready_costs(ctl, payloads, range(24))
    assert len(old.target_request_indices) == 24 and sum(costs[i] for i in old.target_request_indices) == 28
    fixed = call(ctl, budgets, costs)
    assert sum(costs[i] for i in fixed.target_request_indices) <= 24
    assert fixed.deferred_target_request_indices
    assert set(fixed.target_request_indices) | set(fixed.deferred_target_request_indices) == set(range(24))
    # Previously submitted replay reservations cannot be silently repacked.
    ctl, payloads = make(budgets, one_home=True)
    try:
        call(ctl, budgets, costs, reserved_target_request_indices=tuple(range(24)),
             reserved_target_home_batch_id=0)
    except ValueError as error:
        assert 'segment limit' in str(error)
    else:
        raise AssertionError('unsafe irreversible reservation accepted')
    assert rhythm.SpecRhythmPipelineController.build_plan is unchanged
    result = dict(status='PASS_HOST405_REAL_CONTROLLER_LEGACY_EQUIVALENCE_AND_SEGMENT_REGRESSION',
                  legacy_default_equivalence_cases=defaults_equal, actual_bounded_controller_cases=constrained_cases,
                  failed_shape=dict(N=24, B5_owners=4, C=80, old_segments=28,
                                    new_selected=list(fixed.target_request_indices),
                                    new_deferred=list(fixed.deferred_target_request_indices),
                                    new_segments=sum(costs[i] for i in fixed.target_request_indices)),
                  original_controller_method_unmodified=True, complete_ready_ticket_identities_preserved=True,
                  no_budget_or_token_truncation=True, unsafe_replay_reservation_rejected=True,
                  source_projection=packing405.descriptor(), global_active_limit=64,
                  NPU_used=False, model_qualification=False, production_permission=False,
                  Raw=None, Goodput=None)
    output = Path(__file__).with_name('HOST405_PACKING_RESULT.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result[k] for k in ('status', 'legacy_default_equivalence_cases',
                                          'actual_bounded_controller_cases', 'NPU_used')}))


if __name__ == '__main__':
    main()
