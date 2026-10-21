"""Exercise real controller tickets, original priorities and the failed geometry."""
import copy
import json
from pathlib import Path
import random

from vllm_ascend.spec_decode.pearl import spec_rhythm as rhythm
from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan, pack_selected_tree_plan
import packing409


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
        return packing409.build_plan(controller, list(range(len(budgets))), **kwargs)
    return packing409.build_plan(controller, list(range(len(budgets))), max_target_segments=24,
                                 ready_segment_counts=costs, **kwargs)


def main():
    import packing398
    import packing401
    unchanged = rhythm.SpecRhythmPipelineController.build_plan
    rng = random.Random(0)
    narrow = wide = actual_wide = 0
    for trial in range(600):
        n = rng.randrange(1, 65)
        budgets = [rng.randrange(1, 4 if trial % 2 else 6) for _ in range(n)]
        ctl, payloads = make(budgets, seed=trial)
        expected = copy.deepcopy(ctl)
        resources = packing409.ready_costs(ctl, payloads, range(n))
        flags = dict(active_indices_registered_unique=True, merge_ready_homes=True,
                     priority=bool(trial % 2), target_slo_first=bool(trial % 3),
                     goodput_edf=bool(trial % 5), projected_wait_ms=17.,
                     max_target_requests=32, verification_budget=96)
        if n > 24 and max(budgets) <= 3:
            truth = packing401.build_plan(expected, range(n), **dict(flags,
                verification_budget=88, ready_candidate_counts=dict(enumerate(budgets))),
                max_large_groups401=24,
                ready_large_groups401={i: int(b == 3) for i,b in enumerate(budgets)})
            wide += 1
        else:
            truth = packing398.build_plan(expected, range(n), **dict(flags,
                max_target_requests=24, ready_candidate_counts=dict(enumerate(budgets))),
                max_target_segments=24, ready_segment_counts=resources.segments)
            narrow += 1
        tickets = dict(ctl.ready)
        got = packing409.build_plan(ctl, range(n), ready_resources=resources, **flags)
        assert got == truth and ctl.__dict__ == expected.__dict__
        assert all(ctl.ready[i] is ticket for i,ticket in tickets.items())
        actual_wide += len(got.target_request_indices) > 24
    # A previously submitted complete wide reservation can coexist with an
    #unselected deep ticket. Its payload and exact32 rows remain unchanged.
    budgets = [3] * 24 + [2] * 8 + [5]
    ctl, payloads = make(budgets, one_home=True)
    resources = packing409.ready_costs(ctl, payloads, range(33))
    got = packing409.build_plan(ctl, range(33), ready_resources=resources,
        active_indices_registered_unique=True, merge_ready_homes=True,
        max_target_requests=32, verification_budget=96,
        reserved_target_request_indices=tuple(range(32)), reserved_target_home_batch_id=0)
    assert got.target_request_indices == tuple(range(32))
    assert got.target_candidate_budgets == dict(enumerate(budgets[:32]))
    ctl, payloads = make([5] * 4 + [3] * 20, one_home=True)
    resources = packing409.ready_costs(ctl, payloads, range(24))
    try:
        packing409.build_plan(ctl, range(24), ready_resources=resources,
            active_indices_registered_unique=True, merge_ready_homes=True,
            max_target_requests=32, verification_budget=96,
            reserved_target_request_indices=tuple(range(24)), reserved_target_home_batch_id=0)
    except ValueError:
        pass
    else:
        raise AssertionError('unsafe narrow28-group reservation accepted')
    assert rhythm.SpecRhythmPipelineController.build_plan is unchanged
    assert narrow and wide and actual_wide
    result = dict(status='PASS_HOST409_ACTUAL_CPU_TREE_PLANS_DUAL_PACKING_AND_RESERVATIONS',
        exact_narrow_route_cases=narrow, exact_wide_route_cases=wide,
        actual_selected_N_gt24_cases=actual_wide,
        real_CPU_tree_plan_resource_collection=True,
        actual_source_callback_original_controller_class_preserved=True,
        exact_priority_wait_age_and_complete_ticket_identity=True,
        safe_wide_reservation_with_unselected_deep_ticket=True,
        unsafe_narrow28_group_reservation_rejected=True,
        global_active_limit=64, NPU_used=False, model_constructors=0,
        production_permission=False, Raw=None, Goodput=None)
    (Path(__file__).resolve().parent / 'HOST409_PACKING_RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
