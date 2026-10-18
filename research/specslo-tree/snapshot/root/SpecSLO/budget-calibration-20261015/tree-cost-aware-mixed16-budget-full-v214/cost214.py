"""Budget units remain retained candidates; no new Native graph or math path."""
from collections import Counter
from dataclasses import replace
import types


def repair_plan(policy, plan, *, context_len, counts):
    counts['shape_calls'] += 1
    normal = plan.normal_budgets
    if (len(plan.requested_normal) != 16 or len(normal) != 16
            or set(normal) != set(plan.requested_normal)
            or set(normal.values()) != {3, 4}):
        return plan
    counts['eligible_mixed16'] += 1
    extra = sum(4 - b for b in normal.values())
    allowed = (type(context_len) is int and context_len + 4 <= 4096
               and all(plan.caps[i] >= 4 for i in normal)
               and plan.work_limit.normal_per_request >= 4
               and 64 <= min(plan.work_limit.normal_candidates, plan.target_limit.candidates)
               and 16 <= plan.target_limit.rows and 80 <= plan.target_limit.queries
               and plan.allocated_selected_candidates + extra <= plan.available_selected_candidates)
    if not allowed:
        counts['blocked_by_original_limits'] += 1
        return plan
    upgraded = {i: 4 for i in normal}
    # Original immutable DTO reruns every cap, role, reserve and ownership check.
    result = replace(plan, normal_budgets=upgraded)
    for i, before in normal.items():
        if before == 3:
            if policy.counts['grant_B3'] <= 0:
                raise RuntimeError('ROOT214 original grant counters must precede cost repair')
            policy.counts['grant_B3'] -= 1
            policy.counts['grant_B4'] += 1
    counts['repaired_calls'] += 1
    counts['upgraded_owners'] += extra
    counts['added_selected_candidates'] += extra
    counts['proposed_target_physical_queries_saved'] += 48
    # Already max B4 before repair: neither draft stages nor eager filtering change.
    return result


def bind_candidate_shape(control, state):
    old = control.candidate_shape
    shape = old.__globals__['shape']
    original_solver = shape.__globals__['shape_draft_work']
    def solver(policy, shaper, **kwargs):
        plan = original_solver(policy, shaper, **kwargs)
        generation = getattr(control.engine, '_dynamic_tree_generation', None)
        if type(generation) is not int or generation < 0:
            raise RuntimeError('ROOT214 actual current engine generation required')
        counts = state['by_generation'].setdefault(str(generation), Counter())
        return repair_plan(policy, plan, context_len=kwargs['context_len'], counts=counts)
    private_shape = types.FunctionType(shape.__code__, dict(shape.__globals__, shape_draft_work=solver),
                                       shape.__name__, shape.__defaults__, shape.__closure__)
    control.candidate_shape = types.FunctionType(old.__code__, dict(old.__globals__, shape=private_shape),
                                               old.__name__, old.__defaults__, old.__closure__)
    if control.candidate_shape.__closure__ is not old.__closure__:
        raise RuntimeError('ROOT214 original policy closure changed')
    state['private_shape_bindings'] += 1
    return control.candidate_shape


def budget_controls(Parent, cache_state):
    state = dict(private_shape_bindings=0, by_generation={})
    cache_state['cost_aware_mixed16_budget'] = state
    class Controls(Parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.mode != 'original' or self.engine is not None:
                raise RuntimeError('ROOT214 bind only before original first engine selection')
            bind_candidate_shape(self, state)
    return Controls


def audit_snapshot(cache, full_generation):
    state = cache['cost_aware_mixed16_budget']
    if (state['private_shape_bindings'] != 1
            or str(full_generation) not in state['by_generation']
            or state['by_generation'][str(full_generation)]['shape_calls'] <= 0):
        raise RuntimeError('ROOT214 actual fresh full463 shape execution missing')
    for row in state['by_generation'].values():
        if (any(type(v) is not int or v < 0 for v in row.values())
                or row.get('eligible_mixed16', 0) != row.get('repaired_calls', 0) + row.get('blocked_by_original_limits', 0)
                or row.get('added_selected_candidates', 0) != row.get('upgraded_owners', 0)
                or row.get('proposed_target_physical_queries_saved', 0) != 48 * row.get('repaired_calls', 0)):
            raise RuntimeError('ROOT214 retained repair accounting mismatch')
    return True
