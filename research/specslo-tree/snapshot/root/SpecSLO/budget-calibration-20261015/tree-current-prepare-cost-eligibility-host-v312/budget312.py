"""Buy branches only when predicted priority-weighted progress pays their cost.

Keep primary capacity, original priority order, hard budgets, actual218 bound
repair and original eager admission. Only the branch eligibility set changes.
"""
import math
import types

import budget290 as original
from cost312 import ratio

_SHAPE = None
_BOUND_FACTORY = None
_SHAPE_CALL = None


def eligible(policy, states, normal):
    from split_budget_v132 import PRIORITY
    spare = max(0, min(12, 24 - len(normal), (96 - 4 * len(normal)) // 4))
    policy.counts['ROOT312_cost_gate_calls'] += 1
    if not spare:
        return frozenset()
    gains = {}
    base = 0.
    for index in normal:
        request = states[index]
        cap = min(8, request.max_gamma or 8)
        primary = float(policy.gain(request, min(4, cap)))
        expanded = float(policy.gain(request, cap))
        if not math.isfinite(primary) or not math.isfinite(expanded) or primary < 0 or expanded < 0:
            raise ValueError('ROOT312 finite nonnegative actual policy gains required')
        weight = PRIORITY[request.slo_class]
        base += weight * primary
        if cap > 4 and expanded > primary:
            gains[index] = (expanded - primary, weight)
    ordered = sorted(gains, key=lambda index: (
        PRIORITY[states[index].slo_class], gains[index][0],
        float(states[index].projected_progress_gap(0.)), -index), reverse=True)[:spare]
    best, best_value, extra = 0, base, 0.
    for count, index in enumerate(ordered, 1):
        delta, weight = gains[index]
        extra += weight * delta
        value = (base + extra) / ratio(count)
        if value > best_value + 1e-12:
            best, best_value = count, value
    policy.counts['ROOT312_possible_branch_owners'] += len(ordered)
    policy.counts['ROOT312_selected_branch_owners'] += best
    policy.counts['ROOT312_cost_rejected_branch_owners'] += len(ordered) - best
    return frozenset(ordered[:best])


def prepared_shape():
    global _SHAPE
    if _SHAPE is None:
        template = original.prepared_shape()
        allocator = template.__globals__['shape_draft_work']
        private_allocator = types.FunctionType(
            allocator.__code__, dict(allocator.__globals__, eligible=eligible),
            allocator.__name__, allocator.__defaults__, allocator.__closure__)
        _SHAPE = types.FunctionType(
            template.__code__, dict(template.__globals__, shape_draft_work=private_allocator),
            template.__name__, template.__defaults__, template.__closure__)
    return _SHAPE


def prepared_bound_shape(function):
    global _BOUND_FACTORY
    if _BOUND_FACTORY is None:
        template = original.prepared_bound_shape
        _BOUND_FACTORY = types.FunctionType(
            template.__code__, dict(template.__globals__, _BOUND={}, prepared_shape=prepared_shape),
            template.__name__, template.__defaults__, template.__closure__)
    return _BOUND_FACTORY(function)


def shape(shaper, **kwargs):
    global _SHAPE_CALL
    if _SHAPE_CALL is None:
        template = original.shape
        _SHAPE_CALL = types.FunctionType(
            template.__code__, dict(template.__globals__, prepared_bound_shape=prepared_bound_shape),
            template.__name__, template.__defaults__, template.__closure__)
    return _SHAPE_CALL(shaper, **kwargs)


work_limit_class = original.work_limit_class
controller_function = original.controller_function
build_plan = original.build_plan
filter_eager_parents = original.filter_eager_parents
