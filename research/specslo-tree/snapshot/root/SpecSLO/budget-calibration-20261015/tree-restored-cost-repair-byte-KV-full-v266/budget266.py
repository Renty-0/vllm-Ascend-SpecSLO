"""Spare branch groups with the real, instance-bound ROOT218 repair retained.

The original repair solver runs on every LIVE call. Only its allocator cell is
replaced; control, generation counters and repair code retain their identities.
"""
import types
from pathlib import Path

from budget260 import build_plan, eligible, prepared_shape

_SHAPES = {}


def _cell(value):
    return (lambda: value).__closure__[0]


def prepared_bound_shape(function):
    if function in _SHAPES:
        return _SHAPES[function]
    from cost218 import bind_candidate_shape
    original_shape = function.__globals__['shape']
    solver = original_shape.__globals__['shape_draft_work']
    if (solver.__code__.co_filename != bind_candidate_shape.__code__.co_filename
            or solver.__qualname__ != bind_candidate_shape.__qualname__ + '.<locals>.solver'
            or set(solver.__code__.co_freevars) != {'control', 'state', 'original_solver'}):
        raise RuntimeError('ROOT266 real instance-bound ROOT218 solver required')
    prior_cells = dict(zip(solver.__code__.co_freevars, solver.__closure__))
    control = prior_cells['control'].cell_contents
    if control.candidate_shape is not function:
        raise RuntimeError('ROOT266 current original control function identity')
    template = prepared_shape()
    allocator = template.__globals__['shape_draft_work']
    cells = tuple(_cell(allocator) if name == 'original_solver' else prior_cells[name]
                  for name in solver.__code__.co_freevars)
    repaired_solver = types.FunctionType(solver.__code__, solver.__globals__,
                                        solver.__name__, solver.__defaults__, cells)
    for name in ('control', 'state'):
        pos = solver.__code__.co_freevars.index(name)
        if repaired_solver.__closure__[pos] is not solver.__closure__[pos]:
            raise RuntimeError('ROOT266 original real repair state cell changed')
    value = types.FunctionType(template.__code__,
                              dict(original_shape.__globals__, shape_draft_work=repaired_solver),
                              template.__name__, template.__defaults__, template.__closure__)
    _SHAPES[function] = value
    return value


def shape(shaper, **kwargs):
    function = type(shaper).shape
    cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
    if 'policy' not in cells:
        raise RuntimeError('ROOT266 same original shaper policy closure')
    policy = cells['policy'].cell_contents
    result = prepared_bound_shape(function)(policy, shaper, **kwargs)
    policy.counts['ROOT266_actual_B5_grants'] += sum(
        value == 5 for value in result.normal_budgets.values())
    if len(result.requested_normal) > 8:
        policy.counts['ROOT266_actual_B5_grants_N_gt8'] += sum(
            value == 5 for value in result.normal_budgets.values())
    return result
