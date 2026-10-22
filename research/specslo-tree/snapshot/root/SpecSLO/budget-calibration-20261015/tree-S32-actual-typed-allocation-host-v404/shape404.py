"""HOST prototype: pay S32 resources through the actual typed callback.

Importing this module installs nothing. Wide target math/cost and complete
global64 own-AR are required before this policy can be used in serving.
"""
from collections import Counter
from dataclasses import dataclass, field, replace
import inspect
from pathlib import Path

import normal_admission_v140
from budget_policy import PRIORITY

BASE = Path('/root/SpecSLO/budget-calibration-20261015/tree-high-normal-cap4-native-families-v148')
MAX_ACTIVE = 64
MAX_NORMAL = 32
LARGE_GROUPS = 24
WIDE_CANDIDATES = 88


@dataclass
class Binding:
    method: object
    policy: object
    original_shape: object
    generation: int
    counts: Counter = field(default_factory=Counter)


def bind(shaper, generation):
    method = shaper.shape.__func__
    if Path(method.__code__.co_filename) != BASE / 'candidate_controls_v132.py':
        raise RuntimeError('ROOT404 actual installed typed callback required')
    policy = inspect.getclosurevars(method).nonlocals.get('policy')
    original = method.__globals__.get('shape')
    if (policy is None or not callable(original)
            or Path(original.__code__.co_filename) != BASE / 'split_shape_v132.py'):
        raise RuntimeError('ROOT404 exact original typed policy closure required')
    return Binding(method, policy, original, generation)


def caps(binding, shaper, admitted, states, wait_ms):
    """Class, original deferred age, urgency, then deterministic owner order."""
    ages = normal_admission_v140._state(binding.policy)['ages']
    limits = {i: min(3, shaper.max_gamma, states[i].max_gamma or shaper.max_gamma)
              for i in admitted}
    metrics = {i: states[i].scheduling_metrics(wait_ms) for i in admitted}
    eligible = [i for i in admitted if limits[i] == 3]
    ranked = sorted(eligible, key=lambda i: (
        PRIORITY[states[i].slo_class], ages.get(i, 0),
        metrics[i][1], metrics[i][0] * states[i].expected_acceptance_benefit, -i),
        reverse=True)
    large = set(ranked[:LARGE_GROUPS])
    return {i: min(value, 3 if i in large else 2) for i, value in limits.items()}


def shape(engine, shaper, **kwargs):
    generation = getattr(engine, '_dynamic_tree_generation', None)
    if type(generation) is not int or generation <= 0:
        raise RuntimeError('ROOT404 original positive phase generation required')
    actual = getattr(engine, '_typed404_binding', None)
    if actual is None:
        actual = bind(shaper, generation)
        engine._typed404_binding = actual
    legacy = normal_admission_v140._state(actual.policy)
    if legacy['generation'] != generation or shaper.shape.__func__ is not actual.method:
        raise RuntimeError('ROOT404 original callback and issued phase generation required')
    if actual.generation != generation:
        actual.generation = generation
        actual.counts.clear()
    # Original select is read-only. Calling the actual typed callback below
    # performs the sole original finish/age update for this plan.
    selected = normal_admission_v140.select(
        actual.policy, kwargs['normal_request_indices'], kwargs['eager_request_indices'],
        kwargs['states'], batch_size=kwargs.get('batch_size'))
    admitted = selected.admitted
    if len(admitted) <= LARGE_GROUPS:
        result = actual.original_shape(actual.policy, shaper, **kwargs)
        actual.counts['narrow_original_calls'] += 1
        return result
    if not LARGE_GROUPS < len(admitted) <= MAX_NORMAL:
        raise RuntimeError('ROOT404 original normal admission32 required')
    limits = caps(actual, shaper, admitted, kwargs['states'], kwargs['projected_wait_ms'])
    shadow = dict(kwargs['states'])
    for owner, value in limits.items():
        shadow[owner] = replace(shadow[owner], max_gamma=value)
    roof = kwargs.get('verification_roof')
    roof = actual.policy.candidate_roof if roof is None else roof
    if type(roof) is not int or roof < 0:
        raise ValueError('ROOT404 paid integer candidate roof required')
    work = dict(kwargs, states=shadow, verification_roof=min(roof, WIDE_CANDIDATES))
    result = actual.original_shape(actual.policy, shaper, **work)
    budgets = result.normal_budgets
    if not (result.requested_normal == admitted
            and set(budgets) <= set(admitted)
            and all(type(b) is int and 1 <= b <= limits[i] for i, b in budgets.items())
            and sum(budgets.values()) <= min(roof, WIDE_CANDIDATES)
            and sum(b == 3 for b in budgets.values()) <= LARGE_GROUPS
            and all(b == 3 for b in result.eager_budgets.values())
            and result.allocated_selected_candidates <= 192):
        raise RuntimeError('ROOT404 complete normal/eager paid-resource invariant')
    actual.counts['wide_typed_calls'] += 1
    actual.counts['wide_normal_rows'] += len(budgets)
    actual.counts['wide_B3_rows'] += sum(b == 3 for b in budgets.values())
    actual.counts['wide_B2_cap_rows'] += sum(value == 2 for value in limits.values())
    return result


def snapshot(engine):
    actual = getattr(engine, '_typed404_binding', None)
    return dict(available=actual is not None, global_active_limit=MAX_ACTIVE,
                normal_work_limit=MAX_NORMAL, wide_normal_C=WIDE_CANDIDATES,
                wide_normal_B3_owner_limit=LARGE_GROUPS,
                original_normal_admission_and_defer_age=True,
                original_policy_gain_and_observations=True,
                original_B3_eager_window_and_role_limits_required=True,
                generation=actual.generation if actual else None,
                counts=dict(actual.counts) if actual else {},
                production_runtime_installed=False, NPU_qualified=False,
                Raw=None, Goodput=None)
