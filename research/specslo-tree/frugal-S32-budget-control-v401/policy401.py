"""HOST candidate: pay for S32/C88 geometry before making normal proposals.

This module is not installed into a serving runtime. S32 model qualification,
paired device costs, and complete global64 own-AR remain separate gates.
"""
from dataclasses import replace
import math
from types import MappingProxyType

MAX_ACTIVE_REQUESTS = 64
MAX_WIDE_REQUESTS = 32
LARGE_QUERY_GROUPS = 24
WIDE_CANDIDATES = 88
CLASS_PRIORITY = MappingProxyType({'tight': 3, 'normal': 2, 'loose': 1, None: 0})


def use_wide_normal_scope(*, batch_size, normal_indices):
    if type(batch_size) is not int or not 0 <= batch_size <= MAX_ACTIVE_REQUESTS:
        raise ValueError('actual global64 scope required')
    return batch_size > LARGE_QUERY_GROUPS and LARGE_QUERY_GROUPS < len(normal_indices) <= MAX_WIDE_REQUESTS


def normal_caps(shaper, normal, states, projected_wait_ms):
    if not math.isfinite(projected_wait_ms) or projected_wait_ms < 0:
        raise ValueError('finite nonnegative projected wait required')
    if len(set(normal)) != len(normal) or not 25 <= len(normal) <= MAX_WIDE_REQUESTS:
        raise ValueError('unique finite wide normal requests required')
    caps = {i: min(shaper.max_gamma, states[i].max_gamma or shaper.max_gamma, 3) for i in normal}
    if any(v < 1 for v in caps.values()):
        raise ValueError('positive original request cap required')
    metrics = {i: states[i].scheduling_metrics(projected_wait_ms) for i in normal}
    if any(states[i].slo_class not in CLASS_PRIORITY for i in normal):
        raise ValueError('explicit supported original SLO class required')
    eligible = [i for i in normal if caps[i] == 3]
    ranked = sorted(eligible, key=lambda i: (
        CLASS_PRIORITY[states[i].slo_class],
        shaper._normal_wait_cycles.get(i, 0),
        metrics[i][1], metrics[i][0] * states[i].expected_acceptance_benefit, -i), reverse=True)
    large = set(ranked[:LARGE_QUERY_GROUPS])
    return {i: min(cap, 3 if i in large else 2) for i, cap in caps.items()}


def shape(shaper, **kwargs):
    """Keep original shaper accounting/age; supply per-request resource caps.

    Caps affect only newly proposed work. Existing tickets and real states are
    never shortened. Optional eager work is disabled in this unqualified wide
    candidate; narrow scopes delegate to the original shaper without edits.
    """
    normal = tuple(dict.fromkeys(kwargs['normal_request_indices']))
    if not use_wide_normal_scope(batch_size=kwargs['batch_size'], normal_indices=normal):
        return shaper.shape(**kwargs)
    capped = normal_caps(shaper, normal, kwargs['states'], kwargs['projected_wait_ms'])
    original_states = kwargs['states']
    shadow_states = dict(original_states)
    for i, cap in capped.items():
        shadow_states[i] = replace(original_states[i], max_gamma=cap)
    roof = kwargs.get('verification_roof')
    if roof is None:
        roof = shaper.verification_roof(kwargs['batch_size'], kwargs['context_len'])
    if type(roof) is not int or roof < 0:
        raise ValueError('explicit nonnegative paid candidate roof required')
    work = dict(kwargs, normal_request_indices=normal, eager_request_indices=(),
                eager_reserve_tokens=0, eager_token_cap=0, states=shadow_states,
                verification_roof=min(roof, WIDE_CANDIDATES))
    plan = shaper.shape(**work)
    budgets = plan.normal_budgets
    if not (set(budgets) <= set(normal) and not plan.eager_budgets
            and all(type(b) is int and 1 <= b <= capped[i] for i, b in budgets.items())
            and sum(budgets.values()) <= min(roof, WIDE_CANDIDATES)
            and sum(b == 3 for b in budgets.values()) <= LARGE_QUERY_GROUPS
            and set(budgets).isdisjoint(plan.deferred_normal_request_indices)
            and set(budgets) | set(plan.deferred_normal_request_indices) == set(normal)):
        raise RuntimeError('paid whole-proposal S32/C88 allocation invariant')
    return plan
