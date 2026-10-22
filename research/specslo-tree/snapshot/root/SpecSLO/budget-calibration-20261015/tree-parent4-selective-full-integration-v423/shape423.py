"""Use the actual typed V148 allocator with only normal admission capped24."""
from dataclasses import dataclass
import hashlib
import inspect
from pathlib import Path
import textwrap

import admission423
import normal_admission_v140

BASE = Path('/root/SpecSLO/budget-calibration-20261015/tree-high-normal-cap4-native-families-v148')


@dataclass(frozen=True)
class Binding:
    method: object
    policy: object
    original_shape: object
    projected_shape: object
    original_sha256: str
    projected_sha256: str


@dataclass(frozen=True)
class Work:
    owner: object
    generation: int
    ids: tuple
    eager_ids: tuple
    parents: tuple


def filter_parents(engine, indices, controller, payloads, counters, local_states):
    """Keep the original pre-window location and accounting; add proved B4."""
    from live423 import permission
    from parents421 import collect
    permission(engine)
    eligible = collect(controller, payloads, indices, local_states)
    selected = {p.owner for p in eligible}
    for budget in range(1, 6):
        for suffix in ('seen', 'excluded_unqualified'):
            counters.setdefault(f'spec_rhythm_eager_parent_B{budget}_{suffix}', 0)
    key = 'spec_rhythm_eager_parent_filter_calls'
    counters[key] = counters.get(key, 0)+1
    for index in indices:
        budget = payloads[controller.ready[index].proposal_id]['plan'].candidate_budget
        key = f'spec_rhythm_eager_parent_B{budget}_seen'
        counters[key] += 1
        if index not in selected:
            key = f'spec_rhythm_eager_parent_B{budget}_excluded_unqualified'
            counters[key] += 1
            if budget == 5:
                key = 'spec_rhythm_eager_parent_B5_deferred_first_candidate'
                counters[key] = counters.get(key, 0)+1
    return [i for i in indices if i in selected]


def binding(shaper, engine):
    from callback423 import inspect_actual
    method, policy, original = inspect_actual(engine, shaper)
    before = textwrap.dedent(inspect.getsource(original))
    old, new = 'if len(raw_normal)>32:', 'if len(raw_normal)>24:'
    if before.count(old) != 1:
        raise RuntimeError('ROOT423 one exact original typed admission threshold')
    after = before.replace(old, new)
    if after.replace(new, old) != before:
        raise RuntimeError('ROOT423 complete typed-shape source inverse')
    namespace = dict(original.__globals__, select_normal=admission423.select, finish_normal=admission423.finish)
    exec(compile(after, str(Path(__file__).resolve()) + '::same-typed-allocator-normal24',
                 'exec', dont_inherit=True), namespace)
    return Binding(method, policy, original, namespace['shape'],
                   hashlib.sha256(before.encode()).hexdigest(),
                   hashlib.sha256(after.encode()).hexdigest())


def shape(engine, shaper, *, current_controller, current_payloads, current_local_states, **kwargs):
    actual = getattr(engine, '_normal423_binding', None)
    if actual is None:
        actual = binding(shaper, engine)
        engine._normal423_binding = actual
    if shaper.shape.__func__ is not actual.method:
        raise RuntimeError('ROOT423 original shaper callback changed')
    generation = getattr(engine, '_dynamic_tree_generation', None)
    legacy = normal_admission_v140._state(actual.policy)
    if type(generation) is not int or generation <= 0 or legacy['generation'] != generation:
        raise RuntimeError('ROOT423 exact real existing phase generation required')
    current = getattr(actual.policy, admission423.STATE, None)
    if current is None or current['generation'] != generation:
        admission423.reset(actual.policy, generation)
    from live423 import permission
    from parents421 import collect
    from shape422 import propose
    permission(engine)
    parents = collect(current_controller, current_payloads,
                      kwargs['eager_request_indices'], current_local_states)
    result = propose(actual.policy, shaper, actual.projected_shape, kwargs,
                     parent_budgets={p.owner: p.budget for p in parents})
    plan = result.plan
    if result.future4_owners:
        # The retained proposal performs the original typed allocation and
        # age finish. Account its additional actual work in the same original
        # shape counters once, since the legacy shape wrapper was not called.
        policy = actual.policy
        normal = tuple(plan.normal_budgets)
        policy.counts['draft_normal_N24_shape_calls'] += 1
        policy.counts['normal_cap4_shape_calls'] += 1
        policy.counts['normal_cap4_limited_owners'] += sum(
            min(8, kwargs['states'][i].max_gamma or 8, shaper.max_gamma, 5) > 4 for i in normal)
        for budget in (1, 2):
            policy.counts.setdefault(f'partial_eager_grant_deferred_B{budget}', 0)
        policy.counts['actual_draft_normal_N24_work_calls'] += 1
        policy.counts['actual_draft_nonempty_work_calls'] += 1
        policy.counts['actual_draft_work_rows'] += 28
        policy.counts['actual_draft_normal_rows'] += 24
        policy.counts['actual_draft_eager_rows'] += 4
        policy.counts['split_draft_final_normal_candidates'] += sum(plan.normal_budgets.values())
        policy.counts['split_draft_final_eager_candidates'] += sum(plan.eager_budgets.values())
        policy.counts['split_draft_final_unused_candidates'] += plan.unused_draft_candidates
        policy.counts['ROOT423_actual_E4_work_calls'] += 1
        refs = []
        for owner in result.future4_owners:
            ticket = current_controller.validate_verification(owner)
            payload = current_payloads[ticket.proposal_id]
            refs.append((owner, payload['plan'], tuple(payload['row'])))
        engine._root423_current_work = Work(engine, generation,
            tuple((*plan.normal_budgets, *plan.eager_budgets)),
            tuple(plan.eager_budgets), tuple(refs))
    else:
        engine._root423_current_work = None
    phases = getattr(engine, '_root423_modes', None)
    if phases is None:
        phases = engine._root423_modes = {}
    counts = phases.setdefault(str(generation), {})
    counts[result.mode] = counts.get(result.mode, 0)+1
    return plan


def snapshot(engine):
    actual = getattr(engine, '_normal423_binding', None)
    if actual is None:
        return dict(available=False, global_active_limit=64, normal_work_limit=24)
    return dict(available=True, global_active_limit=64, normal_work_limit=24,
                original_typed_shape_sha256=actual.original_sha256,
                projected_typed_shape_sha256=actual.projected_sha256,
                normal_admission=admission423.snapshot(actual.policy),
                original_gain_model_unchanged=True,
                original_gain_model_and_paid_role_limits_unchanged=False,
                legacy_paid_limits_preserved=True,
                new_exact_E4_paid_work_limit=dict(normal_rows=24, eager_rows=4,
                    normal_per_request=4, eager_per_request=4, total_candidates=112),
                modes_by_generation=dict(getattr(engine, '_root423_modes', {})),
                same_original_policy_object=True, Native_flags_assigned=False)
