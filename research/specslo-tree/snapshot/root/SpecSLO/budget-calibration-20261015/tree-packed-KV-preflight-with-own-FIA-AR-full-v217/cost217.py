"""Budget units remain retained candidates; no new Native graph or math path."""
from collections import Counter
from dataclasses import replace
import types


_COST_TABLE=None


def original_cost_table():
    global _COST_TABLE
    if _COST_TABLE is None:
        from mapper173 import expected_geometry
        values={}
        for n in range(9,17):
            for q in range(4*n+1,5*n):
                m,s,_=expected_geometry(n,q,'causal')
                if m not in(96,128) or m<=80:
                    raise RuntimeError('ROOT217 exact original mixed3/4 geometry changed')
                values[n,q]=(m,s)
        _COST_TABLE=values
    return _COST_TABLE


def repair_plan(policy, plan, *, context_len, counts):
    counts['shape_calls'] += 1
    normal = plan.normal_budgets
    n=len(plan.requested_normal)
    if (not 9<=n<=16 or len(normal) != n
            or set(normal) != set(plan.requested_normal)
            or set(normal.values()) != {3, 4}):
        return plan
    counts['eligible_mixed_N9to16'] += 1
    old_m,old_s=original_cost_table()[n,n+sum(normal.values())]
    extra = sum(4 - b for b in normal.values())
    allowed = (type(context_len) is int and context_len + 4 <= 4096
               and all(plan.caps[i] >= 4 for i in normal)
               and plan.work_limit.normal_per_request >= 4
               and 4*n <= min(plan.work_limit.normal_candidates, plan.target_limit.candidates)
               and n <= plan.target_limit.rows and 5*n <= plan.target_limit.queries
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
                raise RuntimeError('ROOT217 original grant counters must precede cost repair')
            policy.counts['grant_B3'] -= 1
            policy.counts['grant_B4'] += 1
    counts['repaired_calls'] += 1
    counts['upgraded_owners'] += extra
    counts['added_selected_candidates'] += extra
    counts['proposed_target_physical_queries_saved'] += old_m-80
    counts[f'repaired_old_M{old_m}_calls'] += 1
    counts[f'repaired_N{n}_calls'] += 1
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
            raise RuntimeError('ROOT217 actual current engine generation required')
        counts = state['by_generation'].setdefault(str(generation), Counter())
        return repair_plan(policy, plan, context_len=kwargs['context_len'], counts=counts)
    private_shape = types.FunctionType(shape.__code__, dict(shape.__globals__, shape_draft_work=solver),
                                       shape.__name__, shape.__defaults__, shape.__closure__)
    control.candidate_shape = types.FunctionType(old.__code__, dict(old.__globals__, shape=private_shape),
                                               old.__name__, old.__defaults__, old.__closure__)
    if control.candidate_shape.__closure__ is not old.__closure__:
        raise RuntimeError('ROOT217 original policy closure changed')
    state['private_shape_bindings'] += 1
    return control.candidate_shape


def budget_controls(Parent, cache_state):
    table=original_cost_table()  # Exact original mapper geometry prepared once outside model work.
    state = dict(private_shape_bindings=0, by_generation={},original_cost_table=[dict(N=n,Q=q,M=m,S=s)for(n,q),(m,s)in sorted(table.items())])
    cache_state['cost_aware_N9to16_budget'] = state
    class Controls(Parent):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.mode != 'original' or self.engine is not None:
                raise RuntimeError('ROOT217 bind only before original first engine selection')
            bind_candidate_shape(self, state)
    return Controls


def audit_snapshot(cache, full_generation):
    state = cache['cost_aware_N9to16_budget']
    if (state['private_shape_bindings'] != 1
            or str(full_generation) not in state['by_generation']
            or state['by_generation'][str(full_generation)]['shape_calls'] <= 0):
        raise RuntimeError('ROOT217 actual fresh full463 shape execution missing')
    expected=[dict(N=n,Q=q,M=m,S=s)for(n,q),(m,s)in sorted(original_cost_table().items())]
    if state['original_cost_table']!=expected:
        raise RuntimeError('ROOT217 actual precomputed original cost table differs')
    for row in state['by_generation'].values():
        if (any(type(v) is not int or v < 0 for v in row.values())
                or row.get('eligible_mixed_N9to16', 0) != row.get('repaired_calls', 0) + row.get('blocked_by_original_limits', 0)
                or row.get('added_selected_candidates', 0) != row.get('upgraded_owners', 0)
                or row.get('repaired_calls',0)!=sum(row.get('repaired_N'+str(n)+'_calls',0)for n in range(9,17))
                or row.get('repaired_calls',0)!=row.get('repaired_old_M96_calls',0)+row.get('repaired_old_M128_calls',0)
                or row.get('proposed_target_physical_queries_saved', 0) != 16 * row.get('repaired_old_M96_calls', 0)+48 * row.get('repaired_old_M128_calls', 0)):
            raise RuntimeError('ROOT217 retained repair accounting mismatch')
    return True
