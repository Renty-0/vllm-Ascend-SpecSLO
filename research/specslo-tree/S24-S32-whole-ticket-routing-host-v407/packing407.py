"""HOST dual-route packer; whole tickets, one original age update per call.

This is not installed in LIVE generation. Independent S32 model qualification,
dual physical dispatch and full global64 own-AR remain separate requirements.
"""
from dataclasses import dataclass
from types import MappingProxyType

import packing398
import packing401

GLOBAL_LIMIT = 64
NARROW_ROWS = 24
WIDE_ROWS = 32
WIDE_CANDIDATES = 88


@dataclass(frozen=True)
class ReadyResources:
    budgets: object
    segments: object

    def __post_init__(self):
        budgets, segments = dict(self.budgets), dict(self.segments)
        if (set(budgets) != set(segments)
                or any(type(i) is not int or i < 0 for i in budgets)
                or any(type(b) is not int or not 1 <= b <= 5 for b in budgets.values())
                or any(segments[i] != 1 + int(b == 5) for i, b in budgets.items())):
            raise ValueError('HOST407 complete qualified ready-ticket resources required')
        object.__setattr__(self, 'budgets', MappingProxyType(budgets))
        object.__setattr__(self, 'segments', MappingProxyType(segments))


def collect(controller, payloads, active):
    segments = packing398.ready_costs(controller, payloads, active)
    budgets = {i: payloads[controller.ready[i].proposal_id]['plan'].candidate_budget
               for i in segments}
    return ReadyResources(budgets, segments)


def build_plan(controller, active, *, ready_resources, verification_budget,
               max_target_requests=None, **kwargs):
    active = tuple(active)
    if len(active) > GLOBAL_LIMIT or len(set(active)) != len(active):
        raise ValueError('HOST407 unique global64 active scope required')
    budgets = ready_resources.budgets
    segments = ready_resources.segments
    actual_ready = {i for i in active if i in controller.ready}
    if (set(budgets) != actual_ready
            or any(controller.ready[i].gamma != b for i, b in budgets.items())):
        raise ValueError('HOST407 resource snapshot and actual whole tickets differ')
    roof = 96 if verification_budget is None else verification_budget
    if type(roof) is not int or not 0 <= roof <= 96:
        raise ValueError('HOST407 paid candidate roof at most96 required')
    if max_target_requests is not None and (
            type(max_target_requests) is not int or max_target_requests <= 0):
        raise ValueError('HOST407 positive explicit target request ceiling required')
    supplied = kwargs.pop('ready_candidate_counts', None)
    if supplied is not None and dict(supplied) != dict(budgets):
        raise ValueError('HOST407 candidate counts must match exact ready metadata')
    reserved = kwargs.get('reserved_target_request_indices')
    if reserved is None:
        wide = len(budgets) > NARROW_ROWS and all(b <= 3 for b in budgets.values())
    else:
        reserved = tuple(reserved)
        wide = len(reserved) > NARROW_ROWS
        if wide and any(i not in budgets or budgets[i] > 3 for i in reserved):
            raise ValueError('HOST407 wide reservation contains unsupported deep tickets')
        kwargs['reserved_target_request_indices'] = reserved
    limit = WIDE_ROWS if wide else NARROW_ROWS
    if max_target_requests is not None:
        limit = min(limit, max_target_requests)
    paid_roof = min(roof, WIDE_CANDIDATES) if wide else roof
    common = dict(kwargs, max_target_requests=limit, verification_budget=paid_roof,
                  ready_candidate_counts=budgets)
    if wide:
        result = packing401.build_plan(controller, active, **common,
            max_large_groups401=NARROW_ROWS,
            ready_large_groups401={i: int(b == 3) for i, b in budgets.items()})
    else:
        result = packing398.build_plan(controller, active, **common,
            max_target_segments=NARROW_ROWS, ready_segment_counts=segments)
    selected = result.target_request_indices
    if (len(selected) > limit or sum(budgets[i] for i in selected) > paid_roof
            or result.target_candidate_budgets != {i: budgets[i] for i in selected}):
        raise RuntimeError('HOST407 selected complete proposal conservation')
    if len(selected) > NARROW_ROWS:
        if any(budgets[i] > 3 for i in selected) or sum(budgets[i] == 3 for i in selected) > NARROW_ROWS:
            raise RuntimeError('HOST407 actual wide target geometry overrun')
    elif sum(segments[i] for i in selected) > NARROW_ROWS:
        raise RuntimeError('HOST407 actual narrow primary/side geometry overrun')
    return result


def descriptor():
    return dict(global_active_limit=GLOBAL_LIMIT, wide_N=WIDE_ROWS,
                wide_C=WIDE_CANDIDATES, wide_B3_owner_limit=NARROW_ROWS,
                narrow_N=NARROW_ROWS, narrow_C=96, narrow_primary_side_groups=NARROW_ROWS,
                complete_tickets_only=True, original_class_method_unchanged=True,
                original_priority_and_wait_age=True, one_original_build_call=True,
                LIVE_installed=False, NPU_qualified=False, Raw=None, Goodput=None)
