"""HOST-only primary3/alternate3 selection from the unchanged raw8 pool."""
from dataclasses import dataclass
from functools import lru_cache


RAW_ORDER = (0, 1, 2, 4, 5, 6)
PARENTS = (-1, 0, 1, -1, 3, 4)
PRIMARY_DEPTH = 3
SIDE_DEPTH = 3
MAX_GROUPS = 32
MAX_SELECTED = 96
MAX_SIDE_OWNERS = 12


@dataclass(frozen=True)
class Selection:
    plan: object
    raw_pool_indices: tuple[int, ...]
    primary_depth: int
    side_depth: int
    raw_primary_steps: int = 4
    existing_alternate_steps: int = 3


@lru_cache(maxsize=1024, typed=True)
def make(budget, prefix, context=4096):
    if type(budget) is not int or not 1 <= budget <= 6:
        raise ValueError('HOST357 retained B1..6 required')
    from plan269 import make as original_raw8
    from vllm_ascend.spec_decode.pearl.tree import pack_selected_tree_plan
    indices = RAW_ORDER[:budget]
    raw = original_raw8(8, prefix, 0, context)
    selected = pack_selected_tree_plan(raw, indices)
    if tuple(selected.parent_indices.tolist()) != PARENTS[:budget]:
        raise RuntimeError('exact ancestor-closed primary3/alternate3 topology')
    return Selection(selected, indices, min(budget, PRIMARY_DEPTH), max(0, budget - PRIMARY_DEPTH))


def resource_use(budgets):
    budgets = tuple(budgets)
    if not budgets or any(type(b) is not int or not 1 <= b <= 6 for b in budgets):
        raise ValueError('positive selected B1..6 vector')
    branches = sum(b > PRIMARY_DEPTH for b in budgets)
    n = len(budgets)
    if n > MAX_GROUPS or sum(budgets) > MAX_SELECTED or branches > MAX_SIDE_OWNERS or n + branches > MAX_GROUPS:
        raise ValueError('Q4/S32/C96/12-side resource bounds exceeded')
    return dict(N=n, selected=sum(budgets), branches=branches,
                logical_queries=n + sum(budgets), primary_and_side_groups=n + branches,
                physical_rows=128, group_width=4, physical_groups=32)


def spare_side_owners(normal_count):
    if type(normal_count) is not int or not 1 <= normal_count <= MAX_GROUPS:
        raise ValueError('normal owner count1..32 required')
    return min(MAX_SIDE_OWNERS, normal_count, MAX_GROUPS - normal_count,
               (MAX_SELECTED - PRIMARY_DEPTH * normal_count) // SIDE_DEPTH)
