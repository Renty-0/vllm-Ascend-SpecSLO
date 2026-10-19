"""Keep the separately reserved zero page out of all branch-copy scratch."""
from layout358 import is_fork
from layout358 import layout as original


def layout(plans, ids, tables, scratch_ids, **kwargs):
    if 526 in ids:
        raise ValueError('reserved zero owner is never a physical writer')
    return original(plans, ids, tables, [owner for owner in scratch_ids if owner != 526], **kwargs)
