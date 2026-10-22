"""Retain actual431 witnesses; require a nonempty list as a boolean predicate."""
from witness431 import current, need


def snapshot(engine, generation):
    current(engine)
    rows = engine._root431_refill_rows.get(generation)
    need(type(rows) is list and bool(rows), 'actual admissions retained for this generation')
    return list(rows)
