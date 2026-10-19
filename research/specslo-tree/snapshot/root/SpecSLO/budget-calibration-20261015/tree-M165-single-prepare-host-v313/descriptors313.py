"""Same full immutable304 proof; physical writer count is165."""
from descriptors304 import CheckedRows
def separate(checked, query_slots):
    if type(checked) is not CheckedRows:
        raise ValueError('ROOT304 original complete immutable row proof required')
    if len(query_slots) != 165 or any((type(slot) is not int or not 0 <= slot < checked.rows for slot in query_slots)):
        raise ValueError('ROOT304 exact physical120 canonical writers')
    writers = {}
    for row, slot in enumerate(query_slots):
        writers.setdefault(slot, []).append(row)
    aliases = [-1] * 165
    spans = []
    prefix_rows, alias_rows = (0, 0)
    for source, destination in zip(checked.source, checked.destination, strict=True):
        if source == destination:
            continue
        if destination in writers:
            raise ValueError('ROOT304 branch destination overlaps canonical writer')
        if source in writers:
            candidates = writers[source]
            if len(candidates) != 1:
                raise ValueError('ROOT304 ambiguous current alias writer')
            row = candidates[0]
            if aliases[row] != -1:
                raise ValueError('ROOT304 at most one side alias per current writer')
            aliases[row] = destination
            alias_rows += 1
        else:
            if spans and source == spans[-1][0] + spans[-1][2] and (destination == spans[-1][1] + spans[-1][2]):
                spans[-1][2] += 1
            else:
                spans.append([source, destination, 1])
            prefix_rows += 1
    if len(spans) > 12 or prefix_rows + alias_rows != checked.nonidentity:
        raise ValueError('ROOT304 exact complete split and at most12 prefix spans')
    return dict(spans=spans + [[0, 0, 0] for _ in range(12 - len(spans))], query_alias_destination=aliases, prefix_copies=prefix_rows, current_aliases=alias_rows, original_copies=checked.nonidentity, span_count=len(spans), legacy_capture_grid=checked.grid)
