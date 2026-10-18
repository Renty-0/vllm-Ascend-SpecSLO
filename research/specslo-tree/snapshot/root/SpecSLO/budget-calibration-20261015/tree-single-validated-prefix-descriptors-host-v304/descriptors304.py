"""HOST prototype: one full row-map validation, then exact span/alias split.

No serving path imports this module. The immutable map keeps the original
anti-alias proof valid while deriving descriptors without three more full
1792-row validations or an unused1792-row prefix staging buffer.
"""
from dataclasses import dataclass


@dataclass(frozen=True, init=False)
class CheckedRows:
    source: tuple
    destination: tuple
    rows: int
    nonidentity: int
    grid: int

    def __init__(self, source, destination, *, rows=69632):
        from contract262 import checked
        source, destination = tuple(source), tuple(destination)
        proof = checked(source, destination, rows=rows)
        last = max((index + 1 for index, (s, d) in enumerate(zip(source, destination, strict=True))
                    if s != d), default=0)
        grid = next(g for g in (128, 256, 512, 1024, 1792) if g >= last)
        for name, value in (('source', source), ('destination', destination),
                            ('rows', rows), ('nonidentity', proof['nonidentity']), ('grid', grid)):
            object.__setattr__(self, name, value)


def separate(checked, query_slots):
    if type(checked) is not CheckedRows:
        raise ValueError('ROOT304 original complete immutable row proof required')
    if len(query_slots) != 120 or any(type(slot) is not int or not 0 <= slot < checked.rows for slot in query_slots):
        raise ValueError('ROOT304 exact physical120 canonical writers')
    writers = {}
    for row, slot in enumerate(query_slots):
        writers.setdefault(slot, []).append(row)
    aliases = [-1] * 120
    spans = []
    prefix_rows, alias_rows = 0, 0
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
            if (spans and source == spans[-1][0] + spans[-1][2]
                    and destination == spans[-1][1] + spans[-1][2]):
                spans[-1][2] += 1
            else:
                spans.append([source, destination, 1])
            prefix_rows += 1
    if len(spans) > 12 or prefix_rows + alias_rows != checked.nonidentity:
        raise ValueError('ROOT304 exact complete split and at most12 prefix spans')
    # Original destinations are unique. The full anti-alias contract excludes
    # every moved destination from every original source. Removing alias pairs
    # leaves this property intact; alias destinations added as identities are
    # distinct from all retained prefix destinations. No second full scan is
    # needed to prove that the retained prefix copies are snapshot-equivalent.
    return dict(spans=spans + [[0, 0, 0] for _ in range(12 - len(spans))],
                query_alias_destination=aliases, prefix_copies=prefix_rows,
                current_aliases=alias_rows, original_copies=checked.nonidentity,
                span_count=len(spans), legacy_capture_grid=checked.grid)
