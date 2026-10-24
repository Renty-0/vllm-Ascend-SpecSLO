"""Lossless HOST compression of the existing complete1792 copy map."""
CAPACITY = 64


def pack(source, destination, *, rows=69632):
    from contract262 import checked
    checked(source, destination, rows=rows)
    spans = []
    for src, dst in zip(source, destination, strict=True):
        if src == dst:
            continue
        if spans and src == spans[-1][0] + spans[-1][2] and dst == spans[-1][1] + spans[-1][2]:
            spans[-1][2] += 1
        else:
            spans.append([src, dst, 1])
    if len(spans) > CAPACITY:
        raise ValueError('Complete real copy spans must fit; no truncation')
    expected = [(s, d) for s, d in zip(source, destination, strict=True) if s != d]
    actual = [(s + i, d + i) for s, d, n in spans for i in range(n)]
    if actual != expected:
        raise RuntimeError('Lossless source/destination row-pair inverse')
    return dict(spans=spans + [[0, 0, 0] for _ in range(CAPACITY - len(spans))],
                span_count=len(spans), copied_rows=len(actual))
