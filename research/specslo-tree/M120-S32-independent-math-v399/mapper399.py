"""Finite numerical component envelope; no production scheduling permission."""
def envelope(n, q, family):
    if type(n) is not int or type(q) is not int or family != 'causal':
        raise ValueError('finite linear B1..3 target family required')
    if not 1 <= n <= 32 or not 2*n <= q <= min(4*n, 120) or q-n > 88:
        raise ValueError('finite N32/C88/M120 geometry')
    remaining = 120-q
    padding = []
    while remaining:
        value = min(5, remaining)
        padding.append(value)
        remaining -= value
    return 120, n+len(padding), padding
