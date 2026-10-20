"""Finite new S32 cases and genuine independently prefilled prefix families."""
def prefix_bank(original, extent):
    if len(original) != 32 or extent not in (512, 1024, 1536, 2048, 2560):
        raise ValueError('all32 own prefix owners and explicit extent family')
    result = []
    for row in original:
        residue = len(row) % 128
        limit = ((extent-5-residue)//128)*128+residue
        count = min(len(row), limit)
        if count <= 0:
            raise ValueError('genuine nonempty prefix preserving its page residue')
        result.append(list(row[:count]))
    if [len(v)%128 for v in result] != [len(v)%128 for v in original]:
        raise ValueError('all original logical page residues retained')
    if ((max(len(v)+4 for v in result)+511)//512)*512 != extent:
        raise ValueError('fixture must exercise the declared actual KV bucket')
    return result


def cases():
    from mapper399 import envelope
    result = []
    for n in range(25, 33):
        patterns = (
            ('all1', [1]*n),
            ('all2', [2]*n),
            ('saturated', [3]*24+[2]*(n-24)),
            ('reversed', [2]*(n-24)+[3]*24),
            ('mixed', [(i%3)+1 for i in range(n)]),
        )
        for label, budgets in patterns:
            m, s, padding = envelope(n, n+sum(budgets), 'causal')
            result.append(dict(name=f'N{n}-{label}', N=n, budgets=budgets, anchor=0,
                               M=m, S=s, padding=padding, plan_family='causal',
                               query_count=n+sum(budgets), fixture_budget_reduction=None))
    return result
