"""HOST description of a finite linear target diagnostic, not a new scheduler."""
LAYOUTS = {'Q4-S24': (4, 24), 'Q4-S32': (4, 32), 'Q5-S24-control': (5, 24)}


def cases():
    result = []
    for layout, (width, groups) in LAYOUTS.items():
        sizes = (1, 12, 24) if width == 5 else ((1, 4, 12, 24, 32) if groups == 32 else (1, 4, 12, 24))
        for n in sizes:
            for origin in (0, 8, 16):
                result.append(dict(name=f'{layout}-N{n}-origin{origin}', layout=layout,
                    width=width, groups=groups, M=width*groups, N=n, origin=origin,
                    owners=[(origin+i) % 32 for i in range(n)],
                    budgets=[width-1]*n, query_count=width*n))
    return result


def cost_arms():
    return tuple(dict(name=name+'-N24', layout=name, N=24) for name in LAYOUTS) + (
        dict(name='Q4-S32-N32', layout='Q4-S32', N=32),)


def teacher_rows(owners, width):
    return [[103+17*owner+29*j for j in range(width-1)] for owner in owners]


def make_plans(prefixes, owners, width):
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan
    return [cached_cpu_tree_speculation_plan(1, width-1, len(prefixes[owner]), 4096,
            candidate_budget=width-1) for owner in owners]
