"""Choose physical padding from unchanged real budgets; never drop a query."""
import mapper255 as original
from source377 import source_module, function

_small, _small_source = source_module(original, 'mapper255.py', {'envelope'}, {120: 80})


def choose(budgets, *, enabled):
    if (type(enabled) is not bool or not budgets or len(budgets) > 24
            or any(type(b) is not int or not 1 <= b <= 5 for b in budgets)
            or sum(budgets) > 96):
        raise ValueError('ROOT377 original actual N24/C96/B1..5 limits')
    groups = len(budgets) + budgets.count(5)
    queries = len(budgets) + sum(budgets)
    if groups > 24 or queries > 120:
        raise ValueError('ROOT377 complete real primary and side groups must fit')
    return 80 if enabled and groups <= 16 and queries <= 80 else 120


def physical(engine, plans):
    return choose([p.candidate_budget for p in plans],
                  enabled=engine._branch_tail_state.adaptive377_enabled)


def envelope(n, q, family, *, physical):
    if physical not in (80, 120) or physical == 80 and n > 16:
        raise ValueError('ROOT377 selected complete physical family')
    return (_small['envelope'] if physical == 80 else original.envelope)(n, q, family)


def adjust(case, *, enabled):
    m = choose(case['budgets'], enabled=enabled)
    selected, s, padding = envelope(case['N'], case['query_count'], case['plan_family'], physical=m)
    return dict(case, M=selected, S=s, padding=padding)


def pad_function():
    import padding255
    return function(padding255.pad, (
        ('from mapper255 import envelope', 'from geometry377 import envelope, physical'),
        ('m,s,lengths=envelope(n,sum(p.candidate_budget+1 for p in plans),plan_family(plans))',
         'm,s,lengths=envelope(n,sum(p.candidate_budget+1 for p in plans),plan_family(plans),physical=physical(engine,plans))'),
        ('sum(p.candidate_budget+1 for p in pp)!=120', 'sum(p.candidate_budget+1 for p in pp)!=m'),
        ('all actual physical120 rows accounted', 'all actual selected physical rows accounted'),
    ))
