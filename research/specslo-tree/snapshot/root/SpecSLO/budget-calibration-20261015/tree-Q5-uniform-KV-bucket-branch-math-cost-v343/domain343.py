"""Original complete408 plus a short-context subset preserving page residues."""
from oracle343 import cases
from audit_static_calls311 import expected_grid


def family(case):
    return (11 if any(b>4 for b in case['budgets'])else 10,expected_grid(case))


def prelude():
    selected=[];seen=set()
    for case in cases():
        key=family(case)
        if key not in seen:
            selected.append(case);seen.add(key)
    if len(selected)!=6:
        raise RuntimeError('all6 original copy families needed at maximum extent')
    return selected


def short_cases():
    selected=[case for case in cases()if case['N']in(1,2,4,8,12,24)]
    if len(selected)!=90 or {family(c)for c in selected}!={family(c)for c in cases()}:
        raise RuntimeError('all90 short-context cases and six families required')
    return selected


def short_prefixes(prefixes):
    result=[list(row[:len(row)%128 or 128])for row in prefixes]
    if len(result)!=24 or any(not 0<len(row)<=128 for row in result):
        raise RuntimeError('all24 short teacher prefixes')
    if [len(x)%128 for x in result]!=[len(x)%128 for x in prefixes]:
        raise RuntimeError('all original copy-grid/page-tail residues retained')
    return result


def mid_prefixes(prefixes):
    result=[list(row) for row in prefixes]
    result[0]=result[0][:609]
    result[1]=result[1][:1659]
    if [len(x)%128 for x in result]!=[len(x)%128 for x in prefixes]:
        raise RuntimeError('mid-context original copy residues required')
    return result


TOTAL_CASES=6+408+90+90
