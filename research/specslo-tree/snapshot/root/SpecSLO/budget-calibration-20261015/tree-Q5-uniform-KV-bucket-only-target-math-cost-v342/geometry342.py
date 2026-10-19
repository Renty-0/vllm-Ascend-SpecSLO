"""Same Q5/M120; compare three explicit attention families per cohort."""
FAMILIES = ('FULL2560', 'FULLbucket')


def cohorts():
    result = []
    for n in (1, 4, 12, 24):
        for origin in (0, 8, 16):
            result.append(dict(name=f'N{n}-origin{origin}', N=n,
                owners=[(origin+i)%32 for i in range(n)]))
    result.append(dict(name='N24-allshort', N=24, owners=list(range(2,24))+[26,27]))
    result.append(dict(name='N1-last-prefix', N=1, owners=[31]))
    return result


def cases():
    return [dict(cohort, family=family, name=cohort['name']+'/'+family,
                 M=120, groups=24, width=5, budgets=[4]*cohort['N'],
                 query_count=5*cohort['N']) for cohort in cohorts() for family in FAMILIES]


def extent(case, prefixes):
    last = max(len(prefixes[owner])+5 for owner in case['owners'])
    value = ((last+511)//512)*512
    if not 0 < value <= 2560:
        raise RuntimeError('all real visible contexts must fit selected uniform KV bucket')
    return 2560 if case['family']=='FULL2560' else value


def arms():
    result=[]
    for label, owners in [('mixed',list(range(24))), ('short',list(range(2,24))+[26,27])]:
        for family in FAMILIES:
            result.append(dict(name=label+'/'+family, family=family, N=24, owners=owners))
    return result
