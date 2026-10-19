"""Original complete node/hidden/KV checks, explicitly specialized forM160 with independent32-owner banks."""
import ast,copy
from pathlib import Path
import neural_oracle231 as original


def configurations():
    result = []
    for n in (4, 8, 12, 16, 20, 24, 28, 31):
        maximum = min(n, 32-n, (128-4*n)//4, 12)
        for forks in sorted({1, maximum}):
            result.append((n, forks))
    return result


def cases():
    from oracle311 import cases as previous
    from plan326 import envelope
    result = []
    for old in previous():
        case = dict(old)
        case['M'], case['S'], case['padding'] = envelope(case['N'], case['query_count'], case['plan_family'])
        case['ROOT326_original408_fixture_preserved'] = True
        result.append(case)
    for n in range(25,33):
        for label, raw in [('all1',[1]*n),('all4',[4]*n),('fork5',[5]*n),
                           ('chain6',[6]*n),('chain7',[7]*n),('chain8',[8]*n),
                           ('mixed8',[(i%8)+1 for i in range(n)])]:
            budgets = list(raw)
            adjustments = []
            while n+sum(b>4 for b in budgets)>32 or sum(b>4 for b in budgets)>12:
                i = next(i for i in reversed(range(n)) if budgets[i]>4)
                adjustments.append(dict(owner=i,before=budgets[i],after=4))
                budgets[i] = 4
            while sum(budgets)>128:
                i = next(i for i in reversed(range(n)) if budgets[i]>1)
                adjustments.append(dict(owner=i,before=budgets[i],after=budgets[i]-1))
                budgets[i] -= 1
            family = 'full9' if max(budgets)>4 else 'causal'
            m,s,padding = envelope(n,n+sum(budgets),family)
            for anchor in range(9-max(budgets)) if max(budgets)>4 else (0,):
                result.append(dict(name=f'N{n}-{label}-anchor{anchor}',N=n,budgets=budgets,anchor=anchor,
                    M=m,S=s,padding=padding,plan_family=family,query_count=n+sum(budgets),
                    fixture_budget_reduction=adjustments or None,ROOT326_original408_fixture_preserved=False))
    seen = {(tuple(c['budgets']),c['anchor']) for c in result}
    for n,forks in configurations():
        for budget in (4,5,6,7,8):
            budgets = [budget]*forks+[4]*(n-forks)
            key = (tuple(budgets),0)
            if key in seen:
                continue
            seen.add(key)
            family = 'full9' if budget>4 else 'causal'
            m,s,padding = envelope(n,n+sum(budgets),family)
            result.append(dict(name=f'N{n}-paired-forks{forks}-B{budget}-anchor0',N=n,budgets=budgets,
                anchor=0,M=m,S=s,padding=padding,plan_family=family,query_count=n+sum(budgets),
                fixture_budget_reduction=None,ROOT326_original408_fixture_preserved=False))
    return result


CASE_COUNT=len(cases())


def prepare(case,prefixes):
    from oracle255 import prepare as previous
    if max(case['budgets'])<=5:return previous(case,prefixes)
    from plan269 import make
    plans=[make(b,len(prefixes[i]),case['anchor'])for i,b in enumerate(case['budgets'])]
    rows=[[103+17*i+29*j for j in range(b)]for i,b in enumerate(case['budgets'])]
    return plans,rows


def specialized(name):
    source=Path(original.__file__).read_text()
    before=copy.deepcopy(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name==name))
    node=copy.deepcopy(before)
    # These are explicit physical-row and own-bank dimensions. All original
    # numerical matchers, graph/eager checks and commit-prefix checks remain.
    for n in ast.walk(node):
        if isinstance(n,ast.Constant)and type(n.value)is int:
            if n.value==80:n.value=160
            elif name=='run'and n.value==16:n.value=32
        elif isinstance(n,ast.ImportFrom):
            n.module={'metadata231':'metadata326','reference_schedule231':'reference_schedule326',
                      'baseline234':'baseline326','branch_tail231':'branch_tail326'}.get(n.module,n.module)
        elif name=='reference'and isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='canonical':
            n.func.id='_checked_reference'
    from metadata326 import checked
    namespace=dict(vars(original),cases=cases,candidate=candidate,reference=reference,_checked_reference=checked)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__+'::original-'+name+'-with-declared128-domain','exec'),namespace)
    return namespace[name]


def candidate(*args,**kwargs):return specialized('candidate')(*args,**kwargs)
def reference(*args,**kwargs):return specialized('reference')(*args,**kwargs)
def legacy_reference(*args,**kwargs):return reference(*args,**kwargs)


def run(*args,**kwargs):
    bank=args[3]if len(args)>3 else kwargs['bank']
    function=specialized('run')
    function.__globals__['prepare']=lambda case:prepare(case,bank['prefixes'])
    return function(*args,**kwargs)
