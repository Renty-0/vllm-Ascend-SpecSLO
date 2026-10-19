"""Original complete node/hidden/KV checks, explicitly specialized forM128."""
import ast,copy
from pathlib import Path
import neural_oracle231 as original


def cases():
    from oracle255 import cases as previous
    from plan269 import envelope
    result=list(previous())
    for n in range(1,25):
        for label,raw in [('chain6',[6]*n),('chain7',[7]*n),('chain8',[8]*n),('mixed8',[(i%8)+1 for i in range(n)])]:
            budgets=list(raw);adjustments=[]
            while n+sum(b>4 for b in budgets)>24:
                i=next(i for i in reversed(range(n))if budgets[i]>4)
                adjustments.append(dict(owner=i,before=budgets[i],after=4));budgets[i]=4
            while sum(budgets)>96:
                i=next(i for i in reversed(range(n))if budgets[i]>1)
                adjustments.append(dict(owner=i,before=budgets[i],after=budgets[i]-1));budgets[i]-=1
            if max(budgets)<6:continue
            m,s,padding=envelope(n,n+sum(budgets),'full9')
            for anchor in range(9-max(budgets)):
                result.append(dict(name=f'N{n}-{label}-anchor{anchor}',N=n,budgets=budgets,anchor=anchor,M=m,S=s,
                    padding=padding,plan_family='full9',query_count=n+sum(budgets),fixture_budget_reduction=adjustments or None))
    seen={(tuple(c['budgets']),c['anchor'])for c in result}
    for n in (4,8,12,16,20,23):
        maximum=min(n,24-n,(96-4*n)//4)
        for forks in sorted({1,maximum}):
            for budget in (4,5,6,7,8):
                budgets=[budget]*forks+[4]*(n-forks);key=(tuple(budgets),0)
                if key in seen:continue
                seen.add(key);family='full9'if budget>4 else'causal'
                m,s,padding=envelope(n,n+sum(budgets),family)
                result.append(dict(name=f'N{n}-paired-forks{forks}-B{budget}-anchor0',N=n,budgets=budgets,anchor=0,M=m,S=s,
                    padding=padding,plan_family=family,query_count=n+sum(budgets),fixture_budget_reduction=None))
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
            if n.value==80:n.value=120
            elif name=='run'and n.value==16:n.value=24
        elif isinstance(n,ast.ImportFrom):
            n.module={'metadata231':'metadata255','reference_schedule231':'reference_schedule269',
                      'baseline234':'baseline255','branch_tail231':'branch_tail255'}.get(n.module,n.module)
        elif name=='reference'and isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='canonical':
            n.func.id='_checked_reference'
    from metadata255 import checked
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
