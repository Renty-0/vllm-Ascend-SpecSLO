"""Original complete node/hidden/KV checks, explicitly specialized forM128."""
import ast,copy
from pathlib import Path
import neural_oracle231 as original


def cases():
    from mapper249 import envelope
    result=[]
    for n in range(1,25):
        for label,raw in [('all1',[1]*n),('all4',[4]*n),('fork5',[5]*n),('mixed',[(i%5)+1 for i in range(n)])]:
            budgets=list(raw);adjustments=[]
            while n+budgets.count(5)>25 or sum(budgets)>96:
                index=next(i for i in reversed(range(n))if budgets[i]==5)if n+budgets.count(5)>25 else next(i for i in reversed(range(n))if budgets[i]>1)
                adjustments.append(dict(owner=index,before=budgets[index],after=budgets[index]-1));budgets[index]-=1
            family='full9'if 5 in budgets else'causal';m,s,padding=envelope(n,n+sum(budgets),family)
            for anchor in(range(4)if 5 in budgets else(0,)):
                result.append(dict(name=f'N{n}-{label}-anchor{anchor}',N=n,budgets=budgets,anchor=anchor,M=m,S=s,
                    padding=padding,plan_family=family,query_count=n+sum(budgets),fixture_budget_reduction=adjustments or None))
    for label,budgets in [('rootonly',[4]*23+[1]),('fork-rootonly',[5]+[4]*21+[1,3])]:
        family='full9'if 5 in budgets else'causal';m,s,padding=envelope(24,24+sum(budgets),family)
        for anchor in(range(4)if 5 in budgets else(0,)):
            result.append(dict(name=f'N24-{label}-anchor{anchor}',N=24,budgets=budgets,anchor=anchor,M=m,S=s,
                padding=padding,plan_family=family,query_count=24+sum(budgets),fixture_budget_reduction=None))
    return result


CASE_COUNT=len(cases())


def prepare(case,prefixes):
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan,pack_selected_tree_plan
    plans=[];rows=[]
    for i,b in enumerate(case['budgets']):
        raw=cached_cpu_tree_speculation_plan(2,4,len(prefixes[i]),4096,candidate_budget=8)
        chosen=list(range(min(b,4)))+([4+case['anchor']]if b==5 else[])
        plans.append(pack_selected_tree_plan(raw,chosen));rows.append([103+17*i+29*j for j in chosen])
    return plans,rows


def specialized(name):
    source=Path(original.__file__).read_text()
    before=copy.deepcopy(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name==name))
    node=copy.deepcopy(before)
    # These are explicit physical-row and own-bank dimensions. All original
    # numerical matchers, graph/eager checks and commit-prefix checks remain.
    for n in ast.walk(node):
        if isinstance(n,ast.Constant)and type(n.value)is int:
            if n.value==80:n.value=128
            elif name=='run'and n.value==16:n.value=24
        elif isinstance(n,ast.ImportFrom):
            n.module={'metadata231':'metadata251','reference_schedule231':'reference_schedule251',
                      'baseline234':'baseline251','branch_tail231':'branch_tail249'}.get(n.module,n.module)
        elif name=='reference'and isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='canonical':
            n.func.id='_checked_reference'
    from metadata251 import checked
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
