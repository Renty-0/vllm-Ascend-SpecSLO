"""HOST prototype: explicit normal cap8, unchanged C96/future96/global192.

No global original class is changed. The actual218 bound repair solver and
its original control/state cells remain on the candidate shape call path.
"""
import ast,copy,inspect,textwrap,types
from pathlib import Path
from budget260 import altered
B=Path(__file__).resolve().parent.parent
_SHAPE=None;_LIMIT=None;_BOUND={};_CONTROLLERS={};_FACTORY=None;_FILTER=None

def eligible(policy,states,normal):
    from split_budget_v132 import PRIORITY
    ordered=sorted(normal,key=lambda i:(PRIORITY[states[i].slo_class],
        max(0.,float(policy.gain(states[i],8))-float(policy.gain(states[i],4))),
        float(states[i].projected_progress_gap(0.)),-i),reverse=True)
    return frozenset([i for i in ordered if policy.gain(states[i],8)>policy.gain(states[i],4)][:12])

def work_limit_class():
    global _LIMIT
    if _LIMIT is not None:return _LIMIT
    import split_budget_v132 as original
    node=ast.parse(textwrap.dedent(inspect.getsource(original.DraftWorkLimit))).body[0]
    field=next(n for n in node.body if isinstance(n,ast.AnnAssign)and ast.unparse(n.target)=='normal_per_request')
    maximum=next(n for n in ast.walk(node)if isinstance(n,ast.keyword)and n.arg=='normal_per_request')
    if field.value.value!=5 or maximum.value.value!=5:raise RuntimeError('ROOT284 exact original normal perowner max5')
    body=altered(node,[(field,'value',ast.Constant(8)),(maximum,'value',ast.Constant(8))],
        'explicit per-normal-owner5to8, every other DTO limit unchanged')
    ns=dict(vars(original));exec(compile(body,__file__+'::private-cap8-DTO','exec'),ns)
    _LIMIT=ns['DraftWorkLimit'];return _LIMIT

def prepared_shape():
    global _SHAPE
    if _SHAPE is not None:return _SHAPE
    import split_budget_v132 as allocation,split_shape_v132 as shaping
    node=ast.parse(textwrap.dedent(inspect.getsource(allocation.shape_draft_work))).body[0]
    loops=[n for n in ast.walk(node)if isinstance(n,ast.For)and ast.unparse(n.iter)=='indices']
    loop=next(n for n in loops if any(isinstance(v,ast.Assign)and ast.unparse(v.targets[0])=='caps[i]'for v in n.body))
    first=next(i for i,v in enumerate(node.body)if v is loop)
    body=list(node.body);body[first:first]=ast.parse("_eligible276=eligible(policy,states,normal)\npolicy.counts['ROOT284_branch_eligible_owners']+=len(_eligible276)").body
    loop_body=list(loop.body);where=next(i for i,v in enumerate(loop_body)if isinstance(v,ast.Assign)and ast.unparse(v.targets[0])=='caps[i]')
    loop_body[where:where]=ast.parse('if i not in eager_set and i not in _eligible276:\n    cap=min(cap,4)').body
    source=altered(node,[(node,'body',body),(loop,'body',loop_body)],'original typed allocator with explicit perowner branch eligibility')
    ns=dict(vars(allocation),eligible=eligible);exec(compile(source,__file__+'::original-typed-allocator','exec'),ns)
    allocator=ns['shape_draft_work']
    node=ast.parse(textwrap.dedent(inspect.getsource(shaping.shape))).body[0]
    limit=next(n for n in ast.walk(node)if isinstance(n,ast.keyword)and n.arg=='normal_per_request')
    edits=[(limit,'value',ast.Constant(8))]
    for statement in node.body:
        if isinstance(statement,ast.AugAssign)and "'normal_cap4_shape_calls'"in ast.unparse(statement):
            edits.append((statement.target,'slice',ast.Constant('ROOT284_normal_cap8_shape_calls')))
        if isinstance(statement,ast.AugAssign)and "'normal_cap4_limited_owners'"in ast.unparse(statement):
            edits.extend([(statement.target,'slice',ast.Constant('ROOT284_private_shape_calls')),(statement,'value',ast.Constant(1))])
    if len(edits)!=4:raise RuntimeError('ROOT284 exact original shape counter anchors')
    source=altered(node,edits,'normal cap8 and truthful counters')
    ns=dict(vars(shaping),DraftWorkLimit=work_limit_class(),shape_draft_work=allocator)
    exec(compile(source,__file__+'::private-cap8-shape','exec'),ns)
    _SHAPE=ns['shape'];return _SHAPE

def prepared_bound_shape(function):
    if function in _BOUND:return _BOUND[function]
    from cost218 import bind_candidate_shape
    original_shape=function.__globals__['shape'];solver=original_shape.__globals__['shape_draft_work']
    if (solver.__code__.co_filename!=bind_candidate_shape.__code__.co_filename
            or solver.__qualname__!=bind_candidate_shape.__qualname__+'.<locals>.solver'
            or set(solver.__code__.co_freevars)!={'control','state','original_solver'}):
        raise RuntimeError('ROOT284 actual original218 instance bound solver')
    prior=dict(zip(solver.__code__.co_freevars,solver.__closure__))
    if prior['control'].cell_contents.candidate_shape is not function:
        raise RuntimeError('ROOT284 actual current control identity')
    template=prepared_shape();allocator=template.__globals__['shape_draft_work']
    def cell(v):return(lambda:v).__closure__[0]
    cells=tuple(cell(allocator)if name=='original_solver'else prior[name]for name in solver.__code__.co_freevars)
    bound=types.FunctionType(solver.__code__,solver.__globals__,solver.__name__,solver.__defaults__,cells)
    value=types.FunctionType(template.__code__,dict(original_shape.__globals__,
        DraftWorkLimit=work_limit_class(),shape_draft_work=bound),template.__name__,template.__defaults__,template.__closure__)
    _BOUND[function]=value;return value

def shape(shaper,**kwargs):
    function=type(shaper).shape;cells=dict(zip(function.__code__.co_freevars,function.__closure__ or ()))
    policy=cells['policy'].cell_contents
    result=prepared_bound_shape(function)(policy,shaper,**kwargs)
    for b in range(1,9):policy.counts[f'ROOT284_actual_normal_B{b}']+=sum(v==b for v in result.normal_budgets.values())
    return result

def controller_function(function):
    global _FACTORY
    if _FACTORY is None:
        import budget260
        source=textwrap.dedent(inspect.getsource(budget260.controller_function))
        if source.count('value == 5')!=1 or source.count('candidate_counts[index] == 5')!=3:
            raise RuntimeError('ROOT284 exact four original group-budget additions')
        source=source.replace('value == 5','value > 4').replace('candidate_counts[index] == 5','candidate_counts[index] > 4')
        ns=dict(vars(budget260),_CONTROLLERS={})
        exec(compile(source,__file__+'::original-controller-factory-allbranches','exec'),ns)
        _FACTORY=ns['controller_function']
    return _FACTORY(function)

def build_plan(controller,*args,**kwargs):
    result=controller_function(type(controller).build_plan)(controller,*args,**kwargs)
    if len(result.target_request_indices)+sum(v>4 for v in result.target_candidate_budgets.values())>24:
        raise RuntimeError('ROOT284 every actual whole-ticket target group budget')
    return result

def filter_eager_parents(indices,controller,payloads,counters):
    global _FILTER
    if _FILTER is None:
        import split_shape_v132 as original
        node=ast.parse(textwrap.dedent(inspect.getsource(original.filter_eager_parents))).body[0]
        ranges=[n for n in ast.walk(node)if isinstance(n,ast.Call)and ast.unparse(n)=='range(1, 6)']
        if len(ranges)!=2:raise RuntimeError('ROOT284 exact original parent-domain checks')
        source=altered(node,[(n,'args',[ast.Constant(1),ast.Constant(9)])for n in ranges],
            'B1..8 recognized; only original qualified B3 remains eligible')
        ns=dict(vars(original));exec(compile(source,__file__+'::same-original-B3-only-eager-filter','exec'),ns)
        _FILTER=ns['filter_eager_parents']
    return _FILTER(indices,controller,payloads,counters)
