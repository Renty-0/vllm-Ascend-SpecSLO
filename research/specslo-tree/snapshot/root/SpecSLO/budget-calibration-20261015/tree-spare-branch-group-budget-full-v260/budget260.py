"""Spend only spare canonical q5 groups on positive B5 marginal gains.

Original scalar gains, minimum needs, priority, whole grants, C96/global192,
ticket publication and selection remain in the reused allocator. No installs
occur at import; these delegates are used only by the new LIVE projection.
"""
import ast,copy,inspect,textwrap
from pathlib import Path
B=Path(__file__).resolve().parent.parent
_SHAPE=None;_CONTROLLERS={}

def eligible(policy,states,normal):
    from split_budget_v132 import PRIORITY
    slots=min(len(normal),max(0,24-len(normal)))
    ordered=sorted(normal,key=lambda i:(
        PRIORITY[states[i].slo_class],
        max(0.,float(policy.gain(states[i],5))-float(policy.gain(states[i],4))),
        float(states[i].projected_progress_gap(0.)),-i),reverse=True)
    return frozenset(i for i in ordered if policy.gain(states[i],5)>policy.gain(states[i],4))if slots>=len(normal)else frozenset(
        [i for i in ordered if policy.gain(states[i],5)>policy.gain(states[i],4)][:slots])

def altered(node,edits,label):
    """Each edit is recorded on the actual AST; reverse restores it fully."""
    original=ast.dump(node,include_attributes=False);changes=[]
    for parent,attribute,value in edits:
        changes.append((parent,attribute,copy.deepcopy(getattr(parent,attribute))))
        setattr(parent,attribute,value)
    value=ast.unparse(ast.fix_missing_locations(node))+'\n'
    for parent,attribute,old in reversed(changes):setattr(parent,attribute,old)
    if ast.dump(node,include_attributes=False)!=original:raise RuntimeError(label+' complete original AST inverse')
    return value

def prepared_shape():
    global _SHAPE
    if _SHAPE is not None:return _SHAPE
    import split_budget_v132 as allocation,split_shape_v132 as shaping
    if Path(allocation.__file__).resolve()!=B/'tree-high-normal-cap4-native-families-v148/split_budget_v132.py':
        raise RuntimeError('exact original typed allocator source')
    if Path(shaping.__file__).resolve()!=B/'tree-high-normal-cap4-native-families-v148/split_shape_v132.py':
        raise RuntimeError('exact original shape wrapper source')
    node=ast.parse(textwrap.dedent(inspect.getsource(allocation.shape_draft_work))).body[0]
    loops=[n for n in ast.walk(node)if isinstance(n,ast.For)and ast.unparse(n.iter)=='indices']
    loop=next(n for n in loops if any(isinstance(v,ast.Assign)and ast.unparse(v.targets[0])=='caps[i]'for v in n.body))
    cap_assign=next(i for i,v in enumerate(loop.body)if isinstance(v,ast.Assign)and ast.unparse(v.targets[0])=='caps[i]')
    first=next(i for i,v in enumerate(node.body)if v is loop)
    new_body=list(node.body);new_body[first:first]=ast.parse("_root260_eligible = eligible(policy, states, normal)\npolicy.counts['ROOT260_B5_eligible_owners'] += len(_root260_eligible)").body
    new_loop=list(loop.body);new_loop[cap_assign:cap_assign]=ast.parse("if i not in eager_set and i not in _root260_eligible:\n    cap = min(cap, 4)").body
    body=altered(node,[(node,'body',new_body),(loop,'body',new_loop)],'capacity-aware individual caps')
    ns=dict(vars(allocation),eligible=eligible);exec(compile(body,__file__+'::original-allocator-with-explicit-group-caps','exec'),ns)
    allocator=ns['shape_draft_work']
    node=ast.parse(textwrap.dedent(inspect.getsource(shaping.shape))).body[0];edits=[]
    limit=next(n for n in ast.walk(node)if isinstance(n,ast.keyword)and n.arg=='normal_per_request')
    edits.append((limit,'value',ast.Constant(5)))
    for statement in node.body:
        if isinstance(statement,ast.AugAssign)and "'normal_cap4_shape_calls'"in ast.unparse(statement):
            edits.append((statement.target,'slice',ast.Constant('ROOT260_group_aware_shape_calls')))
        if isinstance(statement,ast.AugAssign)and "'normal_cap4_limited_owners'"in ast.unparse(statement):
            edits.append((statement.target,'slice',ast.Constant('ROOT260_original_N_gt8_cap_relaxed_calls')))
            edits.append((statement,'value',ast.Constant(1)))
    if len(edits)!=4:raise RuntimeError('one explicit cap and truthful group-policy counters')
    body=altered(node,edits,'shape cap and counters')
    ns=dict(vars(shaping),shape_draft_work=allocator);exec(compile(body,__file__+'::original-wrapper-group-policy','exec'),ns)
    _SHAPE=ns['shape'];return _SHAPE

def shape(shaper,**kwargs):
    function=type(shaper).shape
    cells=dict(zip(function.__code__.co_freevars,function.__closure__ or()))
    if 'policy'not in cells:raise RuntimeError('same original shaper policy closure')
    policy=cells['policy'].cell_contents
    result=prepared_shape()(policy,shaper,**kwargs)
    policy.counts['ROOT260_actual_B5_grants']+=sum(value==5 for value in result.normal_budgets.values())
    if len(result.requested_normal)>8:
        policy.counts['ROOT260_actual_B5_grants_N_gt8']+=sum(value==5 for value in result.normal_budgets.values())
    return result

def controller_function(function):
    if function in _CONTROLLERS:return _CONTROLLERS[function]
    expected=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/spec_rhythm.py')
    if Path(function.__code__.co_filename).resolve()!=expected:raise RuntimeError('same original controller function source')
    node=ast.parse(textwrap.dedent(inspect.getsource(function))).body[0];edits=[]
    constrained=next(n for n in node.body if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='constrained')
    extra=ast.parse('sum(1 + int(value == 5) for value in candidate_counts.values()) > 24',mode='eval').body
    edits.append((constrained,'value',ast.BoolOp(op=ast.Or(),values=[copy.deepcopy(constrained.value),extra])))
    selected_tokens=next(n for n in ast.walk(node)if isinstance(n,ast.Assign)and ast.unparse(n.targets[0])=='selected_tokens')
    # The declaration and loop belong to the same constrained selection arm.
    arm=next(n for n in ast.walk(node)if isinstance(n,ast.If)and selected_tokens in n.body)
    body=list(arm.body);where=body.index(selected_tokens);body[where+1:where+1]=ast.parse('_root260_groups = 0').body
    edits.append((arm,'body',body))
    loop=next(n for n in arm.body if isinstance(n,ast.For)and ast.unparse(n.iter)=='ordered')
    body=list(loop.body);where=next(i for i,n in enumerate(body)if isinstance(n,ast.Expr)and ast.unparse(n)=='selected.append(index)')
    body[where:where]=ast.parse('if _root260_groups + 1 + int(candidate_counts[index] == 5) > 24:\n    continue').body
    body.extend(ast.parse('_root260_groups += 1 + int(candidate_counts[index] == 5)').body);edits.append((loop,'body',body))
    reserved=next(n for n in node.body if isinstance(n,ast.If)and ast.unparse(n.test)=='reserved_target is not None'and any(isinstance(v,ast.Assign)and ast.unparse(v.targets[0])=='reserved_target_set'for v in n.body))
    body=ast.parse("if sum(1 + int(candidate_counts[index] == 5) for index in reserved_target) > 24:\n    raise ValueError('ROOT260 irreversible target reservation exceeds24 physical groups')").body+list(reserved.body)
    edits.append((reserved,'body',body))
    code=altered(node,edits,'whole-ticket target group budget')
    ns=dict(function.__globals__);exec(compile('from __future__ import annotations\n'+code,__file__+'::original-controller-whole-group-budget','exec'),ns)
    result=ns[node.name]
    if inspect.signature(result)!=inspect.signature(function):raise RuntimeError('all original controller defaults retained')
    _CONTROLLERS[function]=result;return result

def build_plan(controller,*args,**kwargs):
    result=controller_function(type(controller).build_plan)(controller,*args,**kwargs)
    selected=result.target_request_indices
    if len(selected)+sum(value==5 for value in result.target_candidate_budgets.values())>24:
        raise RuntimeError('actual target tickets exceed canonical24 group budget')
    return result
