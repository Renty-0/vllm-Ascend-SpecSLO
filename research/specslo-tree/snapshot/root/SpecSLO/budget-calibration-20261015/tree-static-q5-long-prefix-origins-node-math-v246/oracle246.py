"""Original node oracle, explicit supported shapes and static own reference."""
import ast,copy,types
import neural_oracle231 as original

def cases():
    return [c for c in original.cases()if c['N']+c['budgets'].count(5)<=16]

def prepare_at_prefixes(case,prefixes):
    from vllm_ascend.spec_decode.pearl.tree import cached_cpu_tree_speculation_plan,pack_selected_tree_plan
    plans=[];rows=[]
    for i,b in enumerate(case['budgets']):
        raw=cached_cpu_tree_speculation_plan(2,4,len(prefixes[i]),4096,candidate_budget=8)
        chosen=list(range(min(b,4)))+([4+case['anchor']]if b==5 else[])
        plans.append(pack_selected_tree_plan(raw,chosen))
        rows.append([103+17*i+29*node for node in chosen])
    return plans,rows

def reference(*args,**kwargs):
    from pathlib import Path
    from static_reference246 import checked
    tree=ast.parse(Path(original.__file__).read_text())
    node=copy.deepcopy(next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='reference'))
    before=copy.deepcopy(node);changes=[]
    for item in ast.walk(node):
        if isinstance(item,ast.ImportFrom)and item.module=='metadata231':
            changes.append((item,'module',item.module));item.module='static_reference246'
        if isinstance(item,ast.ImportFrom)and item.module=='reference_schedule231':
            changes.append((item,'module',item.module));item.module='reference_schedule246'
        if isinstance(item,ast.Call)and isinstance(item.func,ast.Name)and item.func.id=='canonical':
            changes.append((item.func,'id',item.func.id));item.func.id='_static_reference246_checked'
    if len(changes)!=3:raise RuntimeError('exact three independent reference specialization anchors')
    specialized=copy.deepcopy(node)
    for item,key,value in changes:setattr(item,key,value)
    if ast.dump(node,include_attributes=False)!=ast.dump(before,include_attributes=False):raise RuntimeError('complete original reference inverse AST')
    namespace=dict(vars(original),_static_reference246_checked=checked)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[specialized],type_ignores=[])),__file__+'::own-static-reference','exec'),namespace)
    return namespace['reference'](*args,**kwargs)

def legacy_reference(*args,**kwargs):
    from pathlib import Path
    node=copy.deepcopy(next(x for x in ast.parse(Path(original.__file__).read_text()).body if isinstance(x,ast.FunctionDef)and x.name=='reference'))
    changed=0
    for item in ast.walk(node):
        if isinstance(item,ast.ImportFrom)and item.module=='reference_schedule231':
            item.module='reference_schedule246';changed+=1
    if changed!=1:raise RuntimeError('one actual-prefix scheduler for legacy negative control')
    namespace=dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__+'::legacy-own-reference','exec'),namespace)
    return namespace['reference'](*args,**kwargs)

def run(*args,**kwargs):
    if len(cases())!=111:raise RuntimeError('explicit111 supported positive cases; other48 retained as unsupported')
    bank=args[3] if len(args)>3 else kwargs['bank']
    def actual_prepare(case):
        return prepare_at_prefixes(case,bank['prefixes'])
    from pathlib import Path
    node=copy.deepcopy(next(x for x in ast.parse(Path(original.__file__).read_text()).body if isinstance(x,ast.FunctionDef)and x.name=='run'))
    before=copy.deepcopy(node)
    changes=[]
    for item in ast.walk(node):
        if isinstance(item,ast.ImportFrom)and item.module in ('reference_schedule231','baseline234'):
            changes.append((item,item.module));item.module={'reference_schedule231':'reference_schedule246','baseline234':'baseline246'}[item.module]
    if len(changes)!=3:raise RuntimeError('three original runner dependency imports only')
    specialized=copy.deepcopy(node)
    for item,name in changes:item.module=name
    if ast.dump(node,include_attributes=False)!=ast.dump(before,include_attributes=False):
        raise RuntimeError('complete original comparison and prefix-protection runner AST inverse')
    namespace=dict(vars(original),cases=cases,reference=reference,prepare=actual_prepare)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[specialized],type_ignores=[])),__file__+'::original-run-actual-prefix-imports','exec'),namespace)
    return namespace['run'](*args,**kwargs)
