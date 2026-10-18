"""Original node oracle, explicit supported shapes and static own reference."""
import ast,copy,types
import neural_oracle231 as original

def cases():
    return [c for c in original.cases()if c['N']+c['budgets'].count(5)<=16]

def reference(*args,**kwargs):
    from pathlib import Path
    from static_reference244 import checked
    tree=ast.parse(Path(original.__file__).read_text())
    node=copy.deepcopy(next(x for x in tree.body if isinstance(x,ast.FunctionDef)and x.name=='reference'))
    before=copy.deepcopy(node);changes=[]
    for item in ast.walk(node):
        if isinstance(item,ast.ImportFrom)and item.module=='metadata231':
            changes.append((item,'module',item.module));item.module='static_reference244'
        if isinstance(item,ast.Call)and isinstance(item.func,ast.Name)and item.func.id=='canonical':
            changes.append((item.func,'id',item.func.id));item.func.id='_static_reference244_checked'
    if len(changes)!=2:raise RuntimeError('exact two independent reference specialization anchors')
    specialized=copy.deepcopy(node)
    for item,key,value in changes:setattr(item,key,value)
    if ast.dump(node,include_attributes=False)!=ast.dump(before,include_attributes=False):raise RuntimeError('complete original reference inverse AST')
    namespace=dict(vars(original),_static_reference244_checked=checked)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[specialized],type_ignores=[])),__file__+'::own-static-reference','exec'),namespace)
    return namespace['reference'](*args,**kwargs)

def run(*args,**kwargs):
    if len(cases())!=111:raise RuntimeError('explicit111 supported positive cases; other48 retained as unsupported')
    fn=types.FunctionType(original.run.__code__,dict(vars(original),cases=cases,reference=reference),
                          original.run.__name__,original.run.__defaults__,original.run.__closure__)
    fn.__kwdefaults__=None if original.run.__kwdefaults__ is None else dict(original.run.__kwdefaults__)
    return fn(*args,**kwargs)
