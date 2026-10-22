"""New same-physical reference protocol; old cross-M120 failures stay intact."""
import ast,copy
from pathlib import Path
import neural_oracle231 as oldest
import oracle255
import oracle377

cases=oracle377.cases
candidate=oracle377.candidate

def reference_function():
    source=Path(oldest.__file__).read_text()
    node=copy.deepcopy(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='reference'))
    before=copy.deepcopy(node)
    reversals=[];counts=dict(imports=0,schedule=0,physical_report=0)
    changes=dict(metadata231='metadata459',query1_reference231='metadata459',reference_schedule231='reference_schedule459')
    for item in ast.walk(node):
        if isinstance(item,ast.ImportFrom)and item.module in changes:
            reversals.append((item,'module',item.module));counts['imports']+=1
            item.module=changes[item.module]
            if any(alias.name=='canonical'for alias in item.names):
                for alias in item.names:
                    if alias.name=='canonical':
                        reversals.extend(((alias,'name',alias.name),(alias,'asname',alias.asname)))
                        alias.name='checked';alias.asname='canonical'
        if isinstance(item,ast.Call)and isinstance(item.func,ast.Name)and item.func.id=='schedule':
            reversals.append((item,'keywords',copy.deepcopy(item.keywords)));counts['schedule']+=1
            item.keywords.append(ast.keyword(arg='physical_M',value=ast.Subscript(value=ast.Name(id='case',ctx=ast.Load()),slice=ast.Constant('M'),ctx=ast.Load())))
        if isinstance(item,ast.keyword)and item.arg=='physical_queries':
            reversals.append((item,'value',copy.deepcopy(item.value)));counts['physical_report']+=1
            item.value=ast.Subscript(value=ast.Name(id='case',ctx=ast.Load()),slice=ast.Constant('M'),ctx=ast.Load())
    if counts!=dict(imports=3,schedule=1,physical_report=1):
        raise RuntimeError('ROOT459 exact independent reference source anchors')
    projected=copy.deepcopy(node)
    for item,attribute,value in reversed(reversals):setattr(item,attribute,value)
    if ast.dump(node,include_attributes=False)!=ast.dump(before,include_attributes=False):
        raise RuntimeError('ROOT459 entire independent reference body inverse')
    ns=dict(vars(oldest))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[projected],type_ignores=[])),__file__+'::independent-own-physical-reference','exec'),ns)
    return ns['reference']

def reference(*args,**kwargs):
    return reference_function()(*args,**kwargs)

def specialized_run():
    original=oracle377.specialized_run()
    # All unchanged greedy, Native-tolerance and added bitwise predicates stay.
    # Only the mathematical reference's physical frame and draft's matching
    # no-model stage enumeration are explicitly bound to the selected case M.
    # The original377 helper compiles a function whose source is not a disk
    # function. Its code can be rebound without changing any comparator.
    import types
    ns=dict(original.__globals__,cases=cases,candidate=candidate,reference=reference)
    return types.FunctionType(original.__code__,ns,original.__name__,original.__defaults__,original.__closure__)

def run(engine,native,stage,bank,output):
    body=specialized_run()
    body.__globals__['prepare']=lambda case:oracle255.prepare(case,bank['prefixes'])
    return body(engine,native,stage,bank,output=output)
