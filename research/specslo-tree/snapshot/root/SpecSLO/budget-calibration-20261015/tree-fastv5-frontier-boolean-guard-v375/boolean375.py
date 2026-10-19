"""One source-bound Boolean contract repair; no numerical/operator change."""
import ast
import copy
from pathlib import Path
import forward373

def install():
    source=Path(forward373.__file__).read_text()
    cls=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef)and n.name=='State')
    node=copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='frontier'))
    original=copy.deepcopy(node)
    matches=[n for n in ast.walk(node)if isinstance(n,ast.BoolOp)and isinstance(n.op,ast.And)
             and len(n.values)==2 and isinstance(n.values[-1],ast.Name)and n.values[-1].id=='frontiers']
    if len(matches)!=1:raise RuntimeError('ROOT375 one exact Boolean guard')
    expression=matches[0];old=copy.deepcopy(expression.values[-1])
    expression.values[-1]=ast.Call(func=ast.Name(id='bool',ctx=ast.Load()),args=[copy.deepcopy(old)],keywords=[])
    projected=copy.deepcopy(node)
    expression.values[-1]=old
    if ast.dump(node,include_attributes=False)!=ast.dump(original,include_attributes=False):raise RuntimeError('ROOT375 full method AST inverse')
    namespace=dict(vars(forward373))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[projected],type_ignores=[])),__file__+'::Boolean-only-frontier-guard','exec'),namespace)
    forward373.State.frontier=namespace['frontier']
