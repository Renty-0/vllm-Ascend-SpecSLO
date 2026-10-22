"""Original trace functions; declare new admission and retain actual witnesses."""
import ast
import copy
from types import FunctionType
import trace425 as original
from generation425 import patch as original_generation_patch
from projection431 import patch as refill_patch
from scope432 import report as standalone_report

_NAMESPACE = None


def generation(source):
    return refill_patch(original_generation_patch(source))


def trace_source(active_window):
    from window425 import trace_source as original_source
    before = original_source(active_window)
    tree = copy.deepcopy(before)
    selected = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == 'dict'
                and any(k.arg == 'source_generation_before' for k in n.keywords)]
    if len(selected) != 1:
        raise RuntimeError('ROOT432 exact original saved trace record')
    addition = ast.keyword(arg='ROOT432_refill_witnesses',
        value=ast.parse("__import__('witness431').snapshot(engine,generation)", mode='eval').body)
    selected[0].keywords.append(addition)
    result = copy.deepcopy(ast.fix_missing_locations(tree))
    selected[0].keywords.remove(addition)
    if ast.dump(tree,include_attributes=False) != ast.dump(before,include_attributes=False):
        raise RuntimeError('ROOT432 complete original trace AST inverse')
    return result


def namespace():
    global _NAMESPACE
    if _NAMESPACE is not None:
        return _NAMESPACE
    parent = original.namespace()
    ns = dict(parent, canonical_patch=generation, trace_source=trace_source,
              standalone_report=standalone_report)
    for name in ('make_trace','warm_to_fixedpoint','run_windows'):
        function = parent[name]
        actual = FunctionType(function.__code__,ns,function.__name__,function.__defaults__,function.__closure__)
        actual.__kwdefaults__ = function.__kwdefaults__
        ns[name] = actual
    _NAMESPACE = ns
    return ns


def warm_to_fixedpoint(*args,**kwargs):
    return namespace()['warm_to_fixedpoint'](*args,**kwargs)


def run_windows(*args,**kwargs):
    return namespace()['run_windows'](*args,**kwargs)
