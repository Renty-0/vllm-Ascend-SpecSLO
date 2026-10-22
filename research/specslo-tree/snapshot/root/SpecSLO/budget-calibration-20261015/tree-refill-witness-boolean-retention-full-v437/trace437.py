"""Only fix the post-generation witness-retention predicate."""
import ast
import copy
from types import FunctionType
import trace432 as original
_NAMESPACE=None


def trace_source(active_window):
    before=original.trace_source(active_window)
    tree=copy.deepcopy(before)
    rows=[n for n in ast.walk(tree)if isinstance(n,ast.keyword)and n.arg=='ROOT432_refill_witnesses']
    if len(rows)!=1 or ast.unparse(rows[0].value)!="__import__('witness431').snapshot(engine, generation)":
        raise RuntimeError('ROOT437 exact original post-generation witness-retention expression')
    old=rows[0].value
    rows[0].value=ast.parse("__import__('witness437').snapshot(engine,generation)",mode='eval').body
    result=copy.deepcopy(ast.fix_missing_locations(tree))
    rows[0].value=old
    if ast.dump(tree,include_attributes=False)!=ast.dump(before,include_attributes=False):
        raise RuntimeError('ROOT437 entire original trace AST inverse')
    return result


def namespace():
    global _NAMESPACE
    if _NAMESPACE is not None:return _NAMESPACE
    parent=original.namespace()
    ns=dict(parent,trace_source=trace_source)
    for name in('make_trace','warm_to_fixedpoint','run_windows'):
        fn=parent[name]
        actual=FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)
        actual.__kwdefaults__=fn.__kwdefaults__
        ns[name]=actual
    _NAMESPACE=ns
    return ns


def warm_to_fixedpoint(*args,**kwargs):return namespace()['warm_to_fixedpoint'](*args,**kwargs)
def run_windows(*args,**kwargs):return namespace()['run_windows'](*args,**kwargs)
