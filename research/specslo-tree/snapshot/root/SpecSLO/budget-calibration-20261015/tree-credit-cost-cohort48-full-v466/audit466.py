"""Original full numerical audit with explicit window and cohort bindings."""
import ast
import copy
import inspect
from types import FunctionType
import audit463 as original


def audit_trace(trace):
    function = original.audit_trace
    namespace = dict(function.__globals__, MAX_NEW=48)
    return FunctionType(function.__code__, namespace, function.__name__,
                        function.__defaults__, function.__closure__)(trace)


def constructed():
    tree = ast.parse(inspect.getsource(original.audit))
    before = ast.dump(tree, include_attributes=False)
    node = tree.body[0]
    if [arg.arg for arg in node.args.args] != ['directory']:
        raise RuntimeError('Exact original audit interface required')
    node.args.args.append(ast.arg(arg='windows'))
    calls = [item for item in ast.walk(node) if isinstance(item, ast.Call)
             and ast.unparse(item.func) == 'original.audit']
    if len(calls) != 1 or [ast.unparse(a) for a in calls[0].args] != ['directory']:
        raise RuntimeError('Exact original parent invocation required')
    calls[0].args.append(ast.Name(id='windows', ctx=ast.Load()))
    changed = copy.deepcopy(tree)
    node.args.args.pop()
    calls[0].args.pop()
    if ast.dump(tree, include_attributes=False) != before:
        raise RuntimeError('Complete numerical audit body inverse required')
    namespace = dict(original.audit.__globals__, audit_trace=audit_trace)
    exec(compile(ast.fix_missing_locations(changed), __file__+'::window-interface', 'exec'), namespace)
    return namespace['audit']


def audit(directory, windows):
    if tuple(windows) != (64,):
        raise ValueError('Actual global64 experiment required')
    return constructed()(directory, windows)
