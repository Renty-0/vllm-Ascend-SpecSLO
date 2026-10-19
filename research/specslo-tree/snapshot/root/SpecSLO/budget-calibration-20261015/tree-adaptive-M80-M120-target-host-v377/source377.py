"""Private, reversible dimensional specialization; original modules stay intact."""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path

B = Path(__file__).parent.parent
PIN = B / 'tree-static-q5-M120-SLO-first-full463-own-AR-v257/FREEZE257.json'


def source_module(module, name, names, dimensions=None):
    expected = B / 'tree-static-q5-M120-node-math-v255' / name
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError('ROOT377 actual original source origin: ' + name)
    raw = expected.read_bytes()
    if hashlib.sha256(PIN.read_bytes()).hexdigest() != 'b352ac6bb9f3ae182d33d2f11dc278f7e49c2b8dff9d9d514c0b32d7a14a01f2':
        raise RuntimeError('ROOT377 exact original best257 source manifest')
    pins = json.loads(PIN.read_text())['files']
    if hashlib.sha256(raw).hexdigest() != pins[str(expected)]:
        raise RuntimeError('ROOT377 exact retained source bytes: ' + name)
    nodes = [copy.deepcopy(n) for n in ast.parse(raw).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise RuntimeError('ROOT377 complete original functions required')
    tree = ast.Module(body=nodes, type_ignores=[])
    before = copy.deepcopy(tree)
    deltas = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and type(n.value) is int and n.value in (dimensions or {}):
            deltas.append((n, n.value))
            n.value = dimensions[n.value]
    projected = copy.deepcopy(tree)
    for n, value in deltas:
        n.value = value
    if ast.dump(tree, include_attributes=False) != ast.dump(before, include_attributes=False):
        raise RuntimeError('ROOT377 whole dimensional AST inverse')
    namespace = dict(vars(module))
    exec(compile(ast.fix_missing_locations(projected), __file__ + '::' + name, 'exec'), namespace)
    return namespace, dict(functions=sorted(names), typed_constant_deltas=len(deltas),
                          original_source_sha256=hashlib.sha256(raw).hexdigest())


def function(function, changes, additions=None):
    source = inspect.getsource(function)
    changed = source
    for before, after in changes:
        if changed.count(before) != 1:
            raise RuntimeError('ROOT377 source anchor: ' + before)
        changed = changed.replace(before, after)
    inverse = changed
    for before, after in reversed(changes):
        if inverse.count(after) != 1:
            raise RuntimeError('ROOT377 inverse anchor: ' + after)
        inverse = inverse.replace(after, before)
    if inverse != source:
        raise RuntimeError('ROOT377 entire original function inverse')
    namespace = dict(function.__globals__)
    namespace.update(additions or {})
    exec(compile(changed, __file__ + '::' + function.__name__, 'exec'), namespace)
    return namespace[function.__name__]
