"""Fresh original trace namespace with a distinct admission-window report."""
import ast
import copy
from pathlib import Path
from window160 import BASE, P, checked, require, runtime_bindings, window


def trace_source(active_window):
    active_window = window(active_window)
    path = BASE / 'production_probe_v101.py'
    tree = ast.parse(checked(path).read_text())
    cls = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.ClassDef)
                             and n.name == 'CapacityEntryObservation'))
    fn = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                            and n.name == 'run_trace'))
    imports = [n for n in fn.body if isinstance(n, ast.ImportFrom) and n.module == 'live_decode_patch']
    require(len(imports) == 1 and [(a.name, a.asname) for a in imports[0].names] == [('install', None)],
            'one actual installer import')
    fn.body.remove(imports[0])  # Explicit isolated install is supplied below.
    counters = dict(cap=0, fields=0, callback=0, stream=0)
    for n in ast.walk(fn):
        if isinstance(n, ast.Compare) and ast.unparse(n) == '0 < peak <= 64':
            n.comparators[-1] = ast.Constant(active_window)
            counters['cap'] += 1
        if isinstance(n, ast.keyword) and n.arg == 'actual_capacity64_scope_completed':
            require(ast.unparse(n.value) == 'peak > 32', 'original capacity coverage declaration')
            n.value = ast.Constant(False)
            counters['fields'] += 1
        if isinstance(n, ast.keyword) and n.arg == 'on_token_commit':
            require(ast.unparse(n.value) == 'lambda event: stream.append(dict(event))', 'original event callback')
            n.value = ast.Name(id='_v160_capture_event', ctx=ast.Load())
            counters['callback'] += 1
    for i, n in enumerate(fn.body):
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Tuple) and
                [a.id for a in t.elts if isinstance(a, ast.Name)] == ['stream', 'restored_class'] for t in n.targets):
            extra = ast.parse('''
_v160_first = set()
def _v160_capture_event(event):
    row = dict(event)
    owner = row['request_index']
    if owner not in _v160_first:
        _v160_first.add(owner)
        row['first_token_wall_seconds160'] = time.time()
    stream.append(row)
''').body
            fn.body[i+1:i+1] = extra
            counters['stream'] += 1
            break
    for n in ast.walk(fn):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'dict':
            if any(k.arg == 'actual_capacity64_scope_completed' for k in n.keywords):
                n.keywords += [ast.keyword(arg='configured_capacity', value=ast.Constant(64)),
                               ast.keyword(arg='admission_window160', value=ast.Constant(active_window)),
                               ast.keyword(arg='actual_window_covered160', value=ast.Compare(left=ast.Name(id='peak', ctx=ast.Load()),
                                   ops=[ast.Eq()], comparators=[ast.Constant(active_window)]))]
    require(counters == dict(cap=1, fields=2, callback=1, stream=1), 'exact trace-only adaptation anchors')
    return ast.fix_missing_locations(ast.Module(body=[cls, fn], type_ignores=[]))


def make_trace(active_window):
    import production_probe_v101 as original
    require(Path(original.__file__).resolve() == BASE / 'production_probe_v101.py', 'original actual trace module')
    install, assertion, state = runtime_bindings(active_window)
    ns = dict(vars(original), install=install, _assert_live_decode=assertion)
    exec(compile(trace_source(active_window), str(P / 'probe160.py') + '::actual-trace', 'exec'), ns)
    return ns['run_trace'], state
