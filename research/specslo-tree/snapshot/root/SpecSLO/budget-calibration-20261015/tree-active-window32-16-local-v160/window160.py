"""One exact admission-cap edit; no backend import or parent-global mutation."""
import ast
import copy
import hashlib
import json
from pathlib import Path

P = Path(__file__).resolve().parent
BASE = P.parent / 'tree-high-normal-cap4-native-families-v148'
FILENAME = '<V160-original-live-with-active-window>'
WINDOWS = (32, 24, 16)
DEFAULT_WINDOWS = (32, 16)


def require(value, message):
    if value is not True:
        raise RuntimeError('V160: ' + message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path):
    path = Path(path)
    pins = json.loads((P / 'PARENT_INPUTS160.json').read_text())['files']
    require(pins.get(str(path)) == sha(path), 'exact read-only producer ' + str(path))
    return path


def window(value):
    require(type(value) is int and value in WINDOWS, 'explicit active window32/24/16')
    return value


def windows(values):
    values = tuple(window(v) for v in values)
    require(bool(values) and len(set(values)) == len(values), 'nonempty unique window order')
    return values


def parse_windows(text):
    """Small caller/CLI adapter; it never starts or constructs an engine."""
    require(type(text) is str and bool(text), 'comma-separated window order')
    return windows(tuple(int(v) for v in text.split(',')))


def adapt_source(source, active_window):
    """Cap the actual decode argument, preserving every original statement."""
    active_window = window(active_window)
    tree = ast.parse(source)
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == '_generate_spec_rhythm_tree_decode']
    require(len(funcs) == 1, 'one actual decode definition')
    fn = funcs[0]
    require('initial_batch_size' in [a.arg for a in fn.args.kwonlyargs],
            'original explicit keyword admission argument')
    require(not any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
            and t.id == 'initial_batch_size' for t in n.targets) for n in ast.walk(fn)),
            'no prior cap or reassignment of admission argument')
    original = ast.dump(tree)
    index = int(bool(fn.body and isinstance(fn.body[0], ast.Expr)
                     and isinstance(fn.body[0].value, ast.Constant)
                     and isinstance(fn.body[0].value.value, str)))
    node = ast.parse('initial_batch_size = min(initial_batch_size, ' + str(active_window) + ')').body[0]
    fn.body.insert(index, node)
    reverse = copy.deepcopy(tree)
    reverse.body[tree.body.index(fn)].body.pop(index)
    require(ast.dump(reverse) == original, 'exact complete original AST inverse')
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def isolated_function(path, name, namespace):
    source = checked(path).read_text()
    node = copy.deepcopy(next(n for n in ast.parse(source).body
                             if isinstance(n, ast.FunctionDef) and n.name == name))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(path), 'exec'), namespace)
    return namespace[name]


def runtime_bindings(active_window):
    """Bind only on explicit actual execution after own ROOT readiness.

    The parent factory and fresh-decode shell stay original. An isolated copy
    of install compiles the cap into its delegate, while an isolated original
    source assertion checks exactly that new delegate. No class/model is
    constructed here.
    """
    active_window = window(active_window)
    import live_decode_patch as live
    import live_poisson_probe as observation
    require(Path(live.__file__).resolve() == BASE / 'live_decode_patch.py'
            and Path(observation.__file__).resolve() == BASE / 'live_poisson_probe.py',
            'canonical parent live modules')
    checked(live.__file__)
    checked(observation.__file__)
    state = {}
    namespace = dict(vars(live))
    tree = ast.parse(checked(BASE / 'live_decode_patch.py').read_text())
    node = copy.deepcopy(next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'install'))
    replacements = 0
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and n.value == '<V81-bounded-live-mixed-prefill>':
            n.value = FILENAME
            replacements += 1
    require(replacements == 1, 'one original delegate compile filename')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(P / 'window160.py') + '::isolated-install', 'exec'), namespace)
    installer = namespace['install']

    def install(native, make_hooks):
        state['source'] = adapt_source(live.build_live_runtime(), active_window)
        namespace['build_live_runtime'] = lambda: state['source']
        return installer(native, make_hooks)

    assertion_ns = dict(vars(observation), LIVE_FILENAME=FILENAME)
    def actual_source():
        require('source' in state, 'source assertion follows actual cap installation')
        return state['source']
    assertion_ns['_live_source'] = actual_source
    assertion = isolated_function(BASE / 'live_poisson_probe.py', '_assert_live_decode', assertion_ns)
    return install, assertion, state
