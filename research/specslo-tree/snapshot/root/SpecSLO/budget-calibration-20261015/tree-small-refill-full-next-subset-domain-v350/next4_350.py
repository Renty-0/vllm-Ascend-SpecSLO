"""Original independent nextdraft proof for actual home sizes1,2,4,16."""
import ast
import copy
from pathlib import Path

import next4_209 as original

_NAMESPACE = None


def namespace():
    global _NAMESPACE
    if _NAMESPACE is not None:
        return _NAMESPACE
    source = Path(original.__file__).read_text()
    tree = ast.parse(source)
    unchanged = ast.dump(tree, include_attributes=False)
    changes = []
    owners = []
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            if isinstance(node, ast.Compare) and ast.unparse(node) == 'n in (1, 16)':
                old = copy.deepcopy(node.comparators[0])
                node.comparators[0] = ast.Tuple(elts=[ast.Constant(n) for n in (1,2,4,16)], ctx=ast.Load())
                changes.append((node, old))
                owners.append(function.name)
    if len(changes) != 2 or set(owners) != {'work_plan','eager_reference'}:
        raise RuntimeError('exact two original finite nextdraft domain predicates')
    value = dict(vars(original))
    exec(compile(ast.fix_missing_locations(tree), __file__+'::original-actual-home-size-domain', 'exec'), value)
    for node, old in changes:
        node.comparators[0] = old
    if ast.dump(tree, include_attributes=False) != unchanged:
        raise RuntimeError('whole original nextdraft source AST inverse')
    _NAMESPACE = value
    return value


def generate(*args, **kwargs): return namespace()['generate'](*args, **kwargs)
