"""Keep the whole LIVE program; change only prefix admission selection."""

import ast
import copy
import generation438 as original


def project(source):
    before = ast.parse(source)
    tree = copy.deepcopy(before)
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'admit_available']
    if len(functions) != 1:
        raise RuntimeError('One original admission function required')
    function = functions[0]
    old_body = copy.deepcopy(function.body)
    imports = [n for n in function.body if isinstance(n, ast.ImportFrom) and n.module == 'witness431']
    if len(imports) != 1:
        raise RuntimeError('One original same-model refill witness import required')
    imports[0].module = 'witness463'
    assignments = [n for n in ast.walk(function) if isinstance(n, ast.Assign)
                   and len(n.targets) == 1 and ast.unparse(n.targets[0]) == 'vacant_capacity']
    if len(assignments) != 1:
        raise RuntimeError('One original vacancy expression required')
    assignments[0].value = ast.Call(func=ast.Name(id='min', ctx=ast.Load()),
                                   args=[assignments[0].value, ast.Constant(value=32)], keywords=[])
    minimums = [i for i, n in enumerate(function.body) if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'active and len(admitted) < _REFILL_MIN426']
    if len(minimums) != 1:
        raise RuntimeError('Original minimum check before every home/cache/token mutation required')
    nodes = ast.parse('''
from admission463 import choose as _choose463
admitted = _choose463(self, admitted, active=active, request_params=request_params,
    states=states, local_states=local_states, prefetched=prefetched, now=now, counters=counters)
''').body
    function.body[minimums[0]:minimums[0]] = nodes
    result = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    function.body = old_body
    if ast.dump(tree, include_attributes=False) != ast.dump(before, include_attributes=False):
        raise RuntimeError('Whole original LIVE AST inverse required')
    return result


def patch(source):
    return project(original.patch(source))
