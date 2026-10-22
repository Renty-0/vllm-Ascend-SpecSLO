"""Uninstalled first-cohort scheduling prototype; no backend imports.

Client arrival times, all later cohorts, and all original clocks stay intact.
The proposed server wait is measured inside the original generation interval.
This file grants no device qualification and is not a benchmark result.
"""
import ast
import copy

MINIMUM = 8
MAXIMUM_WAIT_MS = 1000.0
BLOCK = '''if (arrival_gated and not active and not completed and admitted
        and len(admitted) < 8 and len(pending_admission) >= 8
        and not token_budget_saturated):
    if all(request_params[i].arrival_ts is not None for i in admitted):
        _root_cold_oldest = min(request_params[i].arrival_ts for i in admitted)
        if 0.0 <= (now - _root_cold_oldest) * 1000.0 < 1000.0:
            admitted = []
'''


def need(value, message):
    if not value:
        raise ValueError('cold-start prototype: ' + message)


def admission(tree):
    rows = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'admit_available']
    need(len(rows) == 1, 'one original actual admission function')
    return rows[0]


def same(a, b):
    return ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)


def block():
    return ast.parse(BLOCK).body[0]


def patch(source):
    tree = ast.parse(source)
    fn = admission(tree)
    need(not any(same(node, block()) for node in fn.body), 'no duplicate policy')
    anchors = [i for i, node in enumerate(fn.body) if isinstance(node, ast.If)
               and ast.unparse(node.test) == 'admitted and prefill_coalescing']
    need(len(anchors) == 1, 'one original post-selection coalesce decision')
    index = anchors[0]
    need(isinstance(fn.body[index-1], ast.For)
         and ast.unparse(fn.body[index-1].target) == 'index'
         and ast.unparse(fn.body[index-1].iter) == 'admission_candidates',
         'original ready-only admission selection immediately before policy')
    fn.body.insert(index, block())
    value = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    need(same(ast.parse(source), ast.parse(unpatch(value))), 'whole original AST restored')
    return value


def unpatch(source):
    tree = ast.parse(source)
    fn = admission(tree)
    rows = [i for i, node in enumerate(fn.body) if same(node, block())]
    need(len(rows) == 1, 'one exact added policy block')
    del fn.body[rows[0]]
    return ast.unparse(ast.fix_missing_locations(tree)) + '\n'


def evaluate(*, now, admitted, params, pending, active=(), completed=(),
             arrival_gated=True, token_budget_saturated=False):
    """HOST execution of the exact inserted AST, never a model caller."""
    env = dict(now=now, admitted=list(admitted), request_params=params,
               pending_admission=list(pending), active=list(active),
               completed=list(completed), arrival_gated=arrival_gated,
               token_budget_saturated=token_budget_saturated)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(block())],
                 type_ignores=[])), '<cold-start-exact-policy-HOST>', 'exec'), env)
    return env['admitted']
