"""Apply the declared B1..8 domain to the two legacy raw-audit predicates."""
import ast
import copy
import hashlib
import types
from pathlib import Path

import audit209 as original


def declared_functions(path, namespace, **kwargs):
    text = original.functions(path, namespace, **kwargs)
    if Path(path).name != 'audit_production_v101.py' or kwargs.get('names') != ('audit_trace_set',):
        return text
    tree = ast.parse(text)
    before = ast.dump(tree, include_attributes=False)
    changed, functions, target_bounds = [], [], []
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        nodes = [node for node in ast.walk(function)
                 if isinstance(node, ast.Compare) and ast.unparse(node) == '1 <= b <= 5']
        if nodes:
            functions.append(function)
            for node in nodes:
                node.comparators[-1].value = 8
                changed.append(node)
        if function.name == 'audit_trace_set':
            for node in ast.walk(function):
                if isinstance(node, ast.Compare):
                    expression = ast.unparse(node)
                    if expression == 'sum(budgets) <= 96':
                        target_bounds.append((node.comparators[0], 96))
                        node.comparators[0].value = 128
                    elif expression == 'sum((b + 1 for b in budgets)) <= 128':
                        target_bounds.append((node.comparators[0], 128))
                        node.comparators[0].value = 160
    if len(changed) != 2 or {f.name for f in functions} != {'audit_transactions', 'audit_trace_set'}:
        raise RuntimeError('ROOT333 exact two original raw budget predicates required')
    if len(target_bounds) != 2:
        raise RuntimeError('ROOT333 exact LIVE C128/root160 raw audit bounds required')
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(functions), type_ignores=[])),
                 __file__ + '::original-domain8', 'exec'), namespace)
    for node in changed:
        node.comparators[-1].value = 5
    for node, value in target_bounds:
        node.value = value
    if ast.dump(tree, include_attributes=False) != before:
        raise RuntimeError('ROOT333 whole raw auditor AST inverse')
    return text


old = types.ModuleType('ROOT333_original_raw_auditor_declared_B8')
old.__dict__.update(vars(original))
old.functions = declared_functions
