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
    changed, functions = [], []
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
    if len(changed) != 2 or {f.name for f in functions} != {'audit_transactions', 'audit_trace_set'}:
        raise RuntimeError('ROOT307 exact two original raw budget predicates required')
    exec(compile(ast.fix_missing_locations(ast.Module(body=copy.deepcopy(functions), type_ignores=[])),
                 __file__ + '::original-domain8', 'exec'), namespace)
    for node in changed:
        node.comparators[-1].value = 5
    if ast.dump(tree, include_attributes=False) != before:
        raise RuntimeError('ROOT307 whole raw auditor AST inverse')
    return text


old = types.ModuleType('ROOT307_original_raw_auditor_declared_B8')
old.__dict__.update(vars(original))
old.functions = declared_functions
