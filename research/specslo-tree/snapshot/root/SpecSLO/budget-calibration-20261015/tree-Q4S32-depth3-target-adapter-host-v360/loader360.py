"""Checked, reversible construction of a future target-only Q4/S32 adapter.

No Native qualification flag is assigned. This source construction performs
no NPU operation and is not authorization to install or adopt the adapter.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path


P = Path(__file__).resolve().parent
OLD = P.parent / 'tree-M160-S32-N32-target-math-cost-v326'
FREEZE_SHA = '1c8878b7490ed85684aea9686a25652b2488f30074769b0568243e5c476fbc43'
EXPECTED_160 = {'branch_tail': 1, 'static_base': 2, 'static_state': 8, 'write': 4}
IMPORTS = dict(branch_tail326='branch360', static_base326='static_base360',
               projection326='projection358', padding326='padding360',
               layout326='layout360', descriptors326='descriptors359', write326='write360')
LABELS = dict(fixed_q5_groups='fixed_q4_groups',
              ROOT305_original_attention_and_Q5_geometry_preserved='ROOT360_original_attention_body_preserved')


def source(which):
    if which not in EXPECTED_160:
        raise ValueError('explicit original adapter module')
    freeze_raw = (OLD / 'FREEZE326.json').read_bytes()
    if hashlib.sha256(freeze_raw).hexdigest() != FREEZE_SHA:
        raise RuntimeError('exact original326 freeze required')
    path = OLD / (which + '326.py')
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != json.loads(freeze_raw)['files'][str(path)]:
        raise RuntimeError('exact original326 module bytes required')
    tree = ast.parse(raw)
    if which == 'branch_tail':
        tree.body = [node for node in tree.body if isinstance(node, ast.ClassDef)
                     or isinstance(node, ast.FunctionDef) and node.name == 'install']
    before = copy.deepcopy(tree)
    changes = []
    integers = []

    def change(node, key, value):
        changes.append((node, key, getattr(node, key)))
        setattr(node, key, value)

    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value == 160:
            integers.append(node)
            change(node, 'value', 128)
        elif isinstance(node, ast.ImportFrom) and node.module in IMPORTS:
            change(node, 'module', IMPORTS[node.module])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in IMPORTS:
                    change(alias, 'name', IMPORTS[alias.name])
        elif isinstance(node, ast.Name) and node.id in IMPORTS:
            change(node, 'id', IMPORTS[node.id])
        elif isinstance(node, ast.arg) and node.arg in IMPORTS:
            change(node, 'arg', IMPORTS[node.arg])
        elif isinstance(node, ast.keyword) and node.arg in LABELS:
            change(node, 'arg', LABELS[node.arg])
    if len(integers) != EXPECTED_160[which]:
        raise RuntimeError('all and only declared physical160 dimensions required')
    projected = copy.deepcopy(tree)
    summary = [(type(node).__name__, key, value, getattr(node, key)) for node, key, value in changes]
    for node, key, value in reversed(changes):
        setattr(node, key, value)
    if ast.dump(tree, include_attributes=False) != ast.dump(before, include_attributes=False):
        raise RuntimeError('complete original module AST inverse required')
    return ast.unparse(ast.fix_missing_locations(projected)) + '\n', summary


def execute(which, module_name):
    projected, changes = source(which)
    namespace = dict(__name__=module_name, __file__=str(P / (module_name + '.py')),
                     ROOT360_SOURCE_CHANGES=changes,
                     ROOT360_NEURAL_MATH_QUALIFIED=False)
    if which == 'branch_tail':
        from contextlib import contextmanager
        from dataclasses import replace
        from layout360 import is_fork, layout
        namespace.update(contextmanager=contextmanager, replace=replace,
                         COPY_SLOTS=1792, BLOCK=128, CONTEXT=4096,
                         is_fork=is_fork, layout=layout)
    exec(compile(projected, str(P / (module_name + '.py')) + '::original326-explicit-Q4S32', 'exec'), namespace)
    return namespace
