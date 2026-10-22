"""Whole-LIVE HOST projection, with an exact complete AST inverse.

Only the admission body changes. Original generation, all clock calls, decode,
planner, payload transport, source controls and publication callbacks remain.
This proposal has no installer and requires future same-model NN proof.
"""
import ast
import copy
import hashlib
from pathlib import Path
import sys

P = Path(__file__).resolve().parent
B = P.parent
sys.path.insert(0, str(B/'tree-nonempty-refill-accounting-witness-host-v429'))
from admission429 import source as admission_source, need


def patch(source):
    original, _, proposed = admission_source()
    before = ast.parse(source)
    tree = copy.deepcopy(before)
    selected = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'admit_available']
    need(len(selected) == 1, 'one actual original admission function')
    function = selected[0]
    old = copy.deepcopy(function.body)
    need(ast.dump(function, include_attributes=False)
         == ast.dump(ast.parse(original).body[0], include_attributes=False),
         'exact frozen actual425 whole admission before declared change')
    candidate = ast.parse(proposed).body[0]
    imports = ast.parse('from witness431 import before as _before429, charged as _charged429, finish as _finish429').body
    constants = ast.parse('_REFILL_LOW_WATER426 = 32\n_REFILL_MIN426 = 8\n_REFILL_MAX426 = 32').body
    function.body = [*imports, *constants, *candidate.body]
    result = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    function.body = old
    need(ast.dump(tree, include_attributes=False) == ast.dump(before, include_attributes=False),
         'entire actual LIVE decode AST inverse after admission replacement')
    return result


def descriptor(source):
    new = patch(source)
    return dict(original_LIVE_sha256=hashlib.sha256(source.encode()).hexdigest(),
        proposed_LIVE_sha256=hashlib.sha256(new.encode()).hexdigest(),
        complete_AST_inverse=True, unchanged_original_clock_calls=True,
        whole_original_decode_outside_admission_unchanged=True,
        global_active_limit=64, maximum_nonempty_refill=32, low_water=32, minimum_refill=8,
        actual427_NN_and_original425_scope_required=True,
        NN_launched=False, serving_installed=False, Raw=None, Goodput=None)
