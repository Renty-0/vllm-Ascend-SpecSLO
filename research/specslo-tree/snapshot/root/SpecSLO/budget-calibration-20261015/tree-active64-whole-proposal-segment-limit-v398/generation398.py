"""Bind the extra physical-segment constraint at the sole actual LIVE call."""
import ast
import copy
import hashlib

from generation257 import patch as parent_patch, descriptor as parent_descriptor


def patch(source):
    base = parent_patch(source)
    before = ast.parse(base)
    tree = copy.deepcopy(before)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and ast.unparse(node.func) == 'controller.build_plan']
    if len(calls) != 1 or not calls[0].args:
        raise RuntimeError('ROOT398 exactly one actual LIVE original controller call')
    call = calls[0]
    original_func, original_args, original_keywords = copy.deepcopy((call.func, call.args, call.keywords))
    active = copy.deepcopy(call.args[0])
    call.func = ast.Name(id='_build_plan398', ctx=ast.Load())
    call.args.insert(0, ast.Name(id='controller', ctx=ast.Load()))
    call.keywords.extend([
        ast.keyword(arg='max_target_segments', value=ast.Constant(24)),
        ast.keyword(arg='ready_segment_counts', value=ast.Call(
            func=ast.Name(id='_ready_costs398', ctx=ast.Load()),
            args=[ast.Name(id='controller', ctx=ast.Load()),
                  ast.Name(id='payloads', ctx=ast.Load()), active], keywords=[])),
    ])
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == '_generate_spec_rhythm_tree_decode']
    if len(functions) != 1:
        raise RuntimeError('ROOT398 declared LIVE decode function')
    function = functions[0]
    index = int(bool(function.body and isinstance(function.body[0], ast.Expr)
                     and isinstance(function.body[0].value, ast.Constant)
                     and isinstance(function.body[0].value.value, str)))
    imported = ast.parse('from packing398 import build_plan as _build_plan398, ready_costs as _ready_costs398').body[0]
    function.body.insert(index, imported)
    result = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    function.body.remove(imported)
    call.func, call.args, call.keywords = original_func, original_args, original_keywords
    if ast.dump(tree, include_attributes=False) != ast.dump(before, include_attributes=False):
        raise RuntimeError('ROOT398 complete LIVE AST inverse')
    return result


def descriptor(source):
    from packing398 import descriptor as packing_descriptor
    return dict(parent_descriptor(source), projected_source_sha256=hashlib.sha256(patch(source).encode()).hexdigest(),
                explicit_segment_packing398=packing_descriptor(), actual_global_active_limit=64)
