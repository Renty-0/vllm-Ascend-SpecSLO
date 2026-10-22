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
        raise RuntimeError('ROOT423 exactly one actual LIVE original controller call')
    call = calls[0]
    original_func, original_args, original_keywords = copy.deepcopy((call.func, call.args, call.keywords))
    active = copy.deepcopy(call.args[0])
    call.func = ast.Name(id='_build_plan423', ctx=ast.Load())
    call.args.insert(0, ast.Name(id='controller', ctx=ast.Load()))
    call.keywords.extend([
        ast.keyword(arg='max_target_segments', value=ast.Constant(24)),
        ast.keyword(arg='ready_segment_counts', value=ast.Call(
            func=ast.Name(id='_ready_costs423', ctx=ast.Load()),
            args=[ast.Name(id='controller', ctx=ast.Load()),
                  ast.Name(id='payloads', ctx=ast.Load()), active], keywords=[])),
    ])
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == '_generate_spec_rhythm_tree_decode']
    if len(functions) != 1:
        raise RuntimeError('ROOT423 declared LIVE decode function')
    function = functions[0]
    index = int(bool(function.body and isinstance(function.body[0], ast.Expr)
                     and isinstance(function.body[0].value, ast.Constant)
                     and isinstance(function.body[0].value.value, str)))
    imported = ast.parse('from packing423 import build_plan as _build_plan423, ready_costs as _ready_costs423').body[0]
    function.body.insert(index, imported)
    imported_shape = ast.parse('from shape423 import shape as _shape423').body[0]
    function.body.insert(index, imported_shape)
    shaped = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
              and ast.unparse(node.func) == 'shaper.shape']
    if len(shaped) != 1:
        raise RuntimeError('ROOT423 one actual typed shaper call')
    shaped_call = shaped[0]
    shaped_original = copy.deepcopy((shaped_call.func, shaped_call.args))
    shaped_call.func = ast.Name(id='_shape423', ctx=ast.Load())
    shaped_call.args[:0] = [ast.Name(id='self', ctx=ast.Load()), ast.Name(id='shaper', ctx=ast.Load())]
    added_shape_keywords = [ast.keyword(arg='current_'+name, value=ast.Name(id=value, ctx=ast.Load()))
                            for name, value in (('controller','controller'), ('payloads','payloads'),
                                                ('local_states','local_states'))]
    shaped_call.keywords.extend(added_shape_keywords)
    filters = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
               and ast.unparse(node.func) == '_v132_filter_eager_parents']
    if len(filters) != 1:
        raise RuntimeError('ROOT423 one exact original pre-window parent filter')
    filtered = filters[0]
    old_filter = copy.deepcopy((filtered.func, filtered.args))
    filtered.func = ast.Name(id='_filter423', ctx=ast.Load())
    filtered.args.insert(0, ast.Name(id='self', ctx=ast.Load()))
    filtered.args.append(ast.Name(id='local_states', ctx=ast.Load()))
    imported_filter = ast.parse('from shape423 import filter_parents as _filter423').body[0]
    function.body.insert(index, imported_filter)
    # The original role refiner preserves every paid B. For the new exact
    # plan, use its explicit private E4 cap rather than the old config cap3.
    caps = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
            and ast.unparse(node) == 'min(cap, effective_eager_cap)']
    if len(caps) != 1:
        raise RuntimeError('ROOT423 one exact final future candidate cap')
    old_cap = copy.deepcopy(caps[0].args[1])
    caps[0].args[1] = ast.parse('budget_plan.work_limit.eager_per_request', mode='eval').body
    result = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    function.body.remove(imported)
    function.body.remove(imported_shape)
    function.body.remove(imported_filter)
    filtered.func, filtered.args = old_filter
    caps[0].args[1] = old_cap
    for keyword in added_shape_keywords:
        shaped_call.keywords.remove(keyword)
    shaped_call.func, shaped_call.args = shaped_original
    call.func, call.args, call.keywords = original_func, original_args, original_keywords
    if ast.dump(tree, include_attributes=False) != ast.dump(before, include_attributes=False):
        raise RuntimeError('ROOT423 complete LIVE AST inverse')
    return result


def descriptor(source):
    from packing423 import descriptor as packing_descriptor
    return dict(parent_descriptor(source), projected_source_sha256=hashlib.sha256(patch(source).encode()).hexdigest(),
                explicit_segment_packing423=packing_descriptor(), actual_global_active_limit=64,
                actual_typed_normal_work_admission_limit=24, original_gain_model_unchanged=True,
                selective_normal_parent4_E4_after_current75_proof=True,
                original_pre_window_filter_location_and_ranked_window_unchanged=True,
                final_future_cap_from_actual_paid_work_limit=True)
