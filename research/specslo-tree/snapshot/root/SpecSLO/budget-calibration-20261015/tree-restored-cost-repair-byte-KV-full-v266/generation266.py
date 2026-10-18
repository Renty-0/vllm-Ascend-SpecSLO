"""LIVE-only spare branch groups, with the actual218 cost repair retained."""
import ast,copy,hashlib
from generation257 import patch as previous_patch,descriptor as previous_descriptor

def patch(source):
    base=previous_patch(source);tree=ast.parse(base);original=ast.dump(tree,include_attributes=False)
    changes=[]
    replacements={'controller.build_plan':('_root266_build_plan','controller'),'shaper.shape':('_root266_shape','shaper')}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call)and ast.unparse(node.func)in replacements:
            name,owner=replacements[ast.unparse(node.func)]
            changes.append((node,copy.deepcopy(node.func),copy.deepcopy(node.args)))
            node.func=ast.Name(id=name,ctx=ast.Load());node.args.insert(0,ast.Name(id=owner,ctx=ast.Load()))
    if len(changes)!=2:raise RuntimeError('one actual LIVE controller and shape delegate each')
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    added=ast.parse('from budget266 import build_plan as _root266_build_plan, shape as _root266_shape').body[0]
    index=1 if isinstance(function.body[0],ast.Expr)and isinstance(function.body[0].value,ast.Constant)and isinstance(function.body[0].value.value,str)else 0
    function.body.insert(index,added)
    counters=[n for n in ast.walk(tree)if isinstance(n,ast.Dict)and any(isinstance(k,ast.Constant)and k.value=='spec_rhythm_target_slo_first_enabled'for k in n.keys)]
    if len(counters)!=1:raise RuntimeError('one original LIVE policy counter initializer')
    counters[0].keys.append(ast.Constant('spec_rhythm_ROOT266_group_budget_enabled'));counters[0].values.append(ast.Constant(1))
    result=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    counters[0].keys.pop();counters[0].values.pop();function.body.remove(added)
    for node,func,args in changes:node.func=func;node.args=args
    if ast.dump(tree,include_attributes=False)!=original:raise RuntimeError('whole previous LIVE AST inverse')
    return result

def descriptor(source):
    previous=previous_descriptor(source)
    result=dict(previous,ROOT266_LIVE_source_sha256=hashlib.sha256(patch(source).encode()).hexdigest(),
        original_typed_allocator_gain_need_priority_and_charges_retained=True,
        primary_plus_side_groups_limit=24,spare_B5_owners_selected_by_priority_and_positive_gain=True,
        target_whole_ticket_group_budget=True,original_physical_M120_S24_KV2560_unchanged=True)
    result.update(previous257_policy_projection_source_sha256=previous['projected_source_sha256'],
        projected_source_sha256=result['ROOT266_LIVE_source_sha256'],
        only_LIVE_policy_and_counter_changed=False,whole_ticket_budget_and_native_maths_unchanged=False,
        native_model_maths_unchanged=True,whole_ticket_integrity_and_C96_global192_preserved=True,
        LIVE_per_owner_caps_and_target_group_selection_changed=True,
        actual_original218_repair_solver_retained=True, original_control_and_generation_state_cells_retained=True)
    return result
