"""LIVE-only spare branch groups, with the actual218 cost repair retained."""
import ast,copy,hashlib
from generation257 import patch as previous_patch,descriptor as previous_descriptor

def _group_patch(source):
    base=previous_patch(source);tree=ast.parse(base);original=ast.dump(tree,include_attributes=False)
    changes=[]
    replacements={'controller.build_plan':('_root333_build_plan','controller'),'shaper.shape':('_root333_shape','shaper')}
    for node in ast.walk(tree):
        if isinstance(node,ast.Call)and ast.unparse(node.func)in replacements:
            name,owner=replacements[ast.unparse(node.func)]
            changes.append((node,copy.deepcopy(node.func),copy.deepcopy(node.args)))
            node.func=ast.Name(id=name,ctx=ast.Load());node.args.insert(0,ast.Name(id=owner,ctx=ast.Load()))
    if len(changes)!=2:raise RuntimeError('one actual LIVE controller and shape delegate each')
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    added=ast.parse('from budget333 import build_plan as _root333_build_plan, shape as _root333_shape').body[0]
    index=1 if isinstance(function.body[0],ast.Expr)and isinstance(function.body[0].value,ast.Constant)and isinstance(function.body[0].value.value,str)else 0
    function.body.insert(index,added)
    counters=[n for n in ast.walk(tree)if isinstance(n,ast.Dict)and any(isinstance(k,ast.Constant)and k.value=='spec_rhythm_target_slo_first_enabled'for k in n.keys)]
    if len(counters)!=1:raise RuntimeError('one original LIVE policy counter initializer')
    counters[0].keys.append(ast.Constant('spec_rhythm_ROOT333_group_budget_enabled'));counters[0].values.append(ast.Constant(1))
    result=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    counters[0].keys.pop();counters[0].values.pop();function.body.remove(added)
    for node,func,args in changes:node.func=func;node.args=args
    if ast.dump(tree,include_attributes=False)!=original:raise RuntimeError('whole previous LIVE AST inverse')
    return result

def descriptor(source):
    previous=previous_descriptor(source)
    result=dict(previous,ROOT333_LIVE_source_sha256=hashlib.sha256(patch(source).encode()).hexdigest(),
        original_typed_allocator_gain_need_priority_and_charges_retained=True,
        primary_plus_side_groups_limit=32,spare_B5_owners_selected_by_priority_and_positive_gain=True,
        target_whole_ticket_group_budget=True,physical_FIA_M160_S32_KV2560_declared=True)
    result.update(previous257_policy_projection_source_sha256=previous['projected_source_sha256'],
        projected_source_sha256=result['ROOT333_LIVE_source_sha256'],
        only_LIVE_policy_and_counter_changed=False,whole_ticket_budget_and_native_maths_unchanged=False,
        native_model_algorithm_retained=True,whole_ticket_integrity_and_C128_global192_preserved=True,
        LIVE_per_owner_caps_and_target_group_selection_changed=True,
        actual_original218_repair_solver_retained=True, original_control_and_generation_state_cells_retained=True)
    return result


def patch(source):
    base=_group_patch(source);tree=ast.parse(base);original=ast.dump(tree,include_attributes=False)
    changes=[];counts=dict(plan=0,select=0,eagerfilter=0,maxgamma=0)
    for node in ast.walk(tree):
        if isinstance(node,ast.Call)and ast.unparse(node.func)=='self._spec_rhythm_tree_plan':
            changes.extend([(node,'func',copy.deepcopy(node.func)),(node,'args',copy.deepcopy(node.args))])
            node.func=ast.Name(id='_root333_plan',ctx=ast.Load());node.args.insert(0,ast.Name(id='self',ctx=ast.Load()));counts['plan']+=1
        if isinstance(node,ast.Name)and isinstance(node.ctx,ast.Load)and node.id in('_dt_select_backbone','_v132_filter_eager_parents'):
            old=node.id;changes.append((node,'id',old));node.id={'_dt_select_backbone':'_root333_select','_v132_filter_eager_parents':'_root333_filter'}[old]
            counts['select'if old=='_dt_select_backbone'else'eagerfilter']+=1
        if isinstance(node,ast.keyword)and node.arg=='spec_rhythm_max_gamma':
            if not isinstance(node.value,ast.Constant)or node.value.value!=5:raise RuntimeError('ROOT333 old request cap5')
            changes.append((node,'value',copy.deepcopy(node.value)));node.value=ast.Constant(8);counts['maxgamma']+=1
    if any(v!=1 for v in counts.values()):raise RuntimeError('ROOT333 exact LIVE plan/refine/filter/cap anchors '+str(counts))
    reserve=next(n for n in ast.walk(tree)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='_root256_pages'for t in n.targets))
    constants=[n for n in ast.walk(reserve.value)if isinstance(n,ast.Constant)and n.value==5]
    if len(constants)!=1:raise RuntimeError('ROOT333 exact original page reserve')
    changes.append((constants[0],'value',5));constants[0].value=8
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef))
    additions=ast.parse("from budget333 import filter_eager_parents as _root333_filter\nfrom plans333 import for_state as _root333_plan,select as _root333_select\nfrom live333 import begin as _root333_begin,observe as _root333_observe\n_root333_begin(self)").body
    index=1 if isinstance(function.body[0],ast.Expr)and isinstance(function.body[0].value,ast.Constant)and isinstance(function.body[0].value.value,str)else 0
    function.body[index:index]=additions
    commit=next(n for n in ast.walk(tree)if isinstance(n,ast.For)and ast.unparse(n.target)=='(row, index)'and ast.unparse(n.iter)=='enumerate(target_indices)'and any(isinstance(s,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='proposed_count'for t in s.targets)for s in n.body))
    where=next(i for i,n in enumerate(commit.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='proposed_count'for t in n.targets))+1
    observation=ast.parse('_root333_observe(self,states[index],tree_plan,accepted_rows[row],output_rows[row])').body[0]
    commit.body.insert(where,observation)
    result=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    commit.body.remove(observation)
    for n in additions:function.body.remove(n)
    for node,attribute,old in reversed(changes):setattr(node,attribute,old)
    if ast.dump(tree,include_attributes=False)!=original:raise RuntimeError('ROOT333 complete second LIVE AST inverse')
    return result


_group_descriptor=descriptor
def descriptor(source):
    value=_group_descriptor(source)
    value.update(actual_request_maxgamma=8,actual_root_alternate_selected_prefix=True,
        actual_same_prefix_g_B1to8=True,original_typed_allocator_gain_need_priority_and_charges_retained=False,
        original_need_formula_priority_and_candidate_charges_retained=True,
        spare_B5_owners_selected_by_priority_and_positive_gain=False,
        actual_B6to8_genuine_generation_required=True,physical_tail_reserved8=True,
        original_fresh_policy_clear_and_actual_generation_preserved=True,
        actual_paired_observation_before_original_verification_observe=True)
    return value


_before_capacity333_patch = patch

def patch(source):
    previous = _before_capacity333_patch(source)
    tree = ast.parse(previous)
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.keyword)
               and node.arg == 'max_target_requests'
               and isinstance(node.value, ast.Constant) and node.value.value == 24]
    if len(matches) != 1:
        raise RuntimeError('ROOT333 one actual inherited target24 admission cap')
    matches[0].value.value = 32
    roofs=[node for node in ast.walk(tree)if isinstance(node,ast.keyword)
           and node.arg=='verification_budget'
           and ast.unparse(node.value)=='self.config.spec_rhythm_verification_budget']
    if len(roofs)!=1:raise RuntimeError('One actual LIVE shaper verification roof required')
    prior_roof=roofs[0].value
    roofs[0].value=ast.parse('self._ROOT333_policy.candidate_roof',mode='eval').body
    result = ast.unparse(ast.fix_missing_locations(tree)) + '\n'
    matches[0].value.value = 24
    roofs[0].value=prior_roof
    if ast.dump(tree, include_attributes=False) != ast.dump(ast.parse(previous), include_attributes=False):
        raise RuntimeError('ROOT333 whole target admission cap AST inverse')
    return result
