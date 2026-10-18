"""Build reviewed HOST source strings; never install/execute a Native method.

Two semantic delegate hooks: the shape wrapper and draft refinement. Separate
accounting/conservation edits are mandatory, not concealed by a roof192 alias.
Input is the pinned V119 source component, not an already-qualified live AST.
"""
import ast
import copy
import hashlib

SHAPE_BODY = '''return _v123_shape_draft_work(policy, instance,
    admitted_eager_indices=tuple(kwargs['eager_request_indices']),
    normal_gt16_cap4=_v123_use_familycap(), **kwargs)'''

RUNTIME_RULES = (
    ('target_roof_counter',
     "counters['spec_rhythm_last_verification_roof'] = budget_plan.verification_roof",
     "counters['spec_rhythm_last_verification_roof'] = budget_plan.target_limit.candidates"),
    ('actual_target_unused_counter',
     "counters['spec_rhythm_unused_verification_tokens'] = int(budget_plan.unused_verification_tokens)",
     "counters['spec_rhythm_unused_verification_tokens'] = budget_plan.target_limit.candidates - actual_candidates"),
    ('draft_unused_after_shape',
     "counters['spec_rhythm_unused_verification_tokens'] = max(0, int(budget_plan.verification_roof) - sum(normal_budgets.values()) - sum(eager_budgets.values()))",
     "counters['spec_rhythm_unused_draft_candidates'] = budget_plan.unused_draft_candidates"),
    ('actual_work_conservation',
     "tickets = [controller.new_ticket(index, gamma=budget, eager=is_eager) for index, budget, is_eager in zip(work_indices, work_budgets, work_eager)]",
     "budget_plan.require_work(work_indices, work_budgets, work_eager, work_plans)\n"
     "tickets = [controller.new_ticket(index, gamma=budget, eager=is_eager) for index, budget, is_eager in zip(work_indices, work_budgets, work_eager)]"),
    ('two_original_group_refinements',
     "refined = _dt_select_backbone(exploratory, budget_plan.verification_roof)",
     "refined = _v123_refine_draft_groups(exploratory, budget_plan, _dt_select_backbone)"),
    ('published_ticket_conservation',
     "controller.publish(published_tickets)",
     "budget_plan.require_published_tickets(published_tickets)\ncontroller.publish(published_tickets)"),
    ('draft_unused_after_publish',
     "counters['spec_rhythm_unused_verification_tokens'] = budget_plan.verification_roof - sum((ticket.gamma for ticket in published_tickets))",
     "counters['spec_rhythm_unused_draft_candidates'] = budget_plan.available_selected_candidates - sum((ticket.gamma for ticket in published_tickets))"),
)


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def ast_digest(tree):
    return digest(ast.dump(tree,include_attributes=False))


def require_source(source, expected_sha):
    if digest(source)!=expected_sha:
        raise RuntimeError('V123 refuses an unbound/changed source component')


def _apply_sequences(tree,rules):
    counts={name:0 for name,_,_ in rules}
    compiled=[(name,tuple(ast.unparse(n) for n in ast.parse(old).body),ast.parse(new).body)
              for name,old,new in rules]
    cache={}
    def normalized(node):
        if node not in cache:cache[node]=ast.unparse(node)
        return cache[node]
    def visit(node):
        for key,value in ast.iter_fields(node):
            if isinstance(value,ast.AST):visit(value)
            elif isinstance(value,list):
                for child in value:
                    if isinstance(child,ast.AST):visit(child)
                if not value or not all(isinstance(n,ast.stmt) for n in value):continue
                for name,old,new in compiled:
                    i=0
                    while i<=len(value)-len(old):
                        if tuple(normalized(n) for n in value[i:i+len(old)])==old:
                            value[i:i+len(old)]=copy.deepcopy(new)
                            counts[name]+=1;i+=len(new)
                        else:i+=1
    visit(tree)
    if any(n!=1 for n in counts.values()):
        raise RuntimeError('V123 exact AST anchor count differs: '+repr(counts))
    return counts


def prepare_shape_adapter(source,expected_sha):
    require_source(source,expected_sha)
    tree=ast.parse(source);before=ast_digest(tree)
    funcs=[n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='shape']
    if len(funcs)!=1 or ast.unparse(funcs[0].args)!='instance, **kwargs':
        raise RuntimeError('V123 actual shape wrapper anchor differs')
    function=funcs[0];old=copy.deepcopy(function.body)
    if not any(isinstance(n,ast.Return) and isinstance(n.value,ast.Call)
               and ast.unparse(n.value.func)=='rhythm.SpecRhythmBudgetPlan' for n in old):
        raise RuntimeError('V123 requires the original Native plan-returning wrapper')
    function.body=ast.parse(SHAPE_BODY).body
    result=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    changed=ast_digest(tree)
    function.body=old
    if ast_digest(tree)!=before:raise RuntimeError('V123 shape edit touched another AST region')
    return result,dict(kind='one_shape_wrapper_body',before_AST_sha256=before,
        after_AST_sha256=changed,original_body=ast.unparse(ast.Module(body=old,type_ignores=[])),
        new_body=SHAPE_BODY,exact_reverse_restores_original_AST=True,
        required_unbound_globals=['_v123_shape_draft_work','_v123_use_familycap'])


def prepare_runtime_adapter(source,expected_sha):
    require_source(source,expected_sha)
    tree=ast.parse(source);before=ast_digest(tree)
    counts=_apply_sequences(tree,RUNTIME_RULES)
    result=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    changed=ast_digest(tree)
    # No old target-roof field may accidentally serve as a draft work roof.
    for node in ast.walk(tree):
        if isinstance(node,ast.Attribute) and ast.unparse(node.value)=='budget_plan' and node.attr in (
                'verification_roof','unused_verification_tokens'):
            raise RuntimeError('V123 left an ambiguous old draft budget field')
    reverse=tuple((name,new,old) for name,old,new in RUNTIME_RULES)
    _apply_sequences(tree,reverse)
    if ast_digest(tree)!=before:raise RuntimeError('V123 runtime edit touched an undeclared AST region')
    return result,dict(kind='runtime_refinement_plus_mandatory_accounting_and_conservation',
        before_AST_sha256=before,after_AST_sha256=changed,counts=counts,
        exact_reverse_restores_original_AST=True,
        rules=[dict(name=name,original=old,candidate=new) for name,old,new in RUNTIME_RULES],
        required_unbound_globals=['_v123_refine_draft_groups'],
        changes_current_target_admission=False,adds_NPU_ops_or_collectives=False,
        full_live_AST_qualified=False)
