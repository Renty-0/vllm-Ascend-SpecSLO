"""Uninstalled HOST review prototype. No Torch/Native import or device launcher.

This proves only proposal arithmetic and unchanged selector-body contracts.
Caller-supplied eligibility/order is a scenario, never an observed V96 gate.
"""
import ast
import copy
from dataclasses import dataclass
import hashlib
from pathlib import Path

PARENT=Path('/root/SpecSLO/budget-calibration-20261015/tree-packed-eager01-cost-v102/packed_root_seed_eager3.py')
PARENT_SHA='0ab08f7d3822622baf64056ac0ff64a573f30cb94d6bfa027a8ded5c6ac3bef3'


def reserve_one_eager3(normal,*,eager_owner,reduction_order):
    """Counterfactual C96 reservation before tickets/plans are created.

    Preserve every normal row's B>=1 and one normal B>=4. Caller chooses the
    reduction order; this module has no measured marginal-loss/SLO estimate.
    This deliberately spends up to3 normal candidates instead of raising C96.
    """
    before=dict(normal);order=tuple(reduction_order)
    if (not 1<=len(before)<=32 or type(eager_owner) is not int or eager_owner in before
        or any(type(i) is not int or i<0 or type(b) is not int or not 1<=b<=5 for i,b in before.items())
        or len(order)!=len(set(order)) or set(order)!=set(before) or sum(before.values())>96):
        raise ValueError('Explicit normal1..32/B1..5/C96 and disjoint eager owner/order required')
    anchors=[i for i,b in before.items() if b>=4]
    if not anchors:raise ValueError('All-B<=3 needs a separate shorter-tail qualification')
    anchor=min(anchors);after=dict(before);debit=max(0,sum(before.values())+3-96);remaining=debit
    for owner in order:
        minimum=4 if owner==anchor else 1
        take=min(remaining,after[owner]-minimum)
        after[owner]-=take;remaining-=take
    if remaining:raise ValueError('Cannot fund E1B3 while preserving normal minimums and four-step anchor')
    assert sum(after.values())+3<=96 and max(after.values())>=4
    return dict(normal_budgets=after,eager_budgets={eager_owner:3},normal_candidates_debited=debit,
        preserved_four_step_normal_owner=anchor,combined_C=sum(after.values())+3,
        combined_draft_rows=len(after)+1,normal_target_queries=sum(after.values())+len(after),
        supplied_eligibility_and_reduction_order_are_actual_measurements=False,
        installed=False,NPU_qualified=False,performance_qualified=False)


def prepared_selector_source():
    """Exactly two budget guards plus an explicit original-four-step gate."""
    raw=PARENT.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PARENT_SHA:raise ValueError('Frozen packed selector parent changed')
    tree=ast.parse(raw);original=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select')
    patched=copy.deepcopy(original);changes={}
    class Rewrite(ast.NodeTransformer):
        def visit_Compare(self,node):
            text=ast.unparse(node)
            if text=='int(plan.candidate_budget) != (3 if seq in sources else 5)':
                changes['normal_B1_to5']=changes.get('normal_B1_to5',0)+1
                return ast.parse('(int(plan.candidate_budget) != 3 if seq in sources else not 1 <= int(plan.candidate_budget) <= 5)',mode='eval').body
            if text=='int(parent.candidate_budget) != 5':
                changes['parent_B1_to5']=changes.get('parent_B1_to5',0)+1
                return ast.parse('not 1 <= int(parent.candidate_budget) <= 5',mode='eval').body
            return self.generic_visit(node)
    patched=Rewrite().visit(patched)
    if changes!={'normal_B1_to5':1,'parent_B1_to5':1}:raise ValueError('Exact frozen budget anchors changed')
    marker=ast.parse('if not plans or max(int(p.candidate_budget) for p in plans) < 4:\n    return None').body[0]
    # Preserve the docstring and both original imports.
    patched.body.insert(3,marker)
    source=ast.unparse(ast.fix_missing_locations(patched))+'\n'
    compile(source,'<uninstalled-lowB-packed-select>','exec')
    # Reverse only the three declared edits, then demand the whole original AST.
    restored=ast.parse(source).body[0];restored.body.pop(3)
    class Reverse(ast.NodeTransformer):
        def visit_IfExp(self,node):
            if ast.unparse(node)=='int(plan.candidate_budget) != 3 if seq in sources else not 1 <= int(plan.candidate_budget) <= 5':
                return ast.parse('int(plan.candidate_budget) != (3 if seq in sources else 5)',mode='eval').body
            return self.generic_visit(node)
        def visit_UnaryOp(self,node):
            if ast.unparse(node)=='not 1 <= int(parent.candidate_budget) <= 5':
                return ast.parse('int(parent.candidate_budget) != 5',mode='eval').body
            return self.generic_visit(node)
    restored=Reverse().visit(restored)
    if ast.dump(restored,include_attributes=False)!=ast.dump(original,include_attributes=False):
        raise ValueError('Undeclared packed selector change')
    return source,dict(parent=str(PARENT),parent_sha256=PARENT_SHA,
        prepared_selector_sha256=hashlib.sha256(source.encode()).hexdigest(),
        changes=[*changes,'require_max_work_B_at_least4'],all_other_selector_AST_unchanged=True,
        normal_eager_frontier_prefix_parent_path_ledger_row_rawQ_guards_preserved=True,
        all_B_le3_supported=False,actual_NPU_qualification=False,installed=False)


def compile_host_selector():
    source,proof=prepared_selector_source();tree=ast.parse(PARENT.read_text())
    selection=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Selection')
    steps=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='STEPS' for t in n.targets))
    namespace=dict(dataclass=dataclass,__name__=__name__,STEPS=ast.literal_eval(steps))
    exec(compile(ast.Module(body=[selection],type_ignores=[]),str(PARENT),'exec'),namespace)
    exec(compile(source,'<uninstalled-lowB-packed-select>','exec'),namespace)
    return namespace['select'],proof
