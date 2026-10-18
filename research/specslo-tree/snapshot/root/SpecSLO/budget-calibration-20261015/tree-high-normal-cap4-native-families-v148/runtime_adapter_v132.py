"""Exact typed draft hooks after the original live builder; no device work."""
import ast
import copy
import hashlib
import json
from pathlib import Path
from ast_adapter_v123 import prepare_runtime_adapter,_apply_sequences,ast_digest
P=Path(__file__).resolve().parent
EXTRA_RULES=(
 ('eligible_parent_B3_B5_before_window',
  'exploration_cost = self.config.spec_rhythm_tree_width * self.config.spec_rhythm_tree_depth',
  'eager = _v132_filter_eager_parents(eager, controller, payloads, counters)\nexploration_cost = self.config.spec_rhythm_tree_width * self.config.spec_rhythm_tree_depth'),
)

def adapt_live_source(source):
    binding=json.loads((P/'BASE_RUNTIME_BINDING_V132.json').read_text())
    typed,proof=prepare_runtime_adapter(source,binding['generated_live_sha256'])
    tree=ast.parse(typed);before=ast_digest(tree)
    counts=_apply_sequences(tree,EXTRA_RULES)
    changed=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='shaper.shape':
            kw=next((kw for kw in node.keywords if kw.arg=='eager_token_cap'),None)
            if kw is None or ast.unparse(kw.value)!='effective_eager_cap or None':raise RuntimeError('V132 explicit eager cap anchor differs')
            kw.value=ast.Name(id='effective_eager_cap',ctx=ast.Load());changed.append(kw)
    if len(changed)!=1:raise RuntimeError('V132 exactly one actual caller eager cap required')
    result=ast.unparse(ast.fix_missing_locations(tree))+'\n'
    reverse=copy.deepcopy(tree)
    _apply_sequences(reverse,tuple((name,new,old) for name,old,new in EXTRA_RULES))
    for node in ast.walk(reverse):
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='shaper.shape':
            next(kw for kw in node.keywords if kw.arg=='eager_token_cap').value=ast.parse('effective_eager_cap or None',mode='eval').body
    if ast_digest(reverse)!=before:raise RuntimeError('V132 extras changed undeclared AST')
    return result

def source_contract():
    from ast_adapter_v123 import RUNTIME_RULES
    return dict(schema='V132.typed-live-exact-AST-hooks.v1',typed_rules=[dict(name=n,old=o,new=v) for n,o,v in RUNTIME_RULES],
        extra_rules=[dict(name=n,old=o,new=v) for n,o,v in EXTRA_RULES],
        eager_cap_or_None_removed_at_actual_caller=True,
        fresh_decode_shell_policy_and_clock_unchanged=True,target_admission_and_verifier_unchanged=True,
        original_live_AST_inverse_checked=True,adds_NPU_ops_or_collectives=False,
        actual_live_device_qualified=False)
