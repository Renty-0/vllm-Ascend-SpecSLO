"""Prepare a separate uniform-q5 capacity hypothesis; no runtime installation."""
import ast,copy,datetime,hashlib,json
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');OLD=B/'tree-static-q5-M120-node-math-v255';P=B/'tree-uniform-q5-M165-S33-C128-host-v285'
assert not P.exists();P.mkdir()
modules={'projection255':'projection285','branch_tail255':'branch_tail285','padding255':'padding285','frames255':'frames285','static_state255':'static_base285','metadata255':'metadata285','mapper255':'plan285'}
proof=[]
for src,dest in [('projection255','projection285'),('branch_tail255','branch_tail285'),('static_state255','static_base285'),('frames255','frames285'),('metadata255','metadata285'),('padding255','padding285')]:
 f=OLD/(src+'.py');text=f.read_text();tree=ast.parse(text);original=ast.dump(tree,include_attributes=False);edits=[]
 for node in ast.walk(tree):
  if isinstance(node,ast.Constant)and type(node.value)is int and node.value in(120,121,24):
   edits.append((node,'value',node.value));node.value={120:165,121:166,24:33}[node.value]
  elif isinstance(node,ast.Constant)and isinstance(node.value,str):
   changed=node.value.replace('120','165').replace('121','166').replace('24','33')
   if changed!=node.value:edits.append((node,'value',node.value));node.value=changed
  elif isinstance(node,ast.ImportFrom)and node.module in modules:
   edits.append((node,'module',node.module));node.module=modules[node.module]
  elif isinstance(node,ast.alias)and node.name in modules:
   edits.append((node,'name',node.name));node.name=modules[node.name]
  elif isinstance(node,ast.Name)and node.id in modules:
   edits.append((node,'id',node.id));node.id=modules[node.id]
 # The pad's two domain imports must both use the new explicit envelope.
 if src=='padding255':
  for node in ast.walk(tree):
   if isinstance(node,ast.ImportFrom)and node.module=='target_layout':edits.append((node,'module',node.module));node.module='plan285'
 changed=ast.unparse(ast.fix_missing_locations(tree))+'\n';compile(changed,str(P/(dest+'.py')),'exec')
 for node,attribute,old in reversed(edits):setattr(node,attribute,old)
 assert ast.dump(tree,include_attributes=False)==original,src
 (P/(dest+'.py')).write_text(changed);proof.append(dict(source=str(f),source_sha256=hashlib.sha256(text.encode()).hexdigest(),output=dest+'.py',edits=len(edits),full_AST_inverse=True))
s=(B/'tree-alternate-chain-target-host-v269/layout269.py').read_text().replace('branch_tail255','branch_tail285')
(P/'layout285.py').write_text(s)
review=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='separate_HOST_source_hypothesis_only',physical_M=165,physical_S=33,uniform_q=5,KV_extent=2560,proposed_max_real_requests=32,proposed_candidate_cap=128,explicit_branch_cap=12,unchanged_copy_buffer_slots=1792,unchanged_model_precision='BF16',original_global_selected_budget192_preserved_as_future_requirement=True,
 intended_question='whether larger real batches and the existing NZ14 threshold161 can improve useful throughput; actual cost and all own-AR math must first be measured',
 pending=['CPU full Native protocol and independent metadata/copy ancestry','complete checked raw-copy binding for new physical shape','actual same-prefix neural/greedy/KV and paired whole-model cost','LIVE capacity DTO/controller/shaper changes if worthwhile','fresh full463 ownAR and benchmark'],
 current284_source_or_policy_modified=False,NPU_used=False,neural_qualified=False,runtime_installed=False,Goodput=None,adopted=False,source_specialization_proofs=proof)
(P/'SOURCE_ONLY285_REVIEW.json').write_text(json.dumps(review,indent=2)+'\n');print(P)
