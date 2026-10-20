"""No model entry without own HOST/ROOT freeze and explicit P151 actual dependency."""
import ast,json
from pathlib import Path
from scope152 import P,require,sha,source_contract as scope,digest
from reuse152 import record,checked,Q
CPU=P/'CPU_RESULT152.json';FREEZE=P/'FROZEN_SOURCE152.json'
def local_files():return {str(p):sha(p) for p in sorted(P.iterdir()) if p.is_file() and (p.suffix=='.py' or p.name=='PARENT_FILES152.json')}
def source_contract():
 external=record()['files'];require(all(sha(p)==h for p,h in external.items()),'selected readonly originals intact')
 for p in P.glob('*.py'):
  for n in ast.walk(ast.parse(p.read_text())):
   targets=n.targets if isinstance(n,ast.Assign) else [n.target] if isinstance(n,(ast.AnnAssign,ast.AugAssign)) else []
   require(not any(isinstance(t,ast.Attribute) and t.attr in ('runtime_validated','validated_real_row_count') for t in targets),'no Native qualification flag assignment')
 # All local FastV5 differences must be the retained exact uninstalled patch.
 original=Path(record()['module_origins']['fast_tree_draft_v5']).read_text()
 patched=(P/'fast_tree_draft_v5.py').read_text()
 import difflib
 diff=''.join(difflib.unified_diff(original.splitlines(keepends=True),patched.splitlines(keepends=True),fromfile='a/fast_tree_draft_v5.py',tofile='b/fast_tree_draft_v5.py'))
 expected=Path(P.parent/'budget-diagnostics/draft-repair-frontier-fusion-v149-source/fast_tree_draft_v5.integration.patch').read_text()
 require(diff==expected,'precise original FastV5 patch with no other algorithm changes')
 require('case.repair_vector' in (P/'draft152.py').read_text() and '[case.repair]*32' not in (P/'draft152.py').read_text(),'real per-owner partial repair vector')
 require('union_call=PA(e,ids,positions,tokens,case.union_M)' in (P/'draft152.py').read_text(),'independent original producer uses SAME physical U')
 return dict(scope=scope(),selected_external_files=len(external),exact_original_functions_reused=True,no_parent_globals_edited=True,Native_math_or_flags_changed=False,qualification_inherited=False,dynamic_prefix_cost_enabled=False)
def require_ready(expected_identity=None):
 require(CPU.is_file() and FREEZE.is_file(),'own HOST/ROOT source freeze required')
 c=json.loads(CPU.read_text());f=json.loads(FREEZE.read_text());local=local_files();external=record()['files']
 require(c['status']=='passed' and c['tests_run']==6 and c['failures']==c['errors']==0 and c['Torch_imported'] is c['NPU_initialized'] is c['model_initialized'] is False
  and c['input_hashes']==c['input_hashes_after']==f['files']==local
  and c['external_input_hashes']==c['external_input_hashes_after']==f['external_files']==external
  and f['CPU_sha256']==sha(CPU) and f['ROOT_source_reviewed'] is True,'own exact source/HOST/freeze')
 dep=f['P151_actual_dependency'];require(dep['ROOT_actual_conclusion_reviewed'] is True and dep['status'] in ('passed_component','passed_subset')
  and Path(dep['evidence_path']).is_file() and sha(dep['evidence_path'])==dep['evidence_sha256'],'P151 actual conclusion prerequisite, no qualification inherited')
 identity=dict(CPU_sha256=sha(CPU),freeze_sha256=sha(FREEZE),source_map_sha256=digest(local))
 require(expected_identity is None or expected_identity==identity,'worker exact ROOT source identity')
 return dict(identity=identity,source_contract=source_contract(),actual_device_qualification=False,formal_Goodput=False)
