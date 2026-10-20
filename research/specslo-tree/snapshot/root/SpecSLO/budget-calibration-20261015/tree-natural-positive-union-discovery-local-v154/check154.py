"""Exact own HOST/source prerequisite; no inherited numerical qualification."""
import ast,json
from pathlib import Path
from scope154 import P,require,sha,digest,contract
from reuse154 import inputs,checked
CPU=P/'CPU_RESULT154.json';FREEZE=P/'FROZEN_SOURCE154.json'
def local_files():
 return {str(p):sha(p) for p in sorted(P.iterdir()) if p.is_file() and (p.suffix=='.py' or p.name in ('INPUTS154.json','FIXTURE_INPUT154.json','SCOPE154.txt'))}
def source_contract():
 values=inputs();require(all(sha(p)==h for p,h in values['files'].items()),'selected read-only sources unchanged')
 dep=values['P151_actual_conclusion'];proof=json.loads(checked(dep['path']).read_text())
 require(proof['ROOT_actual_conclusion_reviewed'] is True and proof['status']=='passed_subset'
  and proof['qualification_inherited'] is False and proof['natural_positive_qualified'] is False,
  'explicit actual P151 negative prerequisite, no positive inheritance')
 for p in P.glob('*.py'):
  for n in ast.walk(ast.parse(p.read_text())):
   targets=n.targets if isinstance(n,ast.Assign) else [n.target] if isinstance(n,(ast.AnnAssign,ast.AugAssign)) else []
   require(not any(isinstance(t,ast.Attribute) and t.attr in ('runtime_validated','validated_real_row_count') for t in targets),
    'no Native qualification flag assignment')
 from provider154 import fixture,source_contexts
 from scope154 import case
 f=fixture();require(f['runtime_uses_no_post_root_target_outputs'] is True and len(f['real_source_prefixes'])==8,'retained-prefix-only fixture')
 for row in f['real_source_prefixes']:
  require(len(row['source_token_ids_through_current_root'])==129 and row['source_token_ids_through_current_root']==row['prompt_token_ids']+row['actual_previous_completion_token_ids']
   and not any('next_delivered' in k or 'available_actual' in k for k in row),'post-root actual target output not a runtime input')
 rows=source_contexts(case());require(len(rows)==64 and sum(r['V154_source_kind']=='retained-full463-p128-input-prefix' for r in rows)==16,'exact16 candidate +16 original eager negative owner allocation')
 return dict(scope=contract(),external_files=len(values['files']),P151_actual_conclusion=dep,qualification_inherited=False)
def require_ready(expected_identity=None):
 require(CPU.is_file() and FREEZE.is_file(),'own HOST proof and ROOT source freeze required')
 c=json.loads(CPU.read_text());f=json.loads(FREEZE.read_text());local=local_files();external=inputs()['files']
 require(c['status']=='passed' and c['tests_run']==6 and c['failures']==c['errors']==0
  and c['Torch_imported'] is c['NPU_initialized'] is c['model_initialized'] is False
  and c['input_hashes']==c['input_hashes_after']==f['files']==local
  and c['external_input_hashes']==c['external_input_hashes_after']==f['external_files']==external
  and f['CPU_sha256']==sha(CPU) and f['ROOT_source_reviewed'] is True,'own source/HOST/ROOT freeze identity')
 identity=dict(CPU_sha256=sha(CPU),freeze_sha256=sha(FREEZE),source_map_sha256=digest(local))
 require(expected_identity is None or expected_identity==identity,'worker exact ROOT source identity')
 return dict(identity=identity,source_contract=source_contract(),actual_device_qualification=False,formal_Goodput=False)
