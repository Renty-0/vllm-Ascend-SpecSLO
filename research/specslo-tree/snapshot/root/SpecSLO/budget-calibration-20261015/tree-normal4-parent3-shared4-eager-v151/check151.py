"""Source-only readiness; HOST preparation never authorizes actual execution."""
import ast,json
from pathlib import Path
from scope151 import P,require,sha,digest,source_contract as scope
from originals151 import pins,checked
CPU=P/'CPU_RESULT151.json';FREEZE=P/'FROZEN_SOURCE151.json'

def local_files():
 return {str(p):sha(p) for p in sorted(P.iterdir()) if p.is_file() and (p.suffix=='.py' or p.name=='PARENT_FILES151.json')}
def source_contract():
 external=pins();require(all(Path(p).is_file() and sha(p)==h for p,h in external.items()),'pinned original source bytes unchanged')
 for path in P.glob('*.py'):
  for n in ast.walk(ast.parse(path.read_text())):
   targets=(n.targets if isinstance(n,ast.Assign) else [n.target] if isinstance(n,(ast.AnnAssign,ast.AugAssign)) else [])
   require(not any(isinstance(t,ast.Attribute) and t.attr in ('runtime_validated','validated_real_row_count') for t in targets),'no Native qualification flag assigned '+path.name)
 record=json.loads((P/'PARENT_FILES151.json').read_text())
 require('verification_qualification' not in record['runtime_load_order'],'explicit all4 target qualification replaces role-local multi-TP wrapper')
 require(set(record['runtime_load_order'])<=set(record['module_origins']),'all canonical helper origins explicit')
 require('controlled_successors' not in (P/'protocol151.py').read_text() and 'valid_promotion' not in (P/'protocol151.py').read_text(),'no synthetic verifier outcomes')
 require('tuple(seen[\'draft\'][0][\'encoded\'].shape)==(64,4,2)' in (P/'draft151.py').read_text(),'actual4step raw8 tensor shape gate')
 tree=ast.parse((P/'runtime151.py').read_text());worker=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
 calls=[ast.unparse(n.func) for n in ast.walk(worker) if isinstance(n,ast.Call)]
 require(calls.count('native.NativePearlEngine')==1 and calls.index('install')<calls.index('native.NativePearlEngine'),'own runtime one Native constructor after hooks')
 require('common-complete-trace-Native-delta' in (P/'actor151.py').read_text(),'role-local final Native guard remains in all4 boundary')
 return dict(scope=scope(),frozen_parent_local_and_external_files=len(external),runtime_helper_origins=record['module_origins'],
  no_parent_globals_mutation=True,no_frozen_source_edit=True,qualification_inherited=False,formal_Goodput=False)
def require_ready(expected_identity=None):
 require(CPU.is_file() and FREEZE.is_file(),'ROOT-reviewed source/HOST freeze required before any model initialization')
 c=json.loads(CPU.read_text());f=json.loads(FREEZE.read_text());local=local_files();external=pins()
 require(c['status']=='passed' and c['tests_run']>0 and c['Torch_imported'] is c['NPU_initialized'] is c['model_initialized'] is False
  and c['input_hashes']==c['input_hashes_after']==f['files']==local and c['external_input_hashes']==c['external_input_hashes_after']==f['external_files']==external
  and f['CPU_sha256']==sha(CPU) and f['ROOT_source_reviewed'] is True,'own exact final HOST/source identity; no foreign receipt qualification')
 identity=dict(CPU_sha256=sha(CPU),freeze_sha256=sha(FREEZE),source_map_sha256=digest(local))
 require(expected_identity is None or expected_identity==identity,'worker exact ROOT frozen source identity')
 return dict(identity=identity,source_contract=source_contract(),actual_device_qualification=False,formal_Goodput=False)
