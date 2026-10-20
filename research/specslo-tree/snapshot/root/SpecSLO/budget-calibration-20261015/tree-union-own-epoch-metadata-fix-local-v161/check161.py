"""Own small HOST/source binding; original P154 archive checked in place."""
import ast,json
from base161 import P,require,sha,record,parent_ready
CPU=P/'CPU_RESULT161.json';FREEZE=P/'FROZEN_SOURCE161.json'
def local_files():return {str(p):sha(p) for p in sorted(P.iterdir()) if p.is_file() and (p.suffix=='.py' or p.name in ('INPUTS161.json','SCOPE161.txt'))}
def source_contract():
 parent,ready=parent_ready();inputs=record()
 require(all(sha(p)==h for p,h in inputs['files'].items()),'all direct dependency pins')
 for p in P.glob('*.py'):
  for n in ast.walk(ast.parse(p.read_text())):
   targets=n.targets if isinstance(n,ast.Assign) else [n.target] if isinstance(n,(ast.AnnAssign,ast.AugAssign)) else []
   require(not any(isinstance(t,ast.Attribute) and t.attr in ('runtime_validated','validated_real_row_count') for t in targets),'no Native flag assignment')
 return dict(parent_P154_ready=ready,parent_source_bytes_reused=True,
  same_math_parent_verdict_protocol_next=True,metadata_current_epoch_physical_exact=True,
  fresh_oracle_independent_pages=True,full_hidden_encoded_and56KV_gates_retained=True,
  full_qualification=False,two_complete_zero_traces=False,formal_Goodput=False,adopted=False)
def require_ready(expected_identity=None):
 require(CPU.is_file() and FREEZE.is_file(),'own CPU and ROOT freeze required')
 c=json.loads(CPU.read_text());f=json.loads(FREEZE.read_text());local=local_files()
 require(c['status']=='passed' and c['tests_run']==4 and c['failures']==c['errors']==0
  and c['Torch_imported'] is c['NPU_used'] is False and c['input_hashes']==c['input_hashes_after']==f['files']==local
  and f['CPU_sha256']==sha(CPU) and f['ROOT_source_reviewed'] is True,'own source/CPU/ROOT identity')
 identity=dict(CPU_sha256=sha(CPU),freeze_sha256=sha(FREEZE))
 require(expected_identity is None or expected_identity==identity,'worker current epoch identity')
 return dict(identity=identity,source_contract=source_contract(),full_qualification=False,formal_Goodput=False)
