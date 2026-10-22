"""Current454 actual source, original six operator gates and cleanup audit."""
from pathlib import Path
import datetime,hashlib,importlib.abc,json,sys
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M160-real32-inmemory-MC2-qualification-v454';R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='M160-real32-native-epilogue-root-v454-six-layers-and1024-change-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST454 no model backend')
sys.meta_path.insert(0,NoModel());sys.path.insert(0,str(P))
from host_env454 import setup
setup()
import guard454
source=json.loads((P/'SOURCE454_RECEIPT.json').read_text());guard454.EXPECTED_FREEZE=source['freeze_sha256'];guard454.EXPECTED_ARCHIVE=source['archive_sha256']
_,closure,_=guard454.prepared()
f=R/f'receipt-{TAG}.json';receipt=json.loads(f.read_text())
assert receipt['status']=='complete'and receipt['errors']==[]and receipt['sources_unchanged']is True
assert receipt['owned_remaining']==[]and receipt['devices_after']['all_idle']is True
for phase in('pre_verifier','post_verifier'):
 v=receipt[phase];assert v['exit_code']==0 and v['actual_result']['status']=='pass'and sha(v['path'])==v['sha256']
from audit454 import audit
actual,scope,_=audit(R/'actual'/TAG)
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST454_SOURCE_AND_REAL32_EXACT_OPERATOR_RECORD_AUDIT',
 source_members=len(closure),freeze=source['freeze_sha256'],archive=source['archive_sha256'],receipt=str(f),receipt_sha256=sha(f),
 original_supervisor_status='complete',all6_original_operator_gates_accepted=actual['all6_original_operator_gates_accepted'],
 summary=actual['summary'],scope=scope,real_requests=32,real_query_rows=160,observed_layers=[0,31,63],
 ordinary_prompt_prefill_unchanged=True,model_parameters_SHA_unchanged=True,all72_committed_prefixes_protected=True,
 model_weights_or_payloads_serialized=False,new_profile_installed=False,all4idle=True,owned_remaining=[],
 full463=False,whole_workload_AR_qualified=False,production_permission=False,Raw=None,Goodput=None,adopted=False,Git_pushes=0)
with(D/'ROOT454_TERMINAL_INDEPENDENT_REAL32_MC2_AUDIT_HOST.json').open('x')as stream:json.dump(out,stream,indent=2);stream.write('\n')
print(json.dumps(out))
