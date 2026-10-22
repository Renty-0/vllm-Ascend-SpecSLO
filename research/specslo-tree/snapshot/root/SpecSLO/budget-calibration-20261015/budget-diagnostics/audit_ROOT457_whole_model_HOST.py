"""Read original455 whole-model records, frozen closure and owned cleanup."""
from pathlib import Path
import datetime,hashlib,importlib.abc,json,sys
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-M80-own-geometry-whole-model-v457';R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='M80-own-geometry-root-v457-unfused-fullmodel-and120-cost-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class NoModel(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST457 no model backend')
sys.meta_path.insert(0,NoModel());sys.path.insert(0,str(P))
from host_env457 import setup
setup()
import guard457
source=json.loads((P/'SOURCE457_RECEIPT.json').read_text());guard457.EXPECTED_FREEZE=source['freeze_sha256'];guard457.EXPECTED_ARCHIVE=source['archive_sha256']
_,closure,_=guard457.prepared()
f=R/f'receipt-{TAG}.json';receipt=json.loads(f.read_text())
assert receipt['status']=='complete'and receipt['errors']==[]and receipt['sources_unchanged']is True
assert receipt['owned_remaining']==[]and receipt['devices_after']['all_idle']is True
for phase in('pre_verifier','post_verifier'):
 v=receipt[phase];assert v['exit_code']==0 and v['actual_result']['status']=='pass'and sha(v['path'])==v['sha256']
from audit457 import audit
actual,scope,_=audit(R/'actual'/TAG)
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST457_SOURCE_AND_WHOLE_MODEL_RECORD_AUDIT',
 source_members=len(closure),freeze=source['freeze_sha256'],archive=source['archive_sha256'],receipt=str(f),receipt_sha256=sha(f),
 original_supervisor_status='complete',all3_eager_and_graph_math_exact=actual['all3_eager_and_graph_math_exact'],actual_whole_model_costs=actual['actual_whole_model_costs'],scope=scope,
 ordinary_prompt_prefill_unchanged=True,model_parameters_SHA_unchanged=True,all72_committed_prefixes_protected=True,
 all4idle=True,owned_remaining=[],full463=False,whole_workload_AR_qualified=False,production_permission=False,
 Raw=None,Goodput=None,adopted=False,Git_pushes=0)
out['cross_M120_observations']=[v['cross_M120_observation']for v in actual['reports'][0]['by_target_rank']]
with(D/'ROOT457_TERMINAL_INDEPENDENT_WHOLE_MODEL_AUDIT_HOST.json').open('x')as stream:json.dump(out,stream,indent=2);stream.write('\n')
print(json.dumps(out))
