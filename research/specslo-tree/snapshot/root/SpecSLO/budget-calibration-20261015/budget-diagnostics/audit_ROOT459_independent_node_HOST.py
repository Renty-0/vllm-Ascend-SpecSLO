"""Original459 source/receipt/cleanup and all693 own-physical case records."""
from pathlib import Path
import datetime,hashlib,importlib.abc,json,sys
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';P=B/'tree-adaptive-own-geometry-node-math-v459';R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='adaptive-own-physical-root-v459-693-node-and-paired-cost-v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):
            raise RuntimeError('HOST459 no model backend')
sys.meta_path.insert(0,NoModel());sys.path.insert(0,str(P))
from host_env459 import setup
setup()
import guard459
source=json.loads((P/'SOURCE459_RECEIPT.json').read_text());guard459.EXPECTED_FREEZE=source['freeze_sha256'];guard459.EXPECTED_ARCHIVE=source['archive_sha256'];_,closure,_=guard459.prepared()
f=R/f'receipt-{TAG}.json';receipt=json.loads(f.read_text())
assert receipt['status']=='complete'and receipt['errors']==[]and receipt['sources_unchanged']is True
assert receipt['owned_remaining']==[]and receipt['devices_after']['all_idle']is True
for phase in('pre_verifier','post_verifier'):
    value=receipt[phase];assert value['exit_code']==0 and value['actual_result']['status']=='pass'and sha(value['path'])==value['sha256']
from audit459 import audit
actual,scope,_=audit(R/'actual'/TAG)
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST459_SOURCE_AND_ALL693_OWN_PHYSICAL_NODE_RECORDS',
    original_supervisor_status='complete',original_receipt=str(f),original_receipt_sha256=sha(f),source_members=len(closure),
    source_freeze=source['freeze_sha256'],source_archive=source['archive_sha256'],
    cases_each_target=actual['cases_each_target'],independent_prefills_each_target=actual['independent_prefills_each_target'],
    paired_costs=actual['paired_costs'],independent_reference_same_physical_M=True,original_MC2_profile_unchanged=True,
    all_model_parameter_SHA_unchanged=True,global_active_ceiling=64,Native_flags_assigned=False,
    original_cross_M120_failures_preserved=True,all4idle=True,owned_remaining=[],scope=scope,
    full463=False,whole_workload_AR_qualified=False,NN_permission=False,production_permission=False,
    Raw=None,Goodput=None,adopted=False,Git_pushes=0)
with(D/'ROOT459_TERMINAL_INDEPENDENT_NODE_AUDIT_HOST.json').open('x')as stream:json.dump(out,stream,indent=2);stream.write('\n')
print(json.dumps(out))
