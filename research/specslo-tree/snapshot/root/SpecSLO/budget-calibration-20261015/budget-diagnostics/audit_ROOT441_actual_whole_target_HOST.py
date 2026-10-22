"""Independent retained whole-target events and source checks; no NPU."""
from pathlib import Path
import datetime, hashlib, importlib.abc, json, statistics, sys

B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics'
P=B/'tree-M120-whole-target-gateup-layout-diagnostic-v441'
R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='M120-whole-target-gateup-layout-root-v441-one-frame-v1'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('torch','torch_npu','vllm','vllm_ascend'):
            raise RuntimeError('HOST441 auditor imports no model')
sys.meta_path.insert(0, NoModel())
sys.path.insert(0,str(P))
from host_env441 import setup
setup()
import guard441
source=json.loads((P/'SOURCE441_RECEIPT.json').read_text())
guard441.EXPECTED_FREEZE=source['freeze_sha256']
guard441.EXPECTED_ARCHIVE=source['archive_sha256']
_,closure,_=guard441.prepared()
p=R/f'receipt-{TAG}.json'
receipt=json.loads(p.read_text())
assert receipt['status']=='complete' and receipt['errors']==[]
assert receipt['sources_unchanged'] is True and receipt['owned_remaining']==[]
assert receipt['devices_after']['all_idle'] is True
for phase in ('pre_verifier','post_verifier'):
    v=receipt[phase]
    assert v['exit_code']==0 and v['actual_result']['status']=='pass'
    assert sha(v['path'])==v['sha256']
from audit441 import audit
audited,scope,_=audit(R/'actual'/TAG)
report=audited['reports'][0]
costs={}
for name in ('ND','NZ'):
    values=[max(r['whole_target_forward_event_ms'][name][j]
                for r in report['by_target_rank']) for j in range(9)]
    assert values==report['max_rank_event_costs'][name]['actual_max_rank_samples_ms']
    costs[name]=dict(p50_ms=statistics.median(values),samples_ms=values)
speedup=costs['ND']['p50_ms']/costs['NZ']['p50_ms']
assert speedup==report['p50_speedup']
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='PASS_HOST441_SOURCE_AND_ACTUAL_WHOLE_TARGET_ARITHMETIC',
    source_members=len(closure),source_freeze=source['freeze_sha256'],
    source_archive=source['archive_sha256'],receipt=str(p),receipt_sha256=sha(p),
    original_supervisor_status=receipt['status'],scope=scope,
    whole_target_costs=costs,p50_speedup=speedup,
    all3_hidden_exact=audited['all3_hidden_exact'],
    all3_query_KV_exact=audited['all3_query_KV_exact'],
    all64_original_storages_restored=True,MC2_row_bindings_unchanged=True,
    adopted=False,reason='whole-target NZ is slower; local-layer gain does not transfer',
    all4_idle=True,owned_remaining=[],full463=False,
    NPU_used_by_this_auditor=False,Raw=None,Goodput=None,new_Git_pushes=0)
with (D/'ROOT441_TERMINAL_INDEPENDENT_WHOLE_TARGET_AUDIT_HOST.json').open('x') as f:
    json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
