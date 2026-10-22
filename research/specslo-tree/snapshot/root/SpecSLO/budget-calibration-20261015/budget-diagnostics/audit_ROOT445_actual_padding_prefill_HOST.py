"""Audit actual retained445 frames, numerical comparisons and device costs."""
from pathlib import Path
import datetime,hashlib,importlib.abc,json,statistics,sys
B=Path('/root/SpecSLO/budget-calibration-20261015')
D=B/'budget-diagnostics';P=B/'tree-M120-owned-padding-prefill-math-diagnostic-v445'
R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='M120-N16-owned-padding-prefill-root-v445-lengths1to5-math-cost-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):
            raise RuntimeError('HOST445 audit imports no model')
sys.meta_path.insert(0,NoModel());sys.path.insert(0,str(P))
from host_env445 import setup
setup()
import guard445
source=json.loads((P/'SOURCE445_RECEIPT.json').read_text())
guard445.EXPECTED_FREEZE=source['freeze_sha256'];guard445.EXPECTED_ARCHIVE=source['archive_sha256']
_,closure,_=guard445.prepared()
p=R/f'receipt-{TAG}.json';receipt=json.loads(p.read_text())
assert receipt['status']=='complete'and receipt['errors']==[]and receipt['sources_unchanged']is True
assert receipt['owned_remaining']==[]and receipt['devices_after']['all_idle']is True
for phase in('pre_verifier','post_verifier'):
    v=receipt[phase]
    assert v['exit_code']==0 and v['actual_result']['status']=='pass'and sha(v['path'])==v['sha256']
from audit445 import audit,EXACT_KEYS
audited,scope,_=audit(R/'actual'/TAG);r=audited['reports'][0]
summaries=[]
for length in range(1,6):
    cases=[rank['cases'][length-1]for rank in r['by_target_rank']]
    summaries.append(dict(chunk_length_per_owner=length,prefill_queries=length*8,
        all3_math_exact=all(all(c[k]is True for k in EXACT_KEYS)for c in cases),
        **{k:all(c[k]is True for c in cases)for k in EXACT_KEYS},
        max_prefill_hidden_abs=max(c['prefill_hidden_max_abs']for c in cases),
        max_prefill_KV_abs=max(v['max_abs']for c in cases for v in c['query_KV']),
        baseline_target_p50_ms=r['max_rank_costs']['baseline']['p50_ms'],
        mixed_target_p50_ms=r['max_rank_costs']['chunk'+str(length)]['p50_ms']))
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='PASS_HOST445_SOURCE_AND_ACTUAL_PADDING_PREFILL_DIAGNOSTIC_AUDIT',
    source_members=len(closure),freeze=source['freeze_sha256'],archive=source['archive_sha256'],
    receipt=str(p),receipt_sha256=sha(p),original_supervisor_status='complete',
    scope=scope,summary=summaries,all3_all5_lengths_math_exact=r['all3_all5_lengths_math_exact'],
    all72_original_committed_prefixes_protected=True,all_parameter_storages_unchanged=True,
    all_six_external_graph_frames_retained=True,all4idle=True,owned_remaining=[],
    whole_workload_AR_qualified=False,production_permission=False,
    NPU_reexecuted_by_this_auditor=False,Raw=None,Goodput=None,adopted=False,Git_pushes=0)
with(D/'ROOT445_TERMINAL_INDEPENDENT_PADDING_PREFILL_AUDIT_HOST.json').open('x')as f:
    json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
