"""Independent actual447 source, numerical and device-event audit; no NPU."""
from pathlib import Path
import datetime,hashlib,importlib.abc,json,sys
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-M120-paired-memory-prefill-math-diagnostic-v447'
R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='M120-paired-memory-prefill-root-v447-nine-cases-math-cost-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):
            raise RuntimeError('HOST447 no model backend')
sys.meta_path.insert(0,NoModel());sys.path.insert(0,str(P))
from host_env447 import setup
setup()
import guard447
source=json.loads((P/'SOURCE447_RECEIPT.json').read_text())
guard447.EXPECTED_FREEZE=source['freeze_sha256'];guard447.EXPECTED_ARCHIVE=source['archive_sha256']
_,closure,_=guard447.prepared()
p=R/f'receipt-{TAG}.json';receipt=json.loads(p.read_text())
assert receipt['status']=='complete'and receipt['errors']==[]and receipt['sources_unchanged']is True
assert receipt['owned_remaining']==[]and receipt['devices_after']['all_idle']is True
for phase in('pre_verifier','post_verifier'):
    v=receipt[phase]
    assert v['exit_code']==0 and v['actual_result']['status']=='pass'and sha(v['path'])==v['sha256']
from audit447 import audit,KEYS,CASES
audited,scope,_=audit(R/'actual'/TAG);r=audited['reports'][0]
summary=[]
for i,name in enumerate(CASES):
    cases=[role['cases'][i]for role in r['by_target_rank']]
    summary.append(dict(name=name,known_prefill_queries=cases[0]['known_prefill_queries'],
        physical_groups=cases[0]['physical_groups'],prefix_length=cases[0]['prefix_length'],
        all3_math_exact=all(all(c[k]is True for k in KEYS)for c in cases),
        **{k:all(c[k]is True for c in cases)for k in KEYS},
        max_hidden_abs=max(c['prefill_hidden_max_abs']for c in cases),
        max_query_KV_abs=max(v['max_abs']for c in cases for v in c['query_KV']),
        paired_baseline_target_p50_ms=r['paired_max_rank_costs'][name]['baseline_p50_ms'],
        target_p50_ms=r['paired_max_rank_costs'][name]['variant_p50_ms']))
memory=[]
for role in r['by_target_rank']:
    checks=role['memory_checkpoints']
    assert len(checks)==19 and all(len(rows)==3 and all(v['driver_free']>=3*1024**3 for v in rows)for rows in checks)
    memory.append(dict(rank=role['rank'],stages=len(checks),first=checks[0],minimum_all3_free_GiB=min(v['driver_free']for rows in checks for v in rows)/1024**3))
out=dict(memory_release_and_headroom=memory,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='PASS_HOST447_SOURCE_AND_ACTUAL_INITIAL_BULK_PREFILL_AUDIT',
    source_members=len(closure),freeze=source['freeze_sha256'],archive=source['archive_sha256'],
    receipt=str(p),receipt_sha256=sha(p),original_supervisor_status='complete',
    fresh_same_engine445_all5_math_exact=audited['fresh445_math_exact'],
    all3_all9_cases_math_exact=r['all3_all9_cases_math_exact'],
    summary=summary,baseline_target_p50_ms=r['max_rank_costs']['baseline']['p50_ms'],
    scope=scope,all72_original_committed_prefixes_protected=True,
    at_most2_bulk_diagnostic_graphs_owned_through_each_pair=True,post_pair_hidden_and_query_KV_checked=True,all_parameter_storages_unchanged=True,
    all4idle=True,owned_remaining=[],whole_workload_AR_qualified=False,
    production_permission=False,NPU_reexecuted_by_this_auditor=False,
    Raw=None,Goodput=None,adopted=False,Git_pushes=0)
with(D/'ROOT447_TERMINAL_INDEPENDENT_PAIRED_MEMORY_AUDIT_HOST.json').open('x')as f:
    json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
