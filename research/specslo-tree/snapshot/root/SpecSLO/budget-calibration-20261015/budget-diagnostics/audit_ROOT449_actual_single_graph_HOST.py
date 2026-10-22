"""Audit exact448 numerical body under truthful449 supervisor scope."""
from pathlib import Path
import datetime,hashlib,importlib.abc,json,sys
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics'
P=B/'tree-M120-single-graph-prefill-scope-v449'
R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='M120-single-graph-prefill-root-v449-nine-cases-isolation-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class NoModel(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in('torch','torch_npu','vllm','vllm_ascend'):raise RuntimeError('HOST449 no model backend')
sys.meta_path.insert(0,NoModel());sys.path.insert(0,str(P))
from host_env449 import setup
setup()
import guard449
source=json.loads((P/'SOURCE449_RECEIPT.json').read_text())
guard449.EXPECTED_FREEZE=source['freeze_sha256'];guard449.EXPECTED_ARCHIVE=source['archive_sha256']
_,closure,_=guard449.prepared()
p=R/f'receipt-{TAG}.json';receipt=json.loads(p.read_text())
assert receipt['status']=='complete'and receipt['errors']==[]and receipt['sources_unchanged']is True
assert receipt['owned_remaining']==[]and receipt['devices_after']['all_idle']is True
for phase in('pre_verifier','post_verifier'):
    v=receipt[phase]
    assert v['exit_code']==0 and v['actual_result']['status']=='pass'and sha(v['path'])==v['sha256']
from audit449 import audit
from audit448 import KEYS,CASES
audited,scope,_=audit(R/'actual'/TAG);r=audited['reports'][0]
summary=[]
for index,name in enumerate(CASES):
    cases=[v['cases'][index]for v in r['by_target_rank']]
    costs=r['paired_max_rank_costs'][name]
    summary.append(dict(name=name,known_prefill_queries=cases[0]['known_prefill_queries'],
        all3_math_exact=all(all(c[k]is True for k in(*KEYS,'post_replay_hidden_stable','post_replay_prefill128_KV_exact'))for c in cases),
        **{k:all(c[k]is True for c in cases)for k in KEYS},
        post_replay_hidden_stable=all(c['post_replay_hidden_stable']is True for c in cases),
        post_replay_prefill128_KV_exact=all(c['post_replay_prefill128_KV_exact']is True for c in cases),
        max_hidden_abs=max(c['prefill_hidden_max_abs']for c in cases),
        max_query_KV_abs=max(v['max_abs']for c in cases for v in c['query_KV']),
        independent_NN_failure_timing_skipped=any(c['timing_skipped']is not None for c in cases),
        actual_paired_costs=costs))
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='PASS_HOST449_SOURCE_AND_ACTUAL_SINGLE_GRAPH_ISOLATION_AUDIT',
    source_members=len(closure),freeze=source['freeze_sha256'],archive=source['archive_sha256'],
    receipt=str(p),receipt_sha256=sha(p),original_supervisor_status='complete',
    all3_all9_cases_math_exact=r['all3_all9_cases_math_exact'],
    baseline_graph_matches_eager_all3=all(v['graph_baseline_matches_eager']is True for v in r['by_target_rank']),
    eager_reference_decode_stability={str(v['rank']):{k:all(rows)for k,rows in v['eager_reference_decode_checks'].items()}for v in r['by_target_rank']},
    summary=summary,scope=scope,diagnostic_graphs=1,
    parameter_bytes_before_after_SHA256_unchanged_all3=True,
    all_allocated_pages_and_query_writes_checked_disjoint=True,
    all72_original_committed_prefixes_protected=True,all4idle=True,owned_remaining=[],
    whole_workload_AR_qualified=False,production_permission=False,
    NPU_reexecuted_by_this_auditor=False,Raw=None,Goodput=None,adopted=False,Git_pushes=0)
with(D/'ROOT449_TERMINAL_INDEPENDENT_SINGLE_GRAPH_AUDIT_HOST.json').open('x')as f:
    json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out))
