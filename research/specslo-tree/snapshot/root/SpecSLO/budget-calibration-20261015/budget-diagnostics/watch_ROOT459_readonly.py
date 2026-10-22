"""Read only current459 case files; a completed stage is not a math pass."""
from pathlib import Path
import datetime,json
B=Path('/root/SpecSLO/budget-calibration-20261015');R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='adaptive-own-physical-root-v459-693-node-and-paired-cost-v1';A=R/'actual'/TAG
out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),actual_exists=A.exists())
f=R/f'receipt-{TAG}.json'
if f.exists():
    value=json.loads(f.read_text());out['receipt']={k:value.get(k)for k in('status','errors','sources_unchanged','owned_remaining')}
    out['child_pid']=(value.get('child')or{}).get('pid')
    if value.get('pre_verifier'):out['pre_verifier_exit']=value['pre_verifier']['exit_code']
if A.exists():
    out['dispatch_ranks']=[f.name.split('-')[0]for f in A.glob('rank*-installed-dispatch-before-model.json')]
    banks={}
    for name in('mixed','mid','short'):
        folder=A/name
        if not folder.exists():continue
        detail={}
        for rank in range(4):
            paths=sorted(folder.glob(f'rank{rank}-canonical-case*.json'))
            row=dict(retained_cases=len(paths))
            if paths and rank:
                value=json.loads(paths[-1].read_text());observation=value['observation']
                row['last_case']=value['case']['name'];row['last_exact']={k:observation[k]for k in('exact_greedy','hidden_bitwise_equal','KV_bitwise_equal','graph_eager_KV_bitwise_equal')}
            row['math_before_cost_saved']=(folder/f'rank{rank}-math-before-cost.json').exists()
            row['cost_after_math_saved']=(folder/f'rank{rank}-cost-and-prefix-protection.json').exists()
            detail[str(rank)]=row
        detail['failed_records']=[f.name for f in folder.glob('rank*-FAILED-*.json')]
        banks[name]=detail
    out['banks']=banks
    for f in A.glob('rank*-last-error.json'):out.setdefault('worker_errors',{})[f.name]=json.loads(f.read_text())
    out['completed_protocol']=(A/'completed.json').exists()
f=R/f'local-{TAG}.log'
if f.exists():
    with f.open('rb')as stream:stream.seek(max(0,f.stat().st_size-14000));tail=stream.read().decode(errors='replace')
    out['log_bytes']=f.stat().st_size;out['log_tail']=[line[:300]for line in tail.splitlines()[-5:]]
    out['fatal_tail_indicators']=[line[:350]for line in tail.splitlines()if any(v in line for v in('NPU out of memory.','Traceback','RuntimeError:','ValueError:','FileExistsError:'))][-10:]
print(json.dumps(out))
