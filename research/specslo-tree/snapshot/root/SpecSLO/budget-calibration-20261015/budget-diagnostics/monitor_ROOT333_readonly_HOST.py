"""Read current333 progress without touching model/runtime state."""
import json
from pathlib import Path
import statistics

B=Path('/root/SpecSLO/budget-calibration-20261015')
O=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='M160-actual165-chainedflush-C128-F64-currentcost-root-v333-full463-ownAR-v1'
A=O/'actual'/TAG


def read(path):return json.loads(path.read_text())

def main():
    r=read(O/f'receipt-{TAG}.json')
    result=dict(status=r.get('status'),elapsed_seconds=round(r.get('elapsed_seconds',0)),errors=r.get('errors',[]))
    if(A/'startup-progress').exists():
        startup=[]
        for p in sorted((A/'startup-progress').glob('*.json')):
            value=read(p)
            startup.append(dict(file=p.name,**{k:value[k]for k in('status','phase','rank','label','stage','last_completed','completed','total','next_M')if k in value}))
        result['startup']=startup[-12:]
    math=A/'canonical-independent-node-math'
    if math.exists():
        rows=[]
        for rank in(1,2,3):
            paths=sorted(math.glob(f'rank{rank}-canonical-case*.json'))
            last=read(paths[-1])if paths else None
            rows.append(dict(rank=rank,completed=len(paths),last_case=last['case']['name']if last else None))
        result['target_math']=rows
    costs=A/'paired-current-target-cost333'
    if costs.exists():
        values=[]
        for rank in(1,2,3):
            path=costs/f'rank{rank}-target-cost.json'
            if path.exists():
                value=read(path);last=value['costs']['rows'][-1]
                values.append(dict(rank=rank,panels=len(value['costs']['rows']),N32_B4_median_ms=statistics.median(p['milliseconds']['4']for p in last['pairs'])))
        result['current_costs']=values
    online=A/'rank0-online-state-latest.json'
    if online.exists():
        value=read(online)
        result['online']={k:value[k]for k in('generation','observations_since_process_start','last_verdict')}
    result['stage_directories']=[p.name for p in A.iterdir()if p.is_dir()]if A.exists()else[]
    files=[p for p in A.rglob('*')if p.is_file()]if A.exists()else[]
    result['actual_files']=len(files)
    result['actual_bytes']=sum(p.stat().st_size for p in files)
    result['latest_artifacts']=[str(p.relative_to(A))for p in sorted(files,key=lambda p:p.stat().st_mtime)[-4:]]
    result['error_files']=[p.name for p in A.glob('*error*')]
    result['terminal_component_reports']=[p.name for p in A.glob('rank*-local-diagnostic.json')]
    result['completed']=(A/'completed.json').exists()
    if '--brief' in __import__('sys').argv:
        result={k:v for k,v in result.items()if k in ('status','elapsed_seconds','errors','target_math','current_costs','online','actual_bytes','latest_artifacts','error_files','terminal_component_reports','completed')}
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
