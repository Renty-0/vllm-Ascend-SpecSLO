"""Bounded read-only progress for the one owned full461 process."""
from pathlib import Path
from collections import Counter
import datetime,json
B=Path('/root/SpecSLO/budget-calibration-20261015');R=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='adaptive-own-history-global64-root-v461-retained-CANN-full463-ownAR-v1'
def snapshot():
    receipt=R/f'receipt-{TAG}.json';value=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG)
    if receipt.exists():
        row=json.loads(receipt.read_text());value.update(status=row['status'],pid=row.get('child',{}).get('pid'),finished_UTC=row.get('finished_UTC'))
    root=R/'actual'/TAG
    if root.exists():
        files=[p for p in root.rglob('*')if p.is_file()];groups=Counter();latest=[]
        for path in files:
            name=path.relative_to(root);group=name.parts[0]if len(name.parts)>1 else'root'
            groups[group]+=1;latest.append((path.stat().st_mtime_ns,str(name),path.stat().st_size))
        value.update(files=len(files),bytes=sum(p.stat().st_size for p in files),groups=dict(groups),latest=[dict(path=name,bytes=size)for _,name,size in sorted(latest,reverse=True)[:5]])
        value['errors']=[dict(path=str(p),error=json.loads(p.read_text())['error'][-4500:])for p in root.glob('rank*-last-error.json')]
    log=R/f'local-{TAG}.log'
    if log.exists():
        with log.open('rb')as stream:stream.seek(max(0,log.stat().st_size-1800));tail=stream.read().decode(errors='replace')
        value.update(log_bytes=log.stat().st_size,log_tail=tail)
    return value
if __name__=='__main__':print(json.dumps(snapshot(),ensure_ascii=False))
