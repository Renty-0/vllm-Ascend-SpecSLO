"""Bounded read-only monitoring; no Torch, device access, or artifact mutation."""
import datetime
import json
from pathlib import Path
import time
B=Path('/root/SpecSLO/budget-calibration-20261015')
DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='static-M120-B5-tail3-root-v372-full463-ownAR-v1'

def main():
    receipt=DONOR/f'receipt-{TAG}.json'
    a=json.loads(receipt.read_text())if receipt.exists()else{}
    directory=DONOR/'actual'/TAG
    recent=[]
    for p in list(directory.glob('*'))+list(directory.glob('*/*')):
        if p.is_file():recent.append((p.stat().st_mtime,str(p.relative_to(directory)),p.stat().st_size))
    recent.sort(reverse=True)
    target={str(rank):len(list((directory/'canonical-independent-node-math').glob(f'rank{rank}-canonical-case*.json')))for rank in range(4)}
    draft=directory/'draft-tail3-same-instance/math371.jsonl'
    count=sum(1 for _ in draft.open())if draft.exists()else 0
    log=DONOR/f'local-{TAG}.log'
    tail=''
    if log.exists():
        with log.open('rb')as f:
            f.seek(max(0,log.stat().st_size-2500));tail=f.read().decode(errors='replace')
    print(json.dumps(dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),tag=TAG,
        status=a.get('status'),started_UTC=a.get('started_UTC'),errors=a.get('errors'),
        target_case_checkpoint_files_by_rank=target,new_draft_math_rows=count,
        recent_files=[dict(path=name,bytes=size,age_seconds=round(time.time()-stamp,1))for stamp,name,size in recent[:8]],
        log_tail=tail),ensure_ascii=False))
if __name__=='__main__':main()
