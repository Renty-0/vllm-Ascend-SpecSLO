"""Request registered stacks only for the four verified live owned workers."""
import datetime,json,os,signal
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');O=B/'tree-causal80-small-cohort-role-aware-audit-full-v209';T='dynamic-root-alternate-root-v287-eager3-compatible-full463-ownAR-v1';A=O/'actual'/T

def stat(pid):
 values=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
 return int(values[1]),int(values[19])

def main():
 receipt=json.loads((O/f'receipt-{T}.json').read_text());assert receipt['status']=='running'
 parent=receipt['child']['pid'];assert b'runtime287.py'in Path(f'/proc/{parent}/cmdline').read_bytes()
 rows=[]
 for rank in range(4):
  p=A/f'rank{rank}-fault-signal-ready.json';r=json.loads(p.read_text())
  assert r['rank']==rank and r['signal_number']==int(signal.SIGUSR1)and r['registered_after_constructor']is True
  pid=r['pid'];ppid,start=stat(pid);assert start==r['proc_start_ticks']
  ancestors=set();current=pid
  while current>1 and current not in ancestors:
   ancestors.add(current);current=stat(current)[0]
  assert parent in ancestors,(pid,parent,ancestors)
  assert Path(r['stack_file']).resolve()==(A/f'rank{rank}-python-stacks.log').resolve()
  r=dict(r,stack_bytes_before=Path(r['stack_file']).stat().st_size)
  rows.append(r)
 now=datetime.datetime.now(datetime.timezone.utc);out=A/('ROOT287-stack-request-'+now.strftime('%Y%m%dT%H%M%S%f')+'.json')
 with out.open('x')as f:json.dump(dict(UTC=now.isoformat(),workers=rows,performance_result_invalidated=True,locals_requested=False),f,indent=2);f.write('\n')
 for row in rows:os.kill(row['pid'],signal.SIGUSR1)
 print(json.dumps(dict(status='requested_all4_registered_owned_stacks',request=str(out),pids=[r['pid']for r in rows])))
if __name__=='__main__':main()
