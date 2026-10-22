"""Small read-only progress and completed-artifact observer; no backend import."""
from pathlib import Path
import json,gzip,datetime
B=Path('/root/SpecSLO/budget-calibration-20261015');P=B/'tree-causal80-small-cohort-role-aware-audit-full-v209';TAG='FP32-ties-target-root-v224-full463-AR-v1';A=P/'actual'/TAG

def read(p):
 raw=p.read_bytes();return json.loads(gzip.decompress(raw)if p.suffix=='.gz'else raw)
def main():
 r=read(P/('receipt-'+TAG+'.json'));out=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=r['status'],elapsed=r.get('elapsed_seconds'),errors=r.get('errors'),physical_cleanup_complete=r.get('devices_after',{}).get('all_idle'))
 for name in('prefill-rank1.json','finite-target-rank1.json','finite-draft-rank0.json'):
  p=A/'startup-progress'/name
  if p.exists():
   x=read(p);out[name]={k:v for k,v in x.items()if k in('status','completed','total','entries','count','qualified')}
 p=A/'startup-progress/finite-inventory-rank1.jsonl'
 if p.exists():
  lines=p.read_text().splitlines()
  for line in reversed(lines):
   try:x=json.loads(line)
   except json.JSONDecodeError:continue
   out['last_inventory_phase']={k:x.get(k)for k in('phase','completed','total')};break
 rows=[]
 for p in A.rglob('*'):
  try:
   if p.is_file():rows.append((p.stat().st_mtime,p))
  except FileNotFoundError:continue
 rows.sort(key=lambda row:row[0],reverse=True)
 out['latest']=[str(p.relative_to(A))for _,p in rows[:4]]
 for p in A.glob('rank*-last-error.json'):out[p.name]=read(p)
 p=A/'rank1-window-checkpoint.json.gz'
 if p.exists():
  x=read(p);out['completed_window_metrics']=[dict(window=v['window'],rank=v['rank'],host_generate_wall_seconds=v['output']['host_generate_wall_seconds'],raw_tokens_per_second=v['output']['raw_tokens_per_second'],Native_paper=v['output']['Native_paper'],host_stream_paper=v['output']['host_stream_paper'])for v in x['metrics']]
 trace=A/'windows/window32/full463/rank1-full463-actual-trace.json.gz'
 if trace.exists():
  try:
   x=read(trace);results=x['generate_results'];out['full463_completed_artifact']=dict(requests=len(results),output_tokens=sum(len(v['completion_token_ids'])for v in results),met=sum(v['slo_attained']for v in results),elapsed_seconds=x['elapsed_seconds'],native_elapsed_seconds=x['native_elapsed_seconds'],host_generate_call_seconds=x['host_generate_call_seconds'])
   cohorts=[];candidate={v['request_id']:v['completion_token_ids']for v in results}
   for p in sorted((A/'postfull-independent-FIA-AR').glob('rank1-cohort*.json.gz')):
    try:c=read(p)
    except(OSError,EOFError,json.JSONDecodeError):continue
    refs=c['reference_completions'];cohorts.append(dict(cohort=c['ordinal'],requests=len(c['request_ids']),equal=sum(candidate[rid]==t for rid,t in zip(c['request_ids'],refs,strict=True))))
   out['completed_independent_AR_cohorts']=cohorts
  except(OSError,EOFError,json.JSONDecodeError):out['full463_artifact_write_in_progress']=True
 print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
