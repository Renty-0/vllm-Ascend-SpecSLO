"""Wait for the existing job, then independently audit; never launch NN or Git."""
import datetime,hashlib,json,subprocess,sys,time
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015');D=B/'budget-diagnostics';DONOR=B/'tree-causal80-small-cohort-role-aware-audit-full-v209'
TAG='slo-aware-refill-parent4-E4-global64-root-v438-fresh75-plus6-full463-ownAR-v1'
R=DONOR/f'receipt-{TAG}.json';AUDITOR=D/'audit_ROOT438_terminal_full_HOST.py';READER=D/'read_ROOT438_common_result_HOST.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 pins={str(p):sha(p)for p in(AUDITOR,READER)}
 assert pins[str(AUDITOR)]=='c4b754d23f3146dbb22252337e3de31432d07fe2c258554f1896140df29b7e49'
 result=D/'ROOT438_TERMINAL_INDEPENDENT_FULL_RESULT.json';metrics=D/'ROOT438_COMPLETE_COMMON_RESULT.json'
 assert not result.exists()and not metrics.exists()
 request=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_pins=pins,tag=TAG,NPU_used=False,Git_commands=False)
 with(D/'ROOT438_TERMINAL_PIPELINE_REQUEST.json').open('x')as f:json.dump(request,f,indent=2);f.write('\n')
 start=time.monotonic()
 while time.monotonic()-start<18600:
  try:receipt=json.loads(R.read_text())
  except (FileNotFoundError,json.JSONDecodeError):receipt={}
  if receipt.get('status')in('complete','failed')and receipt.get('finished_UTC'):break
  time.sleep(10)
 else:raise RuntimeError('existing438 job did not publish terminal receipt in time')
 assert all(sha(p)==h for p,h in pins.items())
 argv=[sys.executable,'-B',str(AUDITOR),'--tag',TAG,'--expected-receipt-sha256',sha(R),'--output',str(result)]
 output=subprocess.run(argv,cwd=D,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=600)
 (D/'ROOT438_TERMINAL_PIPELINE_AUDIT.log').write_text(output.stdout);print(output.stdout[-3000:],flush=True)
 assert output.returncode==0
 terminal=json.loads(result.read_text())
 if receipt['status']=='complete'and terminal['status']=='terminal_completed_compute_audited'and terminal['errors']==[]:
  output=subprocess.run([sys.executable,'-B',str(READER),'--terminal-audit',str(result),'--output',str(metrics)],cwd=D,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
  (D/'ROOT438_TERMINAL_PIPELINE_READER.log').write_text(output.stdout);print(output.stdout[-3000:],flush=True)
  assert output.returncode==0
 else:print(json.dumps(dict(status='incomplete_or_failed_no_adopted_performance',supervisor=receipt['status'],audit=terminal['status'])),flush=True)
if __name__=='__main__':main()
