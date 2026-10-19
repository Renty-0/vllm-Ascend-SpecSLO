"""Verify the diagnostic signal dumps stacks and leaves the owned child alive."""
import json,signal,subprocess,sys,tempfile,time
from pathlib import Path
P=Path(__file__).resolve().parent
source="import sys,time\nfrom types import SimpleNamespace\nfrom fault_handler316 import install\ninstall(SimpleNamespace(rank=99),sys.argv[1])\nprint('registered',flush=True)\ntime.sleep(20)\n"
with tempfile.TemporaryDirectory(prefix='ROOT316-fault-signal-',dir=str(P))as folder:
 p=subprocess.Popen([sys.executable,'-B','-c',source,folder],cwd=P,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
 try:
  assert p.stdout.readline().strip()=='registered'
  report=json.loads((Path(folder)/'rank99-fault-signal-ready.json').read_text());assert report['pid']==p.pid and report['signal_number']==int(signal.SIGUSR1)
  p.send_signal(signal.SIGUSR1)
  deadline=time.monotonic()+3;path=Path(report['stack_file'])
  while not path.stat().st_size and time.monotonic()<deadline:time.sleep(.01)
  text=path.read_text();assert 'Current thread'in text and'<string>'in text and p.poll()is None
  result=dict(status='HOST_registered_fault_signal_stack_and_child_liveness_passed',stack_read_back=True,child_not_terminated_by_signal=True,environment_changed=False,NPU_used=False,bytes=len(text))
 finally:
  if p.poll()is None:p.terminate()
  p.communicate(timeout=5)
with(P/'HOST_FAULT316_RESULT.json').open('x')as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
