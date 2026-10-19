"""Exclusive bounded own-process diagnostic; no models or workload metrics."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

P = Path(__file__).resolve().parent
B = P.parent
with (B / '.ROOT_NPU_EXECUTION.lock').open('a+') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    rows = []
    for dtype in ('int32', 'int64'):
        log = P / f'reference-{dtype}.log'
        with log.open('x') as out:
            proc = subprocess.Popen(['/bin/bash', str(B / 'serial-v1/runtime.sh'), '/bin/bash', str(P / 'probe_reference318.sh'), dtype], stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            start = time.monotonic()
            try:
                code = proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
                raise
        row = dict(dtype=dtype, exit_code=code, seconds=time.monotonic()-start, log=str(log))
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(schema='ROOT318.native-api-slot-dtype-isolation.v1',rows=rows,models_loaded=False,performance=False)
    (P / 'REFERENCE_API_DIAGNOSTIC.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
