"""Capture actual219 HOST checks and source snapshots; no model launcher."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

B = Path('/root/SpecSLO/budget-calibration-20261015')
P = B/'tree-cold-first-cohort-packed-KV-own-AR-full-v219'
PYTHON = '/root/miniconda3/envs/vllm-hust-dev/bin/python3.11'
attempt = P/'HOST-attempts'/str(time.time_ns())
attempt.mkdir(parents=True, exist_ok=False)
sources = {}
for path in sorted(P.iterdir()):
    if path.suffix in ('.py', '.sh') or path.name == 'PROJECTION219.json':
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        shutil.copyfile(path, attempt/path.name)
env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
results = []
for name in ('host_cold219.py', 'host219.py', 'host_slot_preflight219.py', 'host_candidate_roles219.py'):
    start = time.perf_counter()
    value = subprocess.run([PYTHON, '-B', str(P/name)], cwd=P, env=env,
                           text=True, capture_output=True, timeout=150)
    (attempt/(name+'.stdout.log')).write_text(value.stdout)
    (attempt/(name+'.stderr.log')).write_text(value.stderr)
    tail = json.loads(value.stdout.strip().splitlines()[-1]) if value.stdout.strip() else {}
    results.append(dict(name=name, exit_code=value.returncode,
                        elapsed_seconds=time.perf_counter()-start, actual=tail))
    print(json.dumps(results[-1]), flush=True)
    if value.returncode:
        break
unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == sha for p,sha in sources.items())
good = len(results) == 4 and unchanged and all(v['exit_code'] == 0
       and v['actual']['status'] == 'passed' and not v['actual']['Torch_imported'] for v in results)
result = dict(schema='ROOT219.actual-HOST-control.v1',
    UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    status='passed' if good else 'failed', suites=results,
    total_tests=sum(v['actual'].get('tests_run', 0) for v in results),
    source_sha256=sources, source_unchanged=unchanged,
    Torch_imported=False, NPU_used=False, model_launched=False,
    source_peer_review=False)
(attempt/'RESULT.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(dict(result=str(attempt/'RESULT.json'), status=result['status'], tests=result['total_tests'])), flush=True)
raise SystemExit(0 if good else 1)
