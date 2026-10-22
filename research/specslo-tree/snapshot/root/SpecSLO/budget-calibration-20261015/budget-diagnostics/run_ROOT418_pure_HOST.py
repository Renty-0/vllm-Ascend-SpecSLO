"""Pure HOST protocol checks with exact declared child dependency paths."""
import json,os,subprocess,sys
from pathlib import Path
B=Path('/root/SpecSLO/budget-calibration-20261015')
P=B/'tree-global64-S24-S30-union-serving-v418'
frozen='/root/data/specslo-baselines/nano-pearl-ascend-v1'
value=next(l.removeprefix('export PYTHONPATH=') for l in (P/'startup418.sh').read_text().splitlines() if l.startswith('export PYTHONPATH='))
for old,new in (('$repo',frozen+'/source/vllm-ascend-hust'),('$frozen',frozen),('$task_root',str(B/'tree-causal80-small-cohort-role-aware-audit-full-v209')),('${PYTHONPATH:-}','')):value=value.replace(old,new)
assert '$' not in value
env=dict(os.environ,PYTHONPATH=value,PYTHONDONTWRITEBYTECODE='1')
for name in ('check418_HOST.py','check_scope418_HOST.py','check418_capacity_HOST.py','check418_typed_HOST.py','check418_geometry_HOST.py'):
    result=subprocess.run([sys.executable,'-B',str(P/name)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
    (P/('HOST418-'+name+'.log')).write_text(result.stdout)
    print(json.dumps(dict(script=name,exit_code=result.returncode)),flush=True)
    if result.returncode:
        print(result.stdout[-3000:],flush=True)
        raise RuntimeError('actual pure HOST protocol check failed')
