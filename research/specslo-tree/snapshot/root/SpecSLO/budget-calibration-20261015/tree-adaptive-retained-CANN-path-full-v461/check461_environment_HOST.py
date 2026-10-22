"""Reproduce missing tbe and check the exact repaired startup line on CPU."""
from pathlib import Path
import datetime,hashlib,json,subprocess,sys
P=Path(__file__).resolve().parent;B=P.parent;OLD=B/'tree-adaptive-own-history-full-v460'
probe="import importlib.util,json; s=importlib.util.find_spec('tbe'); print(json.dumps(dict(tbe_origin=None if s is None else s.origin)))"
outputs=[]
for version,directory in((460,OLD),(461,P)):
    path=directory/f'startup{version}.sh'
    line=next(s for s in path.read_text().splitlines()if s.startswith('export PYTHONPATH='))
    import shlex
    code=line+'\nexec /root/miniconda3/envs/vllm-hust-dev/bin/python -B -c '+shlex.quote(probe)
    result=subprocess.run(['/bin/bash',str(B/'serial-v1/runtime.sh'),'/bin/bash','-c',code],capture_output=True,text=True,timeout=60)
    if result.returncode:raise RuntimeError('ROOT461 retained runtime CPU environment failed: '+result.stderr[-2000:])
    data=json.loads(result.stdout.strip().splitlines()[-1]);outputs.append(dict(version=version,**data))
assert outputs[0]['tbe_origin']is None
assert outputs[1]['tbe_origin']and Path(outputs[1]['tbe_origin']).resolve().is_relative_to(Path('/root/data/specslo-cann-runtime'))
before=(OLD/'startup460.sh').read_text();after=(P/'startup461.sh').read_text()
oldline=next(s for s in before.splitlines()if s.startswith('export PYTHONPATH='))
newline='export PYTHONPATH='+str(P)+':'+oldline.removeprefix('export PYTHONPATH=')+':${PYTHONPATH:-}'
inverse=after.replace(newline,oldline).replace(str(P/'check461.py'),str(OLD/'check460.py')).replace(str(P/'runtime461.py'),str(OLD/'runtime460.py'))
assert inverse==before,'ROOT461 entire startup inverse outside explicit source/check routing and inherited path preservation'
sys.path.insert(0,str(P));import guard461,extension460,extension438
ns=guard461.execution_namespace();assert ns['execute_locked'].__globals__ is ns
assert ns['MAX_SECONDS']==18000 and ns['MAX_TAG_BYTES']==256*1024**2
for path in P.glob('*.py'):compile(path.read_text(),str(path),'exec')
value=dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='PASS_HOST461_ACTUAL_RETAINED_RUNTIME_TBE_PATH_AND_STARTUP_INVERSE',
    actual_environment_probes=outputs,old_failure_reproduced=True,new_installed_tbe_resolves=True,
    entire_numerical_parent460_unchanged=True,original_full_limits_unchanged=True,
    package_installs=0,package_upgrades=0,NPU_used=False,Native_constructor_calls=0,
    NN_permission=False,Raw=None,Goodput=None)
with(P/'HOST461_ENVIRONMENT_RESULT.json').open('x')as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps(value))
