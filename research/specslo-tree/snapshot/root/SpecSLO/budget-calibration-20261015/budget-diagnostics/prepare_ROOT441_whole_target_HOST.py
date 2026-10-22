"""Prepare an isolated real-frame whole-target layout diagnostic, no launch."""
from pathlib import Path
import re,json
B=Path('/root/SpecSLO/budget-calibration-20261015');OLD=B/'tree-M120-linear-layout-cost-diagnostic-v440';P=B/'tree-M120-whole-target-gateup-layout-diagnostic-v441'
assert not P.exists();P.mkdir()
def rename(t):
    return re.sub(r'[a-f0-9]{64}|(?<!\d)440(?!\d)',lambda m:'441'if m.group()=='440'else m.group(),t).replace('tree-M120-linear-layout-cost-diagnostic-v441',P.name)
# Literal source environment also includes the exact actual440 parent.
t=rename((OLD/'host_env440.py').read_text());t=t.replace("str(Path(__file__).resolve().parent)+':'+value","str(Path(__file__).resolve().parent)+':'+str(B/'tree-M120-linear-layout-cost-diagnostic-v440')+':'+value")
(P/'host_env441.py').write_text(t)
t=rename((OLD/'check440.py').read_text()).replace('check385,check438','check385,check440').replace("check438.ready(result['current438_freeze'])","check440.ready(result['current440_freeze'])")
(P/'check441.py').write_text(t)
t=(OLD/'guard440.py').read_text();t=rename(t).replace('import guard438 as current','import guard440 as current').replace("'SOURCE438_RECEIPT.json'","'SOURCE440_RECEIPT.json'")
t=t.replace('M120-ND-NZ-layout-padding-local-linear-cost','M120-all64-gateup-NZ-whole-target-cost').replace('genuine current layer0 attention/gate-up/down inputs; balanced ND/NZ and zero-row pad128 local matmul event times; no profile installed or full463','same actual M120 full target frame; all64 gate-up ND/NZ layouts, exact hidden/query-KV checks and balanced full-forward event costs; no production qualification')
(P/'guard441.py').write_text(t)
t=rename((OLD/'startup440.sh').read_text()).replace('M120-ND-NZ-layout-padding-local-linear-cost','M120-all64-gateup-NZ-whole-target-cost')
t=t.replace('export PYTHONPATH='+str(P)+':','export PYTHONPATH='+str(P)+':'+str(OLD)+':')
(P/'startup441.sh').write_text(t)
(P/'runtime441.py').write_text('''"""Complete actual440 diagnostic lifecycle, two source gates and new action."""
import ast,copy
from pathlib import Path
from host_env441 import setup
setup()
import runtime440 as original
_RUNTIME=None
ACTION='M120-all64-gateup-NZ-whole-target-cost'

def worker_source():
    before=original.worker_source();after=before
    changes=(('from check440 import ready','from check441 import ready',2),
        ('from probe440 import run','from probe441 import run',1),
        ('M120-ND-NZ-layout-padding-local-linear-cost',ACTION,2))
    for a,b,n in changes:
        if after.count(a)!=n:raise RuntimeError('ROOT441 exact440 lifecycle anchor '+a)
        after=after.replace(a,b)
    inverse=after
    for a,b,_ in reversed(changes):inverse=inverse.replace(b,a)
    if inverse!=before:raise RuntimeError('ROOT441 complete440 source inverse')
    return after

def build_runtime():
    global _RUNTIME
    if _RUNTIME is None:
        from check441 import ready
        ready();parent=original.build_runtime()
        ns=dict(parent['worker'].__globals__,__name__='ROOT441.same440-whole-target-variant')
        nodes=[copy.deepcopy(n)for n in ast.parse(worker_source()).body if isinstance(n,ast.FunctionDef)and n.name in('worker','main')]
        if len(nodes)!=2:raise RuntimeError('ROOT441 both original worker/main')
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::whole-target-layout-diagnostic','exec'),ns)
        actual=ns['worker'];ns['worker']=worker
        _RUNTIME=dict(worker=actual,main=ns['main'])
    return _RUNTIME

def worker(*a,**k):return build_runtime()['worker'](*a,**k)
if __name__=='__main__':build_runtime()['main']()
''')
(P/'probe441.py').write_text('''"""Same385 real model case/capture/prefix protection; whole-target variant."""
from projection425 import project_function
import probe385 as original
run=project_function(original.run,(
    ('from original385 import run as original','from forward441 import measure as original'),
    ('ROOT385/original128-epilogue-graph-qualifier-and-balanced-cost','ROOT441/whole-target-all64-gateup-ND-NZ-frame-and-cost'),
))
''')
plan=dict(schema='ROOT441.whole-target-gateup-layout-diagnostic.v1',scope='same real M120/S24 target frame, current ND gate-up versus NZ gate-up across all64 layers; hidden and actual query KV plus device costs',original_source_files_changed=False,Native_flags_assigned=False,transient_diagnostic_gateup_parameter_layout_swaps=True,all_original_parameter_storages_restored_before_exit=True,down_projection_and_MC2_static_weight_bindings_unchanged=True,model_or_communication_profile_permanently_installed=False,full463=False,production_permission=False,Raw=None,Goodput=None,original_caps=dict(seconds=3600,tag_MiB=192,aggregate_MiB=3584),Git_pushes=0)
(P/'PLAN441.json').write_text(json.dumps(plan,indent=2)+'\n')
print(P)
