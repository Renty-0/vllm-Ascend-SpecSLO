"""Add a current-SLO admission policy and the missing reporting export."""
from pathlib import Path
import json,re
B=Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-refill-witness-boolean-retention-full-v437'
P=B/'tree-slo-aware-refill-complete-interface-full-v438'
assert not P.exists();P.mkdir()
def write(name,text):(P/name).write_text(text)
def renamed(source):
    return re.sub(r'[0-9a-f]{64}|(?<!\d)437(?!\d)',lambda m:'438'if m.group()=='437'else m.group(),source).replace('tree-refill-witness-boolean-retention-full-v438',P.name)
for name in('check437.py','guard437.py','launch437.sh','startup437.sh'):
    text=renamed((OLD/name).read_text())
    if name=='check437.py':
        text=text.replace("BASE = B / 'tree-nonempty-qualified-RPC-full-v436'",f"BASE = B / '{OLD.name}'")
        text=text.replace('4390b843830383905243d8dfb9aa37328a2c4a0076184cb171605b6cca39e5a3','8ecd9f97279d1ae6023e4ca14188954b34575b26940ec64d1848c8b93a99f8d8').replace('e9d88c023243b1eb67cc410979b126eb9acf047f3bcad0ea1aa4f6ef2144ddc9','1553265a38fa19caf8bcb32067e9e921627f97f9c892429d68bd891e97aa366e')
        text=text.replace('FREEZE436','FREEZE437').replace('SOURCE436','SOURCE437').replace('import check436','import check437').replace('check436.ready','check437.ready')
        text=text.replace("('HOST438_RETENTION_RESULT.json', 'HOST438_CONSTRUCTION_RESULT.json')","('HOST438_POLICY_RESULT.json', 'HOST438_COMPLETE_INTERFACE_RESULT.json')")
    elif name=='guard437.py':
        text=text.replace('import guard436 as parent','import guard437 as parent').replace("parent.P / 'launch436.sh'","parent.P / 'launch437.sh'").replace('parent436=previous','parent437=previous')
    elif name=='startup437.sh':
        text=text.replace('export PYTHONPATH='+str(P)+':','export PYTHONPATH='+str(P)+':'+str(OLD)+':')
    write(name.replace('437','438'),text)
write('admission438.py','''"""Current arrived incumbents determine the refill threshold; global cap64."""
STRICT_TPOT_MS=40.0
STRICT_LOW_WATER=16
OTHER_LOW_WATER=32


def low_water(request_params,active):
    return STRICT_LOW_WATER if any(request_params[i].slo_tpot_ms is not None
        and request_params[i].slo_tpot_ms<=STRICT_TPOT_MS for i in active)else OTHER_LOW_WATER


def audit_trace(trace):
    from admission_audit432 import audit_trace as inherited
    value=inherited(trace)
    for row in value['cohorts']:
        witness=trace['ROOT432_refill_witnesses'][row['ordinal']]
        old=witness['incumbents']
        strict=any(trace['sampling'][i]['slo_tpot_ms']is not None
            and trace['sampling'][i]['slo_tpot_ms']<=40.0 for i in old)
        if len(old)>(16 if strict else 32):
            raise RuntimeError('ROOT438 actual SLO-aware incumbent refill threshold')
    return dict(value,current_SLO_aware_refill_policy=True,strict_ms_threshold=40.0,
        strict_low_water=16,otherwise_low_water=32,global_active_limit=64,
        future_arrivals_or_outcomes_used=False)
''')
write('generation438.py','''"""Whole437 LIVE source with one current-SLO refill threshold expression."""
import ast,copy
from trace432 import generation as original


def patch(source):
    before=ast.parse(original(source));tree=copy.deepcopy(before)
    funcs=[n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='admit_available']
    if len(funcs)!=1:raise RuntimeError('ROOT438 one current LIVE admission')
    fn=funcs[0]
    rows=[n for n in fn.body if isinstance(n,ast.Assign)and len(n.targets)==1
        and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='_REFILL_LOW_WATER426']
    if len(rows)!=1 or ast.unparse(rows[0].value)!='32':raise RuntimeError('ROOT438 exact inherited lowwater32')
    old=rows[0].value;imp=ast.parse('from admission438 import low_water as _low_water438').body[0]
    fn.body.insert(0,imp);rows[0].value=ast.parse('_low_water438(request_params,active)',mode='eval').body
    result=ast.unparse(ast.fix_missing_locations(tree))+'\\n'
    fn.body.remove(imp);rows[0].value=old
    if ast.dump(tree,include_attributes=False)!=ast.dump(before,include_attributes=False):raise RuntimeError('ROOT438 complete LIVE inverse after one scheduling expression')
    return result
''')
write('scope438.py','''"""The same actual432 NN scope; truthfully describe current scheduling."""
from scope432 import Scope,issue
from scope432 import report as original


def report(scope,engine):
    value=original(scope,engine)
    value.update(schema='ROOT438.current-SLO-refill-with-original432-NN-scope.v1',
        low_water_policy438=dict(strict_ms_threshold=40.0,strict_incumbent_limit=16,
            otherwise_incumbent_limit=32,global_active_limit=64,
            empty_admission_limit=64,minimum_nonempty8=True,maximum_nonempty32=True),
        original_NN_incumbent_domain_upper_bound32=True,future_arrivals_or_outcomes_used=False)
    return value
''')
write('trace438.py','''"""Same original437 trace code and retention; declare scheduling explicitly."""
from types import FunctionType
import trace437 as original
from generation438 import patch
from scope438 import report
_NAMESPACE=None


def namespace():
    global _NAMESPACE
    if _NAMESPACE is not None:return _NAMESPACE
    parent=original.namespace();ns=dict(parent,canonical_patch=patch,standalone_report=report)
    for name in('make_trace','warm_to_fixedpoint','run_windows'):
        fn=parent[name];actual=FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)
        actual.__kwdefaults__=fn.__kwdefaults__;ns[name]=actual
    _NAMESPACE=ns;return ns


def warm_to_fixedpoint(*a,**k):return namespace()['warm_to_fixedpoint'](*a,**k)
def run_windows(*a,**k):return namespace()['run_windows'](*a,**k)
''')
write('extension438.py','''"""Exact437 neural callbacks; preserve433's actual preflight snapshot API."""
from pathlib import Path
import extension437 as original
from extension433 import preflight_snapshot
prepare_install=original.prepare_install
install_entry=original.install_entry
after_install=original.after_install
qualify_component=original.qualify_component
cache_snapshot=original.cache_snapshot
postfull_AR=original.postfull_AR
snapshot=original.snapshot
bucket_snapshot=original.bucket_snapshot
metadata_snapshot=original.metadata_snapshot
refill_snapshot=original.refill_snapshot


def source_binding(engine):
    return dict(original.source_binding(engine),current_SLO_aware_refill438=True,
        strict_incumbent_lowwater16=True,otherwise_lowwater32=True,
        global_active_limit=64)


def loaded_origins():
    from check438 import ready
    frozen=ready();here=Path(__file__).resolve().parent
    return dict(original.loaded_origins(),ROOT438_scheduling_and_complete_interface_sources={p:h for p,h in frozen['files'].items()if Path(p).parent==here})
''')
write('runtime438.py','''"""Same437 numerical worker; explicit scheduling/reporting source routing."""
import ast,copy
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'tree-refill-witness-boolean-retention-full-v437'))
import runtime437 as original
_RUNTIME=None


def source():
    before=original.source();after=before
    changes=(("extension_module='extension437'","extension_module='extension438'"),
        ("require(extension_module=='extension437'","require(extension_module=='extension438'"),
        ('from trace437 import run_windows,warm_to_fixedpoint','from trace438 import run_windows,warm_to_fixedpoint'),
        ('from scope432 import issue,report as scope_report','from scope438 import issue,report as scope_report'),
        ('   from audit437 import audit','   from audit438 import audit'))
    for old,new in changes:
        if after.count(old)!=1:raise RuntimeError('ROOT438 exact437 routing '+old)
        after=after.replace(old,new)
    inverse=after
    for old,new in reversed(changes):inverse=inverse.replace(new,old)
    if inverse!=before:raise RuntimeError('ROOT438 entire437 worker inverse')
    return after


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:return _RUNTIME
    from check438 import ready
    ready();parent=original.build_runtime()
    ns=dict(parent['worker'].__globals__,__name__='ROOT438.current-SLO-refill-and-complete-interface')
    nodes=[copy.deepcopy(n)for n in ast.parse(source()).body if isinstance(n,ast.FunctionDef)and n.name in('save','worker','main')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::explicit-scheduling-source','exec'),ns)
    current=ns['worker'];ns['worker']=worker
    _RUNTIME=dict(worker=current,main=ns['main'],save=ns['save']);return _RUNTIME


def worker(*a,**k):return build_runtime()['worker'](*a,**k)
if __name__=='__main__':build_runtime()['main']()
''')
write('audit438.py','''"""Keep every original437 model/protocol audit; audit the tighter policy."""
import gzip,json
from pathlib import Path
from audit437 import audit as original


def audit(directory,windows):
    value=original(directory,windows)
    from admission438 import audit_trace
    directory=Path(directory);out=[]
    for path in sorted(directory.glob('**/rank1-*-actual-trace.json.gz')):
        trace=json.loads(gzip.decompress(path.read_bytes()))
        if trace['role']not in('warmup65','zero65-1','zero65-2','full463'):raise RuntimeError('ROOT438 explicit known trace role')
        out.append(dict(path=str(path),generation=trace['source_generation'],policy=audit_trace(trace)))
    if not out:raise RuntimeError('ROOT438 actual scheduling policy traces required')
    return dict(value,ROOT438_actual_SLO_aware_refill_audits=out,global_active_limit=64)
''')
plan=dict(schema='ROOT438.current-SLO-refill-and-complete-interface.v1',
    parent437_trace_observation=dict(Raw=720.9631138457082,Goodput=303.64537192205853,attained=195,requests=463,adopted=False,supervisor_complete=False),
    reason='Unconditional nonempty refills increased strict TPOT; missing report export prevented final supervision.',
    new_policy=dict(strict_TPOT_threshold_ms=40.,strict_active_low_water=16,otherwise_low_water=32,
        minimum_nonempty=8,maximum_nonempty=32,empty_admission_limit=64,global_active_limit=64,
        no_future_arrival_or_outcome_peeking=True),
    only_neural_callbacks_change=False,neural_callbacks_identity_preserved=True,
    exact_original433_preflight_snapshot_export_restored=True,
    complete_actual_worker_extension_interface_checked_before_NPU=True,
    original227_681_75_plus6_warm_twozero_seal_full463_all3ownAR_fresh_required=True,
    Native_flags_assigned=False,NN_permission_inherited=False,
    private_tag_MiB=256,aggregate_private_MiB=3584,runtime_seconds=18000,
    Raw=None,Goodput=None,new_Git_commits=0,new_Git_pushes=0)
write('PLAN438.json',json.dumps(plan,indent=2)+'\n')
for q in P.glob('*.py'):compile(q.read_text(),str(q),'exec')
print(P)
