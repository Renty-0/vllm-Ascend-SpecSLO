"""Add a trace-retention-only fix; preserve every frozen436 neural callback."""
from pathlib import Path
import hashlib
import json
import re

B=Path('/root/SpecSLO/budget-calibration-20261015')
OLD=B/'tree-nonempty-qualified-RPC-full-v436'
P=B/'tree-refill-witness-boolean-retention-full-v437'
assert not P.exists()
P.mkdir()

def write(name,source):
    (P/name).write_text(source)

def rename(source):
    return re.sub(r'[0-9a-f]{64}|(?<!\d)436(?!\d)',
        lambda m:'437'if m.group()=='436'else m.group(),source)

for name in ('check436.py','guard436.py','launch436.sh','startup436.sh'):
    text=rename((OLD/name).read_text()).replace(str(OLD).replace('436','437'),str(P))
    if name=='check436.py':
        text=text.replace("BASE = B / 'tree-nonempty-wholeprompt-current-model-full-v434'",f"BASE = B / '{OLD.name}'")
        text=text.replace('a79c6bd86cc38230d4e3ed13e4ee35499da05ebcb6d82338317f110ef452a042','4390b843830383905243d8dfb9aa37328a2c4a0076184cb171605b6cca39e5a3')
        text=text.replace('6ff2dd9c0dc0d6f51919627f83bd6e0b06ae575b060e2f8d4cc6edb4588e73d0','e9d88c023243b1eb67cc410979b126eb9acf047f3bcad0ea1aa4f6ef2144ddc9')
        text=text.replace('FREEZE434','FREEZE436').replace('SOURCE434','SOURCE436').replace('import check434','import check436').replace('check434.ready','check436.ready')
        text=text.replace("('HOST437_RPC_RESULT.json', 'HOST437_CONSTRUCTION_RESULT.json', 'HOST437_CLEAN_STARTUP_RUN.json')","('HOST437_RETENTION_RESULT.json', 'HOST437_CONSTRUCTION_RESULT.json')")
    elif name=='guard436.py':
        text=text.replace('import guard434 as parent','import guard436 as parent').replace('parent.P / \'launch434.sh\'','parent.P / \'launch436.sh\'')
        text=text.replace('parent434=previous','parent436=previous')
    elif name=='startup436.sh':
        text=text.replace('export PYTHONPATH='+str(P)+':','export PYTHONPATH='+str(P)+':'+str(OLD)+':')
    write(name.replace('436','437'),text)

write('witness437.py','''"""Retain actual431 witnesses; require a nonempty list as a boolean predicate."""
from witness431 import current, need


def snapshot(engine, generation):
    current(engine)
    rows = engine._root431_refill_rows.get(generation)
    need(type(rows) is list and bool(rows), 'actual admissions retained for this generation')
    return list(rows)
''')

write('trace437.py','''"""Only fix the post-generation witness-retention predicate."""
import ast
import copy
from types import FunctionType
import trace432 as original
_NAMESPACE=None


def trace_source(active_window):
    before=original.trace_source(active_window)
    tree=copy.deepcopy(before)
    rows=[n for n in ast.walk(tree)if isinstance(n,ast.keyword)and n.arg=='ROOT432_refill_witnesses']
    if len(rows)!=1 or ast.unparse(rows[0].value)!="__import__('witness431').snapshot(engine, generation)":
        raise RuntimeError('ROOT437 exact original post-generation witness-retention expression')
    old=rows[0].value
    rows[0].value=ast.parse("__import__('witness437').snapshot(engine,generation)",mode='eval').body
    result=copy.deepcopy(ast.fix_missing_locations(tree))
    rows[0].value=old
    if ast.dump(tree,include_attributes=False)!=ast.dump(before,include_attributes=False):
        raise RuntimeError('ROOT437 entire original trace AST inverse')
    return result


def namespace():
    global _NAMESPACE
    if _NAMESPACE is not None:return _NAMESPACE
    parent=original.namespace()
    ns=dict(parent,trace_source=trace_source)
    for name in('make_trace','warm_to_fixedpoint','run_windows'):
        fn=parent[name]
        actual=FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)
        actual.__kwdefaults__=fn.__kwdefaults__
        ns[name]=actual
    _NAMESPACE=ns
    return ns


def warm_to_fixedpoint(*args,**kwargs):return namespace()['warm_to_fixedpoint'](*args,**kwargs)
def run_windows(*args,**kwargs):return namespace()['run_windows'](*args,**kwargs)
''')

write('extension437.py','''"""Exact436 neural implementation, no algorithm or permission changes."""
from pathlib import Path
import extension436 as original
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
source_binding=original.source_binding


def loaded_origins():
    from check437 import ready
    frozen=ready();here=Path(__file__).resolve().parent
    return dict(original.loaded_origins(),ROOT437_retention_sources={p:h for p,h in frozen['files'].items()if Path(p).parent==here})
''')
write('audit437.py','from audit436 import audit\n')
write('runtime437.py','''"""Exact436 worker, with only declared own routing and trace-retention fix."""
import ast,copy
from pathlib import Path
import sys
P=Path(__file__).resolve().parent
sys.path.insert(0,str(P.parent/'tree-nonempty-qualified-RPC-full-v436'))
import runtime436 as original
_RUNTIME=None


def source():
    before=original.source();after=before
    changes=(("extension_module='extension436'","extension_module='extension437'"),
        ("require(extension_module=='extension436'","require(extension_module=='extension437'"),
        ('from trace432 import run_windows,warm_to_fixedpoint','from trace437 import run_windows,warm_to_fixedpoint'),
        ('   from audit436 import audit','   from audit437 import audit'))
    for old,new in changes:
        if after.count(old)!=1:raise RuntimeError('ROOT437 exact436 routing anchor '+old)
        after=after.replace(old,new)
    inverse=after
    for old,new in reversed(changes):inverse=inverse.replace(new,old)
    if inverse!=before:raise RuntimeError('ROOT437 entire436 runtime inverse')
    return after


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:return _RUNTIME
    from check437 import ready
    ready();parent=original.build_runtime()
    ns=dict(parent['worker'].__globals__,__name__='ROOT437.original436-retention-fix')
    nodes=[copy.deepcopy(n)for n in ast.parse(source()).body if isinstance(n,ast.FunctionDef)and n.name in('save','worker','main')]
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),__file__+'::trace-retention-only','exec'),ns)
    current=ns['worker'];ns['worker']=worker
    _RUNTIME=dict(worker=current,main=ns['main'],save=ns['save'])
    return _RUNTIME


def worker(*args,**kwargs):return build_runtime()['worker'](*args,**kwargs)
if __name__=='__main__':build_runtime()['main']()
''')
plan=dict(schema='ROOT437.actual-refill-witness-boolean-fix.v1',
    actual436_failure='strict need(value is True) rejected a valid nonempty witness list',
    only_semantic_change='type(rows) is list and rows -> type(rows) is list and bool(rows)',
    original436_all_neural_callbacks_identity_retained=True,global_active_limit=64,
    fresh227_681_75_plus6_warm_twozero_seal_full463_all3ownAR_required=True,
    NN_permission_inherited=False,Native_flags_assigned=False,
    private_tag_MiB=256,aggregate_private_MiB=3584,runtime_seconds=18000,
    Git_only_after_confirmed_effective_improvement=True,Raw=None,Goodput=None)
write('PLAN437.json',json.dumps(plan,indent=2)+'\n')
print(P)
