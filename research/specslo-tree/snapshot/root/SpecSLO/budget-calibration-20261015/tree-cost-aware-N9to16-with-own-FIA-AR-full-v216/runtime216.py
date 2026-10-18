"""Explicit source projection of209 runtime; worker proxy remains spawn-pickleable."""
import ast,copy,importlib.util,sys
from check216 import P,DONOR,ready,require
if str(DONOR)not in sys.path:sys.path.insert(1,str(DONOR))
_RUNTIME=None


def build_runtime():
    global _RUNTIME
    if _RUNTIME is not None:return _RUNTIME
    ready()
    source=DONOR/'runtime209.py'
    spec=importlib.util.spec_from_file_location('_ROOT216_unchanged_runtime209_globals',source)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    namespace=dict(vars(module),__name__='ROOT216.projected_runtime209')
    text=source.read_text()
    changes=(("extension_module='extension209'","extension_module='extension216'",1),
        ("require(extension_module=='extension209'","require(extension_module=='extension216'",1),
        ('  require_ready(own_identity)','  from check216 import ready as own_source_ready\n  own_source_ready()\n  require_ready(own_identity)',1),
        (' readiness=require_ready();spec=', ' from check216 import ready as own_source_ready\n own_source_ready()\n readiness=require_ready();spec=',1),
        ("     report['Native_final']=", "     report['postfull_independent_FIA_AR']=extension.postfull_AR(engine,native,stage,report['windows'],output_directory)\n     report['ROOT216_cached_parent_dispatch']=extension.cache_snapshot()\n     report['Native_final']=",1))
    for before,after,count in changes:
        require(text.count(before)==count,'exact runtime209 projection anchor '+before);text=text.replace(before,after)
    names={'save','worker','main'}
    nodes=[copy.deepcopy(n)for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name in names]
    require({n.name for n in nodes}==names,'all original runtime definitions')
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(P/'runtime216.py')+'::explicit209-projection','exec'),namespace)
    actual_worker=namespace['worker'];namespace['worker']=worker
    _RUNTIME=dict(worker=actual_worker,main=namespace['main'],source_projection=changes)
    return _RUNTIME


def worker(*args,**kwargs):return build_runtime()['worker'](*args,**kwargs)


if __name__=='__main__':build_runtime()['main']()
