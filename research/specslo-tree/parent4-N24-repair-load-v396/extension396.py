"""Own canonical FastV5 factory; fresh original baseline and finite M120 proof."""
import ast
import importlib
from pathlib import Path
import sys
import extension199 as original
import extension391


def prepare_install(output):
    source=Path(original.__file__).read_text()
    before='from live199 import prepare_helpers'
    after='from live396 import prepare_helpers'
    if source.count(before)!=1:raise RuntimeError('ROOT396 exact prepare helper anchor')
    projected=source.replace(before,after)
    if projected.replace(after,before)!=source:raise RuntimeError('ROOT396 prepare inverse')
    node=next(n for n in ast.parse(projected).body if isinstance(n,ast.FunctionDef)and n.name=='prepare_install')
    namespace=dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
        __file__+'::original199-prepare-own-canonical-source','exec'),namespace)
    return namespace['prepare_install'](output)


after_install=extension391.after_install


def loaded_origins():
    from base199 import BASE,checked,sha,require
    from import_origins_v132 import KEYS
    from check396 import ready
    pins=ready()['files'];here=Path(__file__).resolve().parent;values={}
    for name in KEYS:
        module=importlib.import_module(name)
        path=Path(vars(module).get('__file__','')).resolve()
        expected=here/'fast396.generated.py'if name=='fast_tree_draft_v5'else BASE/(name+'.py')
        require(path==expected,'actual canonical396 origin '+name)
        if name=='fast_tree_draft_v5':require(pins.get(str(path))==sha(path),'own frozen396 canonical fast source')
        else:checked(path)
        values[name]=dict(path=str(path),sha256=sha(path))
    for name,module in tuple(sys.modules.items()):
        path=vars(module).get('__file__')if hasattr(module,'__dict__')else None
        if type(path)is str and Path(path).is_absolute()and Path(path).resolve().is_relative_to(here):
            path=Path(path).resolve()
            require(path.is_file()and pins.get(str(path))==sha(path),'all actual396 own loaded sources '+name)
    return values


def qualify_component(engine,native,stage,policy,output):
    from component396 import run
    return run(engine,native,stage,policy,output)
