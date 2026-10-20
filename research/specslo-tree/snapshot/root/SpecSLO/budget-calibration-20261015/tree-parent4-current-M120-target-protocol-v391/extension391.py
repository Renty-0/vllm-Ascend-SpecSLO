"""Original199 install, with fresh finite current-target protocol component."""
import extension199 as original
prepare_install=original.prepare_install

def after_install(native,graph,live):
    import ast
    from pathlib import Path
    source=Path(original.__file__).read_text()
    before="functions(V151/'observer151.py',ns,names=('install',));ns['install'](native,graph)"
    after="import observer392\n observer392.install(native,graph)"
    if source.count(before)!=1:raise RuntimeError('ROOT391 exact observer-registration anchor')
    changed=source.replace(before,after)
    if changed.replace(after,before)!=source:raise RuntimeError('ROOT391 complete observer delta inverse')
    node=next(n for n in ast.parse(changed).body if isinstance(n,ast.FunctionDef)and n.name=='after_install')
    namespace=dict(vars(original))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
        __file__+'::original199-after-install-with-early-observer','exec'),namespace)
    return dict(namespace['after_install'](native,graph,live),frontier_observer392_before_constructor=True)



def loaded_origins():
    from check391 import ready
    from pathlib import Path
    value=ready()
    return dict(original.loaded_origins(),ROOT391_current_sources={k:v for k,v in value['files'].items()
        if Path(k).parent.name in ('tree-parent4-current-M120-target-protocol-v391',
            'tree-parent4-best376-target-binding-host-v390',
            'tree-parent4-frontier-observer-before-capture-host-v392')})


def qualify_component(engine,native,stage,policy,output):
    from component391 import run
    return run(engine,native,stage,policy,output)
