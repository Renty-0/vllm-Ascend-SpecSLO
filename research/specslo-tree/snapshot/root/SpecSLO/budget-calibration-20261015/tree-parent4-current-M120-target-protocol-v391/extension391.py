"""Original199 install, with fresh finite current-target protocol component."""
import extension199 as original
prepare_install=original.prepare_install
after_install=original.after_install


def loaded_origins():
    from check391 import ready
    from pathlib import Path
    value=ready()
    return dict(original.loaded_origins(),ROOT391_current_sources={k:v for k,v in value['files'].items()
        if Path(k).parent.name in ('tree-parent4-current-M120-target-protocol-v391',
            'tree-parent4-best376-target-binding-host-v390')})


def qualify_component(engine,native,stage,policy,output):
    from component391 import run
    return run(engine,native,stage,policy,output)
