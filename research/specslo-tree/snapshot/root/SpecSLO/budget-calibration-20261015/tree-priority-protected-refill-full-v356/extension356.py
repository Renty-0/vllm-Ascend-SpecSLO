"""Fresh original348 numerical qualification; only admission source changes."""
import extension348 as parent


prepare_install = parent.prepare_install
install_entry = parent.install_entry
after_install = parent.after_install
qualify_component = parent.qualify_component
cache_snapshot = parent.cache_snapshot
postfull_AR = parent.postfull_AR
snapshot = parent.snapshot


def loaded_origins():
    from check356 import ready
    from pathlib import Path
    value = ready()
    local = {path: digest for path, digest in value['files'].items()
             if Path(path).parent.name in ('tree-priority-protected-refill-full-v356',
                                           'tree-priority-protected-live-refill-host-v354')}
    return dict(parent.loaded_origins(), ROOT356_explicit_source_inventory=local,
                ROOT356_admission_mode='protect_tight_normal', inherited_model_qualification=False)
