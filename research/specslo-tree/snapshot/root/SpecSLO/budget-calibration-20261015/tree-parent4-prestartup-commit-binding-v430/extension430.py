"""All428 implementation and gates retained; require original commit binding."""
from pathlib import Path
import extension428 as original

prepare_install = original.prepare_install
install_entry = original.install_entry
after_install = original.after_install
cache_snapshot = original.cache_snapshot
postfull_AR = original.postfull_AR
snapshot = original.snapshot
bucket_snapshot = original.bucket_snapshot
metadata_snapshot = original.metadata_snapshot


def loaded_origins():
    from check430 import ready
    value = ready()
    here = Path(__file__).resolve().parent
    return dict(original.loaded_origins(), ROOT430_original_commit_binding_sources={
        p: h for p, h in value['files'].items() if Path(p).parent == here})


def qualify_component(engine, *args, **kwargs):
    from lifecycle430 import snapshot as commit_snapshot
    commit_snapshot(engine)
    return original.qualify_component(engine, *args, **kwargs)
