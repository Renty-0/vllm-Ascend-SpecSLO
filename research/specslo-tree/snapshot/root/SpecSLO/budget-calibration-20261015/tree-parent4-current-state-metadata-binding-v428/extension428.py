"""Unchanged425 implementation, with an explicit finite reference correction."""
from pathlib import Path
import extension425 as original
from projection425 import project_function

prepare_install=original.prepare_install
install_entry=original.install_entry
after_install=original.after_install
cache_snapshot=original.cache_snapshot
postfull_AR=original.postfull_AR
snapshot=original.snapshot
bucket_snapshot=original.bucket_snapshot


def loaded_origins():
    from check428 import ready
    value=ready()
    here=Path(__file__).resolve().parent
    return dict(original.loaded_origins(),ROOT428_reference_selector_sources={
        p:h for p,h in value['files'].items()if Path(p).parent==here})


def qualifier():
    return project_function(original.qualify_component,(
        ('from bounded425 import run','from bounded428 import run'),))


def qualify_component(*args,**kwargs):
    return qualifier()(*args,**kwargs)


def metadata_snapshot(engine):
    value=engine._root428_reference_metadata_binding
    if value['current_cases']!=75 or (not engine.is_draft and value['actual_target_metadata_matches']<=0):
        raise RuntimeError('ROOT428 all current cases and actual target metadata matches required')
    return value
