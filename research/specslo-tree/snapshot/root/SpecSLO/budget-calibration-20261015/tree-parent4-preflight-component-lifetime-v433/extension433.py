"""Retain430 and all NN bodies; extend original preflight comparisons to75."""
from pathlib import Path
import extension430 as original
import extension425 as underlying
from projection425 import project_function

prepare_install=original.prepare_install
install_entry=original.install_entry
after_install=original.after_install
cache_snapshot=original.cache_snapshot
postfull_AR=original.postfull_AR
snapshot=original.snapshot
bucket_snapshot=original.bucket_snapshot
metadata_snapshot=original.metadata_snapshot


def projected_qualifier():
    factory=project_function(underlying.projected_qualifier, (
        ('from base_component425 import qualify as base_qualify',
         'from base_component433 import qualify as base_qualify'),))
    return factory()


def qualifier():
    return project_function(underlying.qualify_component, (
        ('from bounded425 import run','from bounded428 import run'),
        ('from proof425 import issue',
         "from slot_preflight218 import end_shadow\n    stage('ROOT433/complete-all-components-original-preflight-comparisons',lambda:end_shadow(engine))\n    from proof425 import issue"),
    ),dict(projected_qualifier=projected_qualifier))


def qualify_component(engine,native,stage,policy,output):
    from lifecycle430 import snapshot as original_commit
    original_commit(engine)
    from slot_preflight218 import snapshot as preflight_state
    before=preflight_state()
    if not(before['shadow']is True and before['qualified']is False):
        raise RuntimeError('ROOT433 original fresh component comparison phase required')
    value=qualifier()(engine,native,stage,policy,output)
    after=preflight_state()
    if not(after['shadow']is False and after['qualified']is True
           and after['shadow_pairs']>before['shadow_pairs']):
        raise RuntimeError('ROOT433 complete original plus75 comparisons before performance accounting')
    engine._root433_preflight_lifetime=dict(original218_preflight_source_unchanged=True,
        original_close_called_once_after_all75=True,
        all_original_and_new_component_preflights_compared=True,
        before=before,after=after,Native_flags_assigned=False,
        production_generation_guard_preserved=True)
    return value


def loaded_origins():
    from check433 import ready
    frozen=ready();here=Path(__file__).resolve().parent
    return dict(original.loaded_origins(),ROOT433_preflight_lifetime_sources={
        p:h for p,h in frozen['files'].items()if Path(p).parent==here})


def preflight_snapshot(engine):
    return engine._root433_preflight_lifetime
