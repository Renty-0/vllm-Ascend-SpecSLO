"""Unchanged465 model/geometry/own-AR callbacks; declare HOST cohort48."""
from pathlib import Path
import extension465 as original

prepare_install = original.prepare_install
install_entry = original.install_entry
after_install = original.after_install
qualify_component = original.qualify_component
cache_snapshot = original.cache_snapshot
bucket_snapshot = original.bucket_snapshot
metadata_snapshot = original.metadata_snapshot
refill_snapshot = original.refill_snapshot
preflight_snapshot = original.preflight_snapshot
postfull_AR = original.postfull_AR
snapshot = original.snapshot


def source_binding(engine):
    return dict(original.source_binding(engine), maximum_new_cohort466=48,
                maximum_empty_cohort466=48, maximum_nonempty_cohort466=32,
                global_active_limit=64, all_original_model_and_AR_callbacks_retained=True)


def loaded_origins():
    from check466 import ready
    own = ready()
    here = Path(__file__).resolve().parent
    return dict(original.loaded_origins(), ROOT466_cohort_sources={p: h for p, h in own['files'].items()
                                                               if Path(p).parent == here})
