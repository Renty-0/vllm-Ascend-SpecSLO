"""Unchanged frozen432 implementation; own source provenance only."""
from pathlib import Path
import extension432 as original

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


def loaded_origins():
    from check434 import ready
    frozen=ready();here=Path(__file__).resolve().parent
    return dict(original.loaded_origins(),ROOT434_frozen_runtime_sources={
        p:h for p,h in frozen['files'].items()if Path(p).parent==here})


def source_binding(engine):
    from witness431 import current
    current(engine)
    return dict(frozen432_implementation_unchanged=True,
        global_active_limit=64,normal_and_single_target_limit=24,
        original427_current_model_NN_required=True,
        actual_incumbent_time_charge_required=True,Native_flags_assigned=False)
