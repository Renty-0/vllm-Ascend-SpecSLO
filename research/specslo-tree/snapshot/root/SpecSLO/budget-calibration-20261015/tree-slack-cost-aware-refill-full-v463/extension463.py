"""Unchanged460 numerical callbacks; bind the separately declared host policy."""

from pathlib import Path
import extension460 as original
from admission463 import validate_model

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
    return dict(original.source_binding(engine), current_credit_cost_refill463=True,
                fixed_cost_model=validate_model(), global_active_limit=64)


def loaded_origins():
    from check463 import ready
    own = ready()
    here = Path(__file__).resolve().parent
    return dict(original.loaded_origins(), ROOT463_credit_cost_sources={p: h for p, h in own['files'].items()
                                                                     if Path(p).parent == here})
