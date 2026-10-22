"""Exact437 neural callbacks; preserve433's actual preflight snapshot API."""
from pathlib import Path
import extension437 as original
from extension433 import preflight_snapshot
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


def source_binding(engine):
    return dict(original.source_binding(engine),current_SLO_aware_refill438=True,
        strict_incumbent_lowwater16=True,otherwise_lowwater32=True,
        global_active_limit=64)


def loaded_origins():
    from check438 import ready
    frozen=ready();here=Path(__file__).resolve().parent
    return dict(original.loaded_origins(),ROOT438_scheduling_and_complete_interface_sources={p:h for p,h in frozen['files'].items()if Path(p).parent==here})
