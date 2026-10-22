"""Exact436 neural implementation, no algorithm or permission changes."""
from pathlib import Path
import extension436 as original
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
source_binding=original.source_binding


def loaded_origins():
    from check437 import ready
    frozen=ready();here=Path(__file__).resolve().parent
    return dict(original.loaded_origins(),ROOT437_retention_sources={p:h for p,h in frozen['files'].items()if Path(p).parent==here})
