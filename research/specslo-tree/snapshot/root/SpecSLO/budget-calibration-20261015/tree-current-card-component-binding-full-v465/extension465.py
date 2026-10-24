"""Original463 callbacks; bind current physical cards in component checks."""
from pathlib import Path
from types import FunctionType
import extension463 as original
import extension460
from projection425 import project_function

prepare_install = original.prepare_install
install_entry = original.install_entry
after_install = original.after_install
cache_snapshot = original.cache_snapshot
bucket_snapshot = original.bucket_snapshot
metadata_snapshot = original.metadata_snapshot
refill_snapshot = original.refill_snapshot
preflight_snapshot = original.preflight_snapshot
postfull_AR = original.postfull_AR
snapshot = original.snapshot


def projected_qualifier():
    return project_function(extension460.projected_qualifier, (
        ('from base_component433 import qualify as base_qualify',
         'from base_component465 import qualify as base_qualify'),
    ))()


def qualifier():
    fn = extension460.qualifier
    return FunctionType(fn.__code__, dict(fn.__globals__, projected_qualifier=projected_qualifier),
                        fn.__name__, fn.__defaults__, fn.__closure__)()


def qualify_component(*args, **kwargs):
    fn = extension460.qualify_component
    return FunctionType(fn.__code__, dict(fn.__globals__, qualifier=qualifier),
                        fn.__name__, fn.__defaults__, fn.__closure__)(*args, **kwargs)


def source_binding(engine):
    return dict(original.source_binding(engine), current_component_cards465=[0, 1, 2, 7],
                original463_admission_and_numerical_callbacks_unchanged=True)


def loaded_origins():
    from check465 import ready
    own = ready()
    here = Path(__file__).resolve().parent
    return dict(original.loaded_origins(), ROOT465_current_component_sources={p: h for p, h in own['files'].items()
                                                                          if Path(p).parent == here})
