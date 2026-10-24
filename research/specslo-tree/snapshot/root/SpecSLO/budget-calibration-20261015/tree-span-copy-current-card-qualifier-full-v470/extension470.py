"""Original466 allocator, draft, clocks and AR; only target branch copies change."""
from pathlib import Path
from types import FunctionType
import extension466 as original
import extension460
import extension432
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


def projected_qualifier():
    # This final projected function still executes the original227 proof.
    # It installs our original-layout copy state before the first such case.
    factory = project_function(extension460.projected_qualifier, (
        ("'from bucket460 import install'", "'from install470 import install'"),
        ('from base_component433 import qualify as base_qualify',
         'from base_component465 import qualify as base_qualify'),
    ))
    return factory()


def qualifier():
    function = extension460.qualifier
    return FunctionType(function.__code__, dict(function.__globals__,
        projected_qualifier=projected_qualifier), function.__name__,
        function.__defaults__, function.__closure__)()


def qualify_component(*args, **kwargs):
    fn = extension432.qualify_component
    body = FunctionType(fn.__code__, dict(fn.__globals__, qualifier=qualifier),
                        fn.__name__, fn.__defaults__, fn.__closure__)
    return body(*args, **kwargs)


def snapshot(engine):
    from span_install import snapshot as copy_snapshot
    return dict(original.snapshot(engine), ROOT470_span_copy=copy_snapshot(engine),
                original_B1to5_budget_allocator_unchanged=True,
                original_own_AR_implementation_unchanged=True)


def source_binding(engine):
    return dict(original.source_binding(engine), ROOT470_contiguous_KV_span_copy=True,
                original_tree_FIA_task_update_unchanged=True,
                candidate_budgets=[1,2,3,4,5], global_active_limit=64)


def loaded_origins():
    from check470 import ready
    own = ready()
    here = Path(__file__).resolve().parent
    return dict(original.loaded_origins(), ROOT470_fresh_copy_sources={p: h for p, h in own['files'].items()
        if Path(p).parent in (here, here.parent/'tree-branch-KV-span-copy-HOST-next')})
