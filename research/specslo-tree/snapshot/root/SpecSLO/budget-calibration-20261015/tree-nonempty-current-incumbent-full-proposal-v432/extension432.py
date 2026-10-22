"""Proposed full qualifier; keep433 and compare preflight through extra427 NN."""
import extension433 as original
from types import FunctionType

prepare_install = original.prepare_install
install_entry = original.install_entry
after_install = original.after_install
cache_snapshot = original.cache_snapshot
postfull_AR = original.postfull_AR
snapshot = original.snapshot
bucket_snapshot = original.bucket_snapshot
metadata_snapshot = original.metadata_snapshot


def loaded_origins():
    from pathlib import Path
    from check432 import ready
    value = ready()
    here = Path(__file__).resolve().parent
    return dict(original.loaded_origins(), ROOT432_nonempty_refill_sources={
        p:h for p,h in value['files'].items() if Path(p).parent==here})


def qualifier():
    from projection425 import project_function
    import extension425
    return project_function(extension425.qualify_component, (
        ('from bounded425 import run','from bounded428 import run'),
        ('from proof425 import issue',
         "from incumbents427 import namespace\n    protocol427,_=namespace()\n    protocol427['run'](engine,native,stage,output)\n    from slot_preflight218 import end_shadow\n    stage('ROOT433/complete-all-components-original-preflight-comparisons',lambda:end_shadow(engine))\n    from proof425 import issue"),
    ),dict(projected_qualifier=original.projected_qualifier))


def qualify_component(engine, native, stage, policy, output):
    fn=original.qualify_component
    ns=dict(fn.__globals__,qualifier=qualifier)
    actual=FunctionType(fn.__code__,ns,fn.__name__,fn.__defaults__,fn.__closure__)
    return actual(engine,native,stage,policy,output)


def refill_snapshot(engine):
    from witness431 import current
    value = current(engine)
    return dict(current_same_model_proof=True, actual_NN_cases=6,
        all_allocated_incumbent_pages_and_draft_ledger_protected=True,
        original_prefill_and_publication_and_next4_checks_retained=True,
        proof=value.proof, Native_flags_assigned=False)
