"""Original438 full lifecycle, with a declared additional adaptive target proof."""
from types import FunctionType
from pathlib import Path
import extension438 as original
import extension425
import extension432
from projection425 import project_function

prepare_install=original.prepare_install
install_entry=original.install_entry
after_install=original.after_install
cache_snapshot=original.cache_snapshot
bucket_snapshot=original.bucket_snapshot
metadata_snapshot=original.metadata_snapshot
refill_snapshot=original.refill_snapshot
preflight_snapshot=original.preflight_snapshot

def projected_qualifier():
    extra=(
        ("record['M']==120","record['M'] in(80,120)"),
        ("record['primary_side_groups']==record['N']+record['branches']<=24",
         "record['primary_side_groups']==record['N']+record['branches']<=record['M']//5"),
        ("and record['actual_q_partition']==list(range(5,121,5))",
         "and record['actual_q_partition']==list(range(5,record['M']+1,5))"),
    )
    anchor="    ), {'_ROOT425_valid_record': valid_record})"
    insertion=''.join('        '+repr(pair)+',\n'for pair in extra)+anchor
    factory=project_function(extension425.projected_qualifier,(
        ('from bucket425 import valid_record','from bucket460 import valid_record'),
        ('from base_component425 import qualify as base_qualify','from base_component433 import qualify as base_qualify'),
        ('from bucket425 import install','from bucket460 import install'),
        ('from scope425 import current','from scope460 import current'),
        ("and record['attention_segments']==24 and _ROOT425_valid_record(record)",
         "and record['attention_segments']==record['M']//5 and _ROOT425_valid_record(record)"),
        (anchor,insertion),
    ))
    return factory()

def qualifier():
    injected='from qualify460 import qualify as _qualify460\n    _qualify460(engine,native,stage,output)\n    from bounded428 import run'
    factory=project_function(extension432.qualifier,(
        (repr('from bounded428 import run'),repr(injected)),
        ('dict(projected_qualifier=original.projected_qualifier)',
         'dict(projected_qualifier=_root460_projected_qualifier)'),
    ),dict(_root460_projected_qualifier=projected_qualifier))
    return factory()

def qualify_component(*args,**kwargs):
    fn=extension432.qualify_component
    body=FunctionType(fn.__code__,dict(fn.__globals__,qualifier=qualifier),fn.__name__,fn.__defaults__,fn.__closure__)
    return body(*args,**kwargs)

def postfull_AR(engine,native,stage,windows,output):
    from reference460 import run
    return run(engine,native,stage,windows,output)

def snapshot(engine):
    value=original.snapshot(engine)
    return dict(value,physical_queries=None,physical_query_families=[80,120],physical_attention_groups=[16,24],
                adaptive_policy='small80 exactly when all current groups and queries fit; otherwise original120',
                global_active_limit=64,normal_and_single_target_limit=24,
                adaptive_same_instance_proof=getattr(engine,'_adaptive460_proof',None))

def source_binding(engine):
    return dict(original.source_binding(engine),current_adaptive460=True,global_active_limit=64,
                ordinary_prompt_prefill_unchanged=True,production_permission_inherited=False)

def loaded_origins():
    from check460 import ready
    frozen=ready();here=Path(__file__).resolve().parent
    return dict(original.loaded_origins(),ROOT460_adaptive_own_history_sources={p:h for p,h in frozen['files'].items()if Path(p).parent==here})
