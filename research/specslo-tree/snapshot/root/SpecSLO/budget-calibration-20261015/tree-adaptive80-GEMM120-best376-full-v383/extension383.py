"""Original257 workload, optimized draft, and new same-instance adaptive math."""
import extension257 as original
from projection383 import project_function

prepare_install=original.prepare_install
install_entry=original.install_entry
after_install=original.after_install
cache_snapshot=original.cache_snapshot
postfull_AR=original.postfull_AR


def loaded_origins():
    from check383 import ready
    from pathlib import Path
    frozen=ready()
    return dict(original.loaded_origins(),ROOT383_fresh_source_files={p:h for p,h in frozen['files'].items()
        if Path(p).parent==Path(__file__).parent})


def projected_qualifier():
    from geometry383 import valid_record
    return project_function(original.qualify_component,(
        ('from static_state257 import install','from initial383 import install'),
        ('from scope257 import current','from scope383 import current'),
        ('fixed_KV_extent=list(meta.sequence_lens),actual_q_partition=list(meta.actual_seq_lengths_q),',
         "fixed_KV_extent=list(meta.sequence_lens),uniform_KV_bucket377=dict(current['uniform_KV_bucket377']),actual_q_partition=list(meta.actual_seq_lengths_q),"),
        ("record['M']==120","record['M'] in (80,120)"),
        ("and record['attention_segments']==24 and record['fixed_KV_extent']==[2560]*24",'and _ROOT383_valid_record(record)'),
        ("record['actual_q_partition']==list(range(5,121,5))","record['actual_q_partition']==list(range(5,record['M']+1,5))"),
    ),{'_ROOT383_valid_record':valid_record})


def qualify_component(engine,native,stage,policy,output):
    import inspect
    from pathlib import Path
    draft=native.NativePearlEngine.draft_tree_forward
    source=Path(__file__).parent.parent/'tree-high-normal-cap4-native-families-v148'
    fast=inspect.getclosurevars(draft).nonlocals.get('original')
    if not(draft.__name__=='draft'and draft.__code__.co_filename==str(source/'root_seed.py')
        and callable(fast)and fast.__code__.co_filename==str(source/'fast_tree_draft_v5.py')):
        raise RuntimeError('ROOT383 actual original optimized RootSeed/FastV5 draft required')
    engine._best257_draft383=draft
    component=projected_qualifier()(engine,native,stage,policy,output)
    from qualify383 import qualify
    qualify(engine,native,stage,output)
    if native.NativePearlEngine.draft_tree_forward is not draft:raise RuntimeError('ROOT383 original draft retained')
    return component


def snapshot(engine):
    result=original.snapshot(engine)
    result.update(physical_queries=None,physical_query_families=[80,120],
        physical_primary_side_groups=None,physical_segment_families=[16,24],
        fixed_KV_extent=None,maximum_KV_extent=2560,original_allocator_draft_and_branch_copy_preserved=True)
    return result


def bucket_snapshot(engine):
    from hybrid382 import snapshot
    return dict(proof=getattr(engine,'_hybrid383_actual_proof',None),hybrid=snapshot(engine))
