"""Best257 unchanged admission and optimized draft, plus fresh target KV math."""
import extension257 as original
from projection376 import project_function

prepare_install = original.prepare_install
install_entry = original.install_entry
after_install = original.after_install
cache_snapshot = original.cache_snapshot
postfull_AR = original.postfull_AR


def loaded_origins():
    from check376 import ready
    from pathlib import Path
    frozen = ready()
    return dict(original.loaded_origins(), ROOT376_fresh_source_files={p: h
        for p, h in frozen['files'].items() if Path(p).parent == Path(__file__).parent})


def projected_qualifier():
    from bucket376 import valid_record
    return project_function(original.qualify_component, (
        ('from static_state257 import install', 'from bucket376 import install'),
        ('from scope257 import current', 'from scope376 import current'),
        ('fixed_KV_extent=list(meta.sequence_lens),actual_q_partition=list(meta.actual_seq_lengths_q),',
         "fixed_KV_extent=list(meta.sequence_lens),uniform_KV_bucket376=dict(current['uniform_KV_bucket376']),actual_q_partition=list(meta.actual_seq_lengths_q),"),
        ("and record['attention_segments']==24 and record['fixed_KV_extent']==[2560]*24",
         "and record['attention_segments']==24 and _ROOT376_valid_record(record)"),
    ), {'_ROOT376_valid_record': valid_record})


def qualify_component(engine, native, stage, policy, output):
    import inspect
    from pathlib import Path
    draft = native.NativePearlEngine.draft_tree_forward
    source = Path(__file__).parent.parent / 'tree-high-normal-cap4-native-families-v148'
    fast = inspect.getclosurevars(draft).nonlocals.get('original')
    if not (draft.__name__ == 'draft' and draft.__code__.co_filename == str(source / 'root_seed.py')
            and callable(fast) and fast.__name__ == 'draft'
            and fast.__code__.co_filename == str(source / 'fast_tree_draft_v5.py')):
        raise RuntimeError('ROOT376 actual best257 optimized RootSeed/FastV5 draft required')
    engine._best257_draft376 = draft
    component = projected_qualifier()(engine, native, stage, policy, output)
    from qualify376 import qualify
    qualify(engine, native, stage, output)
    if native.NativePearlEngine.draft_tree_forward is not draft:
        raise RuntimeError('ROOT376 original optimized draft changed during target qualification')
    return component


def snapshot(engine):
    result = original.snapshot(engine)
    result.update(fixed_KV_extent=None, maximum_KV_extent=2560,
        extent_policy='ceil(max(all original visible real and padding contexts)/512)*512',
        original_best257_admission_allocator_draft_and_branch_copy_preserved=True)
    return result


def bucket_snapshot(engine):
    return getattr(engine, '_uniform_KV_bucket376_proof', None)
