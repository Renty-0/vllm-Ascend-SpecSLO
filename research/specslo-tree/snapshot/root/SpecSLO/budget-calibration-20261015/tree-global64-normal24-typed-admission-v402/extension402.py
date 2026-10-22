"""Best376 algorithms with active64 admission and fresh target KV math."""
import extension257 as original
from projection402 import project_function

prepare_install = original.prepare_install
install_entry = original.install_entry
after_install = original.after_install
cache_snapshot = original.cache_snapshot
def postfull_AR(engine, native, stage, windows, output):
    from reference402 import run
    return run(engine, native, stage, windows, output)


def loaded_origins():
    from check402 import ready
    from pathlib import Path
    frozen = ready()
    return dict(original.loaded_origins(), ROOT402_fresh_source_files={p: h
        for p, h in frozen['files'].items() if Path(p).parent == Path(__file__).parent})


def projected_qualifier():
    from bucket402 import valid_record
    return project_function(original.qualify_component, (
        ('from static_state257 import install', 'from bucket402 import install'),
        ('from scope257 import current', 'from scope402 import current'),
        ('fixed_KV_extent=list(meta.sequence_lens),actual_q_partition=list(meta.actual_seq_lengths_q),',
         "fixed_KV_extent=list(meta.sequence_lens),uniform_KV_bucket402=dict(current['uniform_KV_bucket402']),actual_q_partition=list(meta.actual_seq_lengths_q),"),
        ("and record['attention_segments']==24 and record['fixed_KV_extent']==[2560]*24",
         "and record['attention_segments']==24 and _ROOT402_valid_record(record)"),
    ), {'_ROOT402_valid_record': valid_record})


def qualify_component(engine, native, stage, policy, output):
    import inspect
    from pathlib import Path
    draft = native.NativePearlEngine.draft_tree_forward
    source = Path(__file__).parent.parent / 'tree-high-normal-cap4-native-families-v148'
    fast = inspect.getclosurevars(draft).nonlocals.get('original')
    if not (draft.__name__ == 'draft' and draft.__code__.co_filename == str(source / 'root_seed.py')
            and callable(fast) and fast.__name__ == 'draft'
            and fast.__code__.co_filename == str(source / 'fast_tree_draft_v5.py')):
        raise RuntimeError('ROOT402 actual best257 optimized RootSeed/FastV5 draft required')
    engine._best257_draft402 = draft
    component = projected_qualifier()(engine, native, stage, policy, output)
    from qualify402 import qualify
    qualify(engine, native, stage, output)
    if native.NativePearlEngine.draft_tree_forward is not draft:
        raise RuntimeError('ROOT402 original optimized draft changed during target qualification')
    return component


def snapshot(engine):
    result = original.snapshot(engine)
    result.update(fixed_KV_extent=None, maximum_KV_extent=2560,
        extent_policy='ceil(max(all original visible real and padding contexts)/512)*512',
        original_best257_draft_allocator_and_neural_math_preserved=False,
        original_best257_draft_and_neural_math_preserved=True, actual_normal_draft_policy_limit=24, global_active_limit=64, previous_global_active_limit=64, historical_best_global_active_limit=48, new_LIVE_whole_proposal_segment_packing=True, max_target_attention_segments=24)
    return result


def bucket_snapshot(engine):
    return getattr(engine, '_uniform_KV_bucket402_proof', None)
