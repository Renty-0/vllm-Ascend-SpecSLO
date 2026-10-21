"""Best376 algorithms with active64 admission and fresh target KV math."""
import extension257 as original
from projection409 import project_function

prepare_install = original.prepare_install
install_entry = original.install_entry
after_install = original.after_install
cache_snapshot = original.cache_snapshot
def postfull_AR(engine, native, stage, windows, output):
    from reference409 import run
    return run(engine, native, stage, windows, output)


def loaded_origins():
    from check409 import ready
    from pathlib import Path
    frozen = ready()
    return dict(original.loaded_origins(), ROOT409_fresh_source_files={p: h
        for p, h in frozen['files'].items() if Path(p).parent == Path(__file__).parent})


def projected_qualifier():
    import inspect
    from geometry409 import valid
    source = inspect.getsource(original.qualify_component)
    begin = source.index('            if not(1<=record')
    end = source.index("                raise RuntimeError('ROOT257 actual static geometry violates declared scope')", begin)
    condition = source[begin:end]
    return project_function(original.qualify_component, (
        ('from static_state257 import install', 'from bucket409 import install'),
        ('from scope257 import current', 'from scope409 import current'),
        ('record=dict(N=len(budgets),C=sum(budgets),M=result[\'query_count\'],',
         'record=dict(N=len(budgets),C=sum(budgets),M=result[\'query_count\'],budgets=list(budgets),'),
        ('fixed_KV_extent=list(meta.sequence_lens),actual_q_partition=list(meta.actual_seq_lengths_q),',
         "fixed_KV_extent=list(meta.sequence_lens),uniform_KV_bucket409=dict(current['uniform_KV_bucket409']),actual_q_partition=list(meta.actual_seq_lengths_q),"),
        (condition, '            if not _ROOT409_valid_geometry(record):\n'),
    ), {'_ROOT409_valid_geometry': valid})


def qualify_component(engine, native, stage, policy, output):
    import inspect
    from pathlib import Path
    draft = native.NativePearlEngine.draft_tree_forward
    source = Path(__file__).parent.parent / 'tree-high-normal-cap4-native-families-v148'
    fast = inspect.getclosurevars(draft).nonlocals.get('original')
    if not (draft.__name__ == 'draft' and draft.__code__.co_filename == str(source / 'root_seed.py')
            and callable(fast) and fast.__name__ == 'draft'
            and fast.__code__.co_filename == str(source / 'fast_tree_draft_v5.py')):
        raise RuntimeError('ROOT409 actual best257 optimized RootSeed/FastV5 draft required')
    engine._best257_draft409 = draft
    component = projected_qualifier()(engine, native, stage, policy, output)
    from qualify409 import qualify
    qualify(engine, native, stage, output)
    from s32_409 import qualify as qualify_wide
    qualify_wide(engine, native, stage, output)
    if native.NativePearlEngine.draft_tree_forward is not draft:
        raise RuntimeError('ROOT409 original optimized draft changed during target qualification')
    return component


def snapshot(engine):
    return dict(actual_calls_by_generation=engine._canonical257_actual_calls,
        effective_requests=32, effective_candidates=96, physical_queries=120,
        narrow=dict(N=24,C=96,S=24),wide=dict(N=32,C=88,S=32,max_B3_owners=24),
        fixed_KV_extent=None, maximum_KV_extent=2560,
        original_best257_draft_allocator_and_neural_math_preserved=False,
        original_best257_draft_and_neural_math_preserved=True,
        actual_normal_draft_policy_limit=32, global_active_limit=64,
        new_LIVE_whole_proposal_dual_route_packing=True,
        max_target_attention_segments=32)


def bucket_snapshot(engine):
    return getattr(engine, '_uniform_KV_bucket409_proof', None)
