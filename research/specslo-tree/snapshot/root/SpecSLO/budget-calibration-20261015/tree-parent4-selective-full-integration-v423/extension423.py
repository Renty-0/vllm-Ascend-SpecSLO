"""Best376 algorithms with active64 admission and fresh target KV math."""
import extension257 as original
from projection423 import project_function

def prepare_install(output):
    import importlib.util
    import sys
    from base199 import P, require
    from epoch199 import read
    require('source_epoch' not in sys.modules, 'fresh exact diagnostic source reader')
    spec = importlib.util.spec_from_file_location('source_epoch', P/'source_epoch.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules['source_epoch'] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop('source_epoch', None)
        raise
    require(module.read is read, 'retained diagnostic reader identity')
    from factory423 import prepare_helpers
    prepare_helpers()
    return original.prepare_install(output)


install_entry = original.install_entry
def after_install(native, graph, live):
    result = original.after_install(native, graph, live)
    import observer392
    observer392.install(native, graph)
    record = graph.NativeACLGraphRunner._record_execution
    def tracked(runner, kind, execution, output, *, entry_key=None, token_rows=()):
        if kind == 'target':
            runner._v151_last_target_key = entry_key
        return record(runner, kind, execution, output, entry_key=entry_key, token_rows=token_rows)
    graph.NativeACLGraphRunner._record_execution = tracked
    return dict(result, frontier_observer392_before_constructor=True,
                original_RootSeed_and_legacy_Fast_retained=True,
                explicit_selective_parent4_dispatcher=True,
                E4_production_permission=False)


cache_snapshot = original.cache_snapshot
def postfull_AR(engine, native, stage, windows, output):
    from reference423 import run
    return run(engine, native, stage, windows, output)


def loaded_origins():
    from check423 import ready
    from base199 import BASE, checked, sha, require
    from import_origins_v132 import KEYS
    import importlib
    import sys
    from pathlib import Path
    frozen = ready()
    here = Path(__file__).resolve().parent
    values = {}
    for name in KEYS:
        module = importlib.import_module(name)
        path = Path(vars(module).get('__file__', '')).resolve()
        expected = here/'factory423.py' if name == 'fast_tree_draft_v5' else BASE/(name+'.py')
        require(path == expected, 'actual canonical423 origin '+name)
        if name == 'fast_tree_draft_v5':
            require(frozen['files'].get(str(path)) == sha(path), 'own frozen dispatcher source')
        else:
            checked(path)
        values[name] = dict(path=str(path), sha256=sha(path))
    for name, module in tuple(sys.modules.items()):
        path = vars(module).get('__file__') if hasattr(module, '__dict__') else None
        if type(path) is str and Path(path).is_absolute() and Path(path).resolve().is_relative_to(here):
            path = Path(path).resolve()
            require(path.is_file() and frozen['files'].get(str(path)) == sha(path),
                    'all actual423 own loaded sources '+name)
    return dict(canonical=values, ROOT423_fresh_source_files={p: h
        for p, h in frozen['files'].items() if Path(p).parent == here})


def projected_qualifier():
    from bucket423 import valid_record
    return project_function(original.qualify_component, (
        ('original = extension218.qualify_component(engine, native, stage, policy, output)',
         'from base_component423 import qualify as base_qualify\n    original = base_qualify(engine, native, stage, policy, output)'),
        ('from static_state257 import install', 'from bucket423 import install'),
        ('from scope257 import current', 'from scope423 import current'),
        ('fixed_KV_extent=list(meta.sequence_lens),actual_q_partition=list(meta.actual_seq_lengths_q),',
         "fixed_KV_extent=list(meta.sequence_lens),uniform_KV_bucket423=dict(current['uniform_KV_bucket423']),actual_q_partition=list(meta.actual_seq_lengths_q),"),
        ("and record['attention_segments']==24 and record['fixed_KV_extent']==[2560]*24",
         "and record['attention_segments']==24 and _ROOT423_valid_record(record)"),
    ), {'_ROOT423_valid_record': valid_record})


def qualify_component(engine, native, stage, policy, output):
    from factory423 import current
    current(native)
    draft = native.NativePearlEngine.draft_tree_forward
    engine._best257_draft423 = draft
    component = projected_qualifier()(engine, native, stage, policy, output)
    from qualify423 import qualify
    qualify(engine, native, stage, output)
    from bounded423 import run
    run(engine, native, stage, output)
    from proof423 import issue
    stage('ROOT423/issue-fresh-same-engine-E4', lambda: issue(engine, native))
    if native.NativePearlEngine.draft_tree_forward is not draft:
        raise RuntimeError('ROOT423 installed dispatcher changed during qualification')
    return component


def snapshot(engine):
    result = original.snapshot(engine)
    result.update(fixed_KV_extent=None, maximum_KV_extent=2560,
        extent_policy='ceil(max(all original visible real and padding contexts)/512)*512',
        original_best257_draft_allocator_and_neural_math_preserved=False,
        original_best257_draft_and_neural_math_preserved=False,
        original_RootSeed_and_legacy_Fast_fallback_retained=True,
        selective_parent4_E4_current_proof=__import__('proof423').snapshot(engine),
        actual_normal_draft_policy_limit=24, global_active_limit=64, previous_global_active_limit=64, historical_best_global_active_limit=48, new_LIVE_whole_proposal_segment_packing=True, max_target_attention_segments=24)
    return result


def bucket_snapshot(engine):
    return getattr(engine, '_uniform_KV_bucket423_proof', None)
