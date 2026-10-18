"""Explicit bounded live bindings; importing this module installs nothing."""
import hashlib
import json
from pathlib import Path

from diagnostic_authorization import require_bounded_scope
from original_hook_adapter import OriginalHooks, OriginalHookMissing


P = Path(__file__).resolve().parent


def forward_live_mixed(engine, layout, *, full_query_capacity=128):
    """Use the exact call prepared at the common preflight, without rebuilding.

    This is an asymmetric target-only callback. The existing mixed hidden/head
    operations may use their original target TP group; no WORLD/Gloo boundary,
    allocator mutation, diagnostic eager oracle or KV readback is added here.
    """
    require_bounded_scope()
    hooks = getattr(engine, '_v81_live_hooks', None)
    if hooks is None or engine.is_draft or full_query_capacity != 128:
        raise OriginalHookMissing('Live forward requires its target-only prepared runtime')
    descriptor = hooks._validate_cached()
    call = hooks.prepared_call
    if (call is None or call.layout is not layout or descriptor.flexible.layout is not layout
            or hooks.transactions.current.chunk != descriptor.chunk):
        raise OriginalHookMissing('Live forward lost its submitted prepared call/transaction')
    import mixed_target
    hidden = mixed_target.graph_hidden(engine, call)
    return mixed_target.split_outputs(engine, call, hidden)


def _source_and_config(engine, native):
    import torch
    import mixed_layout
    import mixed_target
    provenance = json.loads((P/'PARENT_PROVENANCE_V75.json').read_text())
    native_path, expected = next((Path(path), sha)
        for path, sha in provenance['native_contract_sources'].items()
        if path.endswith('/native_engine.py'))
    if (Path(native.__file__).resolve() != native_path.resolve()
            or hashlib.sha256(native_path.read_bytes()).hexdigest() != expected):
        raise OriginalHookMissing('Live binding requires the immutable original Native ABI')
    for module, name in ((mixed_layout, 'mixed_layout.py'), (mixed_target, 'mixed_target.py')):
        source, expected = next((Path(path), sha)
            for path, sha in provenance['reference_sources'].items()
            if path.endswith('/tree-mixed-prefill-progressive-v45/'+name))
        if (Path(module.__file__).resolve() != P/name
                or hashlib.sha256((P/name).read_bytes()).hexdigest() != expected
                or hashlib.sha256(source.read_bytes()).hexdigest() != expected):
            raise OriginalHookMissing('Live binding changed the original mixed kernel bridge: '+name)
    if (engine.config.enable_prefix_caching
            or engine.config.spec_rhythm_prefill_token_chunk_size != 0
            or engine.config.spec_rhythm_linear_full_window
            or engine.config.max_num_batched_tokens < 197
            or not isinstance(getattr(engine.model, 'output_nonfinite', None), torch.Tensor)
            or not callable(getattr(engine, '_specslo_account_real_target_plans', None))):
        raise OriginalHookMissing('Live configuration, finite flag or real-tree accounting is absent')
    # This extends the original common finite vote, without clearing its sticky
    # failure state or asserting qualification flags on the graph runner.
    mixed_target.install_native_fault_flag(native)


def make_live_runtime_hooks(engine, *, allowed_partitions, qualified_s35_scope=None):
    require_bounded_scope()
    from live_native_boundaries import LiveNativeBoundaries
    from live_runtime_hooks import LiveRuntimeHooks
    from original_chunk_binding import bind_original_v45_chunk
    from publication_primitives import publish_original_first_token
    from vllm_ascend.spec_decode.pearl import native_engine as native
    boundaries = LiveNativeBoundaries(engine)
    engine._v75_atomic_boundaries = boundaries
    try:
        boundaries.stage('live-original-source-and-config', lambda: _source_and_config(engine, native))
        native_chunk, contract = boundaries.stage('live-original-V45-chunk-binding', lambda:
            bind_original_v45_chunk(engine, native))
        boundaries.stage('live-original-allocation-tracker-install', boundaries.install_activation_tracking)
        def reject_fixture_layout(*args, **kwargs):
            raise OriginalHookMissing('Live metadata must come from the current real descriptor')
        original = OriginalHooks(
            reservation_context_all4=boundaries.reservation_context_all4,
            protected_prefix_finite_all4=boundaries.protected_prefix_finite_all4,
            begin_prefix_protection=boundaries.begin_prefix_protection,
            cycle_preflight_all4=boundaries.cycle_preflight_all4,
            stage_all4=boundaries.stage,
            draft_prompt_local=boundaries.draft_prompt_local,
            build_mixed_layout=reject_fixture_layout,
            mixed_target_forward=forward_live_mixed,
            native_chunk=native_chunk,
            commit_fence_all4=boundaries.commit_fence_all4,
            publication_preflight_all4=boundaries.publication_preflight_all4,
            original_admission_publication=lambda publication, context:
                publish_original_first_token(engine, publication, context),
            original_cache_invalidation=boundaries.original_cache_invalidation,
            close_original_bindings=boundaries.close,
            fatal_release_all=boundaries.fatal_release_all)
        hooks = LiveRuntimeHooks(engine, original,
            allocation_tracker=boundaries.allocation_tracker,
            allowed_partitions=allowed_partitions, qualified_s35_scope=qualified_s35_scope)
        engine._v81_live_hooks = hooks
        engine._v78_atomic_hooks = hooks
        hooks.boundary_events = boundaries.events
        hooks.boundary_events.append({'stage': 'original-V45-chunk-source-contract', **contract})
        return hooks
    except BaseException:
        boundaries.close()
        raise
