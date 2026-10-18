"""Explicit same-engine entry, called only by a future ROOT-owned worker.

No constructor, supervisor, NPU query or installation occurs on import. The
caller must retain original startup/qualification and owns process cleanup.
"""
import json
from pathlib import Path
from window160 import P, DEFAULT_WINDOWS, windows, require
from check160 import require_ready
from probe160 import make_trace
from metrics160 import full_metrics


def run_pair(engine, *, controls, scope, stage, directory, checkpoint=None,
             admission_windows=DEFAULT_WINDOWS):
    """Two independent arrivals/generations, one model and one natural seal."""
    stage('V160/own-source-ready', require_ready)
    selected_windows = windows(admission_windows)
    directory = Path(directory).resolve()
    require(directory.is_relative_to(P), 'private P160 output root')
    def create_root():
        if engine.rank == engine.topology.draft_leader_rank:
            require(not directory.exists(), 'fresh private P160 output root')
            directory.mkdir(parents=True, exist_ok=False)
    stage('V160/create-private-output-root', create_root)

    def configuration():
        c = engine.config
        require(c.max_num_seqs == 64 and c.max_num_queued_seqs == 463
                and engine._cache_storage_sequence_capacity == 527
                and tuple(engine._tree_graph_scratch_sequence_ids) == tuple(range(463, 527))
                and c.spec_rhythm_max_target_batch == 32 and c.spec_rhythm_verification_budget == 96
                and c.spec_rhythm_online_prefill is True and c.enable_continuous_batching is True
                and c.enable_prefix_caching is False and c.num_kvcache_blocks == 544
                and engine.cache_allocation is None and engine.cache_block_tables is None
                and engine.graph_runner.graph_cache_sealed is False,
                'actual configured64/queued463/storage527/scratch64 original online path; unsealed fresh cache')
        return dict(configured_capacity=64, queued=463, storage=527, scratch=64,
                    target_max=32, candidate_roof=96, admission_windows=list(selected_windows))
    config = stage('V160/actual-capacity-and-path', configuration)
    from candidate_phases_v132 import split_snapshot, split_delta, update_snapshot
    from combined_controls_v113 import proposal_counter_snapshot
    from window_cost_state_v132 import snapshot as window_snapshot
    traces, metrics = [], []
    functions = {w: make_trace(w)[0] for w in selected_windows}

    def run(active_window, role, count):
        lane = lambda name, fn: stage('V160/window'+str(active_window)+'/'+role+'/'+name, fn)
        folder = directory / ('window'+str(active_window)) / role
        lane('create-phase-output', lambda: folder.mkdir(parents=True, exist_ok=False)
             if engine.rank == engine.topology.draft_leader_rank else None)
        before = lane('controls-before', controls.observe)
        generation = getattr(engine, '_dynamic_tree_generation', 0)
        # Original reset uses the NEXT original generation. fresh_decode also
        # clears observed g samples before every new actual generate call.
        admission = lane('original-normal-admission-reset',
                         lambda: split_snapshot(engine, reset_normal_admission=True))
        proposals_before = lane('proposal-counters-before', lambda: proposal_counter_snapshot(engine))
        updates_before = lane('task-updates-before', lambda: update_snapshot(engine))
        window_before = lane('window-state-before', lambda: window_snapshot(engine))
        value = functions[active_window](engine, count=count, role=role,
                                         directory=folder, qualified_s35_scope=scope)
        def completed():
            require(value['source_generation'] == generation+1
                    and value['cache_released'] is True
                    and engine.cache_allocation is None and engine.cache_block_tables is None
                    and value['admission_window160'] == active_window
                    and value['actual_capacity64_scope_completed'] is False,
                    'own exact next generation, full cache release and truthful window scope')
            return controls.observe()
        after = lane('controls-and-generation-after', completed)
        admission_after = lane('split-and-gain-after', lambda: split_snapshot(engine))
        proposals_after = lane('proposal-counters-after', lambda: proposal_counter_snapshot(engine))
        updates_after = lane('task-updates-after', lambda: update_snapshot(engine))
        window_after = lane('window-state-after', lambda: window_snapshot(engine))
        traces.append(dict(window=active_window, role=role, original_trace=value,
                           controls_before=before, controls_after=after,
                           actual_normal_admission=dict(before=admission['normal_admission'],
                                                        after=admission_after['normal_admission']),
                           actual_gain_state=dict(before=admission['gain_state'], after=admission_after['gain_state']),
                           actual_split_work_counts=dict(before=admission, after=admission_after,
                                                         delta=split_delta(admission, admission_after)),
                           actual_proposal_counts=dict(before=proposals_before, after=proposals_after,
                               delta={k:proposals_after[k]-v for k,v in proposals_before.items()}),
                           actual_task_update_counts=dict(before=updates_before, after=updates_after,
                               delta={k:updates_after[k]-v for k,v in updates_before.items()}),
                           actual_window_cost_state=dict(before=window_before, after=window_after)))
        if count == 463:
            metric = lane('unchanged-clock-metrics-and-actual-TTFT', lambda: full_metrics(value['artifact']))
            metrics.append(dict(window=active_window, rank=engine.rank, output=metric))
        if checkpoint is not None:
            lane('checkpoint', lambda: checkpoint(traces, metrics))

    try:
        stage('V160/candidate-controls', lambda: controls.select('candidate', engine=engine))
        # Both variants warm before either measured trace; original run_trace
        # requires21 zero discovery deltas for each subsequent zero/full trace.
        for active_window in selected_windows:
            run(active_window, 'warmup65', 65)
        for role in ('zero65-1', 'zero65-2'):
            for active_window in selected_windows:
                run(active_window, role, 65)
        def seal():
            engine.seal_graph_cache()
            require(engine.graph_runner.graph_cache_sealed is True, 'one original Native natural seal')
        stage('V160/single-natural-seal', seal)
        for active_window in selected_windows:
            run(active_window, 'full463', 463)
    finally:
        stage('V160/restore-original-controls', lambda: controls.select('original', engine=engine))
    return dict(schema='V160.same-engine-active-window-pair.v1', configuration=config,
                traces=traces, per_rank_full_metrics=metrics, constructor_repeated=False,
                trace_order=list(selected_windows), outer_arrival_and_TPOT_clocks_unchanged=True,
                first_token_wall_samples_added_per_full=463,
                separate_policy_generation_per_trace=True, adopted=False, formal_Goodput=False)
