@torch.inference_mode()
def _generate_spec_rhythm_tree_decode(self, *, draft_states: list[PearlPipelineState], target_states: list[PearlPipelineState], request_params: Sequence[NativeSamplingParams], initial_batch_size: int, continuous_batching: bool, prefill_elapsed: float, started: float, max_rounds: int | None, prefilled_indices: set[int] | None=None) -> list[dict[str, Any]] | None:
    """Run SpecRhythm with fixed-shape tree proposals end to end.

    The scalar shaper remains the source of truth for every candidate
    budget.  Draft and target workers exchange only active nodes; graph
    padding is local to the tree forward and never counts against B.
    """
    request_params = [replace(p, spec_rhythm_max_gamma=5) for p in request_params]
    states = {index: SpecRhythmRuntimeState(request_index=index, home_batch_id=index % 2, slo_tpot_ms=params.slo_tpot_ms, slo_class=params.slo_class, max_gamma=params.spec_rhythm_max_gamma, output_token_budget=params.max_tokens, arrival_debt_cap_tokens=envs.VLLM_ASCEND_SPECRHYTHM_ARRIVAL_DEBT_CAP_TOKENS or None) for index, params in enumerate(request_params)}
    controller = SpecRhythmPipelineController(states)
    shaper = SpecRhythmBudgetShaper(min_gamma=self.config.spec_rhythm_min_gamma, max_gamma=max(self.config.spec_rhythm_tree_width * self.config.spec_rhythm_tree_depth, self.config.spec_rhythm_min_gamma), verification_budget=self.config.spec_rhythm_verification_budget, acceptance_floor=self.config.spec_rhythm_acceptance_floor, acceptance_ema_alpha=self.config.spec_rhythm_acceptance_ema_alpha, roofline=self.config.spec_rhythm_roofline)
    has_slo_constraints = any((params.slo_tpot_ms is not None or params.slo_class is not None for params in request_params))
    effective_eager_cap = self.config.spec_rhythm_max_eager_tokens or (shaper.max_gamma if has_slo_constraints else 0)
    local_states = draft_states if self.is_draft else target_states
    active: list[int] = []
    pending_admission = list(range(len(local_states)))
    live_admission = getattr(self, '_request_admission_callback', None)
    live_open = live_admission is not None
    prefetched = set(prefilled_indices or ())
    kv_ready_preloaded = set(getattr(self, '_kv_ready_preloaded_indices', ())) if self.config.spec_rhythm_kv_ready_arrivals else set()
    cache_activated = set(kv_ready_preloaded) if self.config.spec_rhythm_kv_ready_arrivals else set(prefilled_indices or ())
    cache_released: set[int] = set()
    payloads: dict[int, dict[str, Any]] = {}
    cache_mappings: dict[int, torch.Tensor] = {}
    completed: dict[int, PearlPipelineState] = {}
    finished_decode_elapsed: dict[int, float] = {}
    first_decode_indices: set[int] = set()
    prefill_coalescing = bool(self.config.spec_rhythm_prefill_coalesce_min_requests > 1 and self.config.spec_rhythm_prefill_coalesce_max_wait_ms > 0)
    counters = {'spec_rhythm_tree_rounds': 0, 'spec_rhythm_tree_nodes': 0, 'spec_rhythm_tree_accepted_nodes': 0, 'spec_rhythm_allocated_draft_tokens': 0, 'spec_rhythm_verified_tokens': 0, 'spec_rhythm_tree_full_accepts': 0, 'spec_rhythm_tree_rejections': 0, 'spec_rhythm_tree_eager_promoted': 0, 'spec_rhythm_tree_eager_invalidated': 0, 'spec_rhythm_tree_eager_dependency_rejections': 0, 'spec_rhythm_tree_eager_promotion_kv_move_rounds': 0, 'spec_rhythm_tree_eager_promotion_kv_move_rows': 0, 'spec_rhythm_tree_eager_promotion_kv_move_slots': 0, 'spec_rhythm_last_verification_roof': 0, 'spec_rhythm_unused_verification_tokens': 0, 'spec_rhythm_prefill_batches': 0, 'spec_rhythm_prefill_requests': 0, 'spec_rhythm_prefill_coalesce_enabled': int(prefill_coalescing), 'spec_rhythm_slo_aware_admission_enabled': int(envs.VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION), 'spec_rhythm_target_slo_first_enabled': int(envs.VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST), 'spec_rhythm_arrival_debt_cap_tokens': int(envs.VLLM_ASCEND_SPECRHYTHM_ARRIVAL_DEBT_CAP_TOKENS), 'spec_rhythm_prefill_coalesce_deferred_polls': 0, 'spec_rhythm_prefill_coalesce_size_releases': 0, 'spec_rhythm_prefill_coalesce_timeout_releases': 0, 'spec_rhythm_prefill_coalesce_forced_releases': 0, 'spec_rhythm_prefill_coalesce_max_wait_observed_ms': 0.0, 'spec_rhythm_prefill_singleton_batches': 0, 'spec_rhythm_prefill_pair_batches': 0, 'spec_rhythm_prefill_multi_batches': 0, 'spec_rhythm_peak_active_requests': 0, 'spec_rhythm_peak_verify_candidates': 0, 'spec_rhythm_target_query_tokens': 0, 'spec_rhythm_target_graph_padding_queries': 0, 'spec_rhythm_target_causal_tree_rounds': 0, 'spec_rhythm_target_full_tree_rounds': 0, 'spec_rhythm_draft_model_calls': 0, 'spec_rhythm_draft_graph_calls': 0, 'spec_rhythm_draft_graph_padding_queries': 0, 'spec_rhythm_draft_linear_full_chain_rounds': 0, 'spec_rhythm_draft_materialized_nodes': 0, 'spec_rhythm_draft_publish_rows': 0, 'spec_rhythm_draft_publish_moved_rows': 0, 'spec_rhythm_draft_publish_skipped_rows': 0, 'spec_rhythm_kv_compaction_rows': 0, 'spec_rhythm_kv_compaction_skipped_rows': 0, 'spec_rhythm_kv_compaction_rounds': 0, 'spec_rhythm_live_admitted_requests': 0, 'spec_rhythm_aborted_requests': 0, 'spec_rhythm_online_cache_activated_sequences': len(kv_ready_preloaded), 'spec_rhythm_online_cache_released_sequences': 0, 'spec_rhythm_online_cache_released_blocks': 0, 'spec_rhythm_kv_ready_host_staging_enabled': int(self.config.spec_rhythm_kv_ready_arrivals), 'spec_rhythm_kv_ready_restored_sequences': len(kv_ready_preloaded), 'spec_rhythm_kv_ready_restored_bytes': int(getattr(self, '_kv_ready_preload_bytes', 0)), 'spec_rhythm_kv_ready_restore_batches': int(bool(kv_ready_preloaded)), 'spec_rhythm_kv_ready_restore_host_ms': float(getattr(self, '_kv_ready_preload_host_ms', 0.0)), 'spec_rhythm_dual_batch_overlap_protocol': 1}
    round_count = 0
    last_cycle_ms = 0.0
    window_estimator = _dt_PackedTreeWindowEstimator(self.config.spec_rhythm_acceptance_ema_alpha)
    self._dynamic_tree_window_estimator = window_estimator
    decode_timeline = []
    host_timeline: list[dict[str, int | float]] = []
    npu_profiler = None
    npu_profile_dir = envs.VLLM_ASCEND_PEARL_NPU_PROFILE_DIR
    npu_profile_enabled = _npu_profile_call_enabled(npu_profile_dir, measurement_only=envs.VLLM_ASCEND_PEARL_NPU_PROFILE_MEASUREMENT_ONLY, profile_host_decode_steps=self.config.profile_host_decode_steps)
    if npu_profile_enabled:
        profile_rank = envs.VLLM_ASCEND_PEARL_NPU_PROFILE_RANK
        if profile_rank is None or profile_rank == -1 or profile_rank == self.rank:
            import torch_npu
            npu_profiler = torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU], schedule=torch_npu.profiler.schedule(wait=1, warmup=1, active=3, repeat=1), on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(npu_profile_dir, worker_name=f'specslo-tree-rank-{self.rank}'), record_shapes=True)
            _start_active_npu_profiler(self, npu_profiler)
    phase_seconds = {'draft': 0.0, 'target': 0.0, 'exchange': 0.0, 'verify': 0.0, 'broadcast': 0.0, 'state_update': 0.0}
    decode_profile_seconds = {'draft_compute': 0.0, 'draft_to_target_communication': 0.0, 'target_compute': 0.0, 'target_verdict': 0.0, 'target_to_draft_communication': 0.0, 'wait_sync': 0.0, 'state_update': 0.0}
    decode_profile_detail_seconds = {'scheduler_plan': 0.0, 'draft_tree_setup': 0.0, 'draft_tree_level_compute': 0.0, 'draft_tree_topk': 0.0, 'draft_tree_materialize_metadata': 0.0, 'draft_tree_materialize_compute': 0.0, 'target_tree_setup': 0.0, 'target_tree_metadata': 0.0, 'target_tree_model': 0.0, 'target_tree_output': 0.0, 'draft_postprocess': 0.0, 'profile_timeline_collective': 0.0, 'proposal_publish': 0.0, 'commit_preflight': 0.0, 'commit_consensus': 0.0, 'kv_compaction': 0.0, 'request_commit': 0.0, 'cycle_accounting': 0.0}
    profiled_decode_steps = 0
    native_tree_forward = getattr(self.draft_tree_forward, '__func__', None) is NativePearlEngine.draft_tree_forward
    native_target_tree_forward = getattr(self.target_tree_forward, '__func__', None) is NativePearlEngine.target_tree_forward

    def profile_begin(enabled: bool) -> float | None:
        """Start an intrusive, explicitly bounded profile interval.

        Synchronization is rank-local.  In particular, a target rank must
        not wait for the draft rank here: doing so would destroy the very
        cross-device overlap this profile is intended to measure.
        """
        if not enabled:
            return None
        torch.npu.synchronize()
        return time.perf_counter()

    def profile_end(started_at: float | None, phase: str) -> float:
        if started_at is None:
            return 0.0
        torch.npu.synchronize()
        elapsed = time.perf_counter() - started_at
        decode_profile_seconds[phase] += elapsed
        return elapsed

    def profile_detail_end(started_at: float | None, phase: str) -> float:
        """Finish a nested diagnostic interval without broad-phase accounting."""
        if started_at is None:
            return 0.0
        torch.npu.synchronize()
        elapsed = time.perf_counter() - started_at
        decode_profile_detail_seconds[phase] += elapsed
        return elapsed

    def finish_profile_round(started_at: float | None, accounted_before: float) -> None:
        """Charge the unclassified critical-path tail to wait/sync.

        ``profile_end`` measures role-local model and transport intervals.
        Everything else on the rank's cycle critical path -- scheduler
        gaps, host/device synchronizations and control collectives -- is
        intentionally exposed as residual wait/sync instead of vanishing
        from the profile.
        """
        if started_at is None:
            return
        torch.npu.synchronize()
        total = time.perf_counter() - started_at
        accounted = sum(decode_profile_seconds.values()) - accounted_before
        decode_profile_seconds['wait_sync'] += max(0.0, total - accounted)

    def activate_online_cache(request_indices: Sequence[int]) -> None:
        """Bind physical KV pages to newly admitted tree rows."""
        if not (self.config.spec_rhythm_online_prefill or self.config.spec_rhythm_kv_ready_arrivals):
            return
        new_indices = [index for index in request_indices if index not in cache_activated]
        if not new_indices:
            return
        for index in new_indices:
            prompt = local_states[index].token_ids[:local_states[index].prompt_length]
            self._activate_cache_sequence(index, prompt, enable_prefix_caching=self.config.enable_prefix_caching and (not self.config.spec_rhythm_kv_ready_arrivals))
        if self.config.spec_rhythm_kv_ready_arrivals:
            snapshots = getattr(self, '_kv_ready_host_snapshots', None)
            batch_snapshots = [None if snapshots is None else snapshots.get(index) for index in new_indices]
            if any((snapshot is None for snapshot in batch_snapshots)):
                raise RuntimeError('KV-ready admission lost its staged prompt KV.')
            restore_started = time.perf_counter()
            restored_bytes = self._restore_prompt_kv_batch_from_host(new_indices, batch_snapshots)
            counters['spec_rhythm_kv_ready_restore_host_ms'] += (time.perf_counter() - restore_started) * 1000.0
            counters['spec_rhythm_kv_ready_restore_batches'] += 1
            counters['spec_rhythm_kv_ready_restored_sequences'] += len(new_indices)
            counters['spec_rhythm_kv_ready_restored_bytes'] += restored_bytes
            assert snapshots is not None
            for index in new_indices:
                snapshots.pop(index)
        for index in new_indices:
            cache_activated.add(index)
            counters['spec_rhythm_online_cache_activated_sequences'] += 1

    def release_online_cache(request_indices: Sequence[int]) -> None:
        """Return completed/aborted tree rows after proposal invalidation."""
        if not (self.config.spec_rhythm_online_prefill or self.config.spec_rhythm_kv_ready_arrivals):
            return
        for index in request_indices:
            if index not in cache_activated or index in cache_released:
                continue
            released_blocks = self._release_cache_sequence(index)
            cache_released.add(index)
            counters['spec_rhythm_online_cache_released_sequences'] += 1
            counters['spec_rhythm_online_cache_released_blocks'] += released_blocks

    def receive_live_admissions(*, block: bool) -> bool:
        """Append controller-synchronized HTTP arrivals at a cycle fence."""
        nonlocal has_slo_constraints, effective_eager_cap
        if live_admission is None:
            return False
        admission_result = live_admission(block)
        if len(admission_result) == 2:
            closed, arrivals = admission_result
            aborted_request_ids: Sequence[str] = ()
        elif len(admission_result) == 3:
            closed, arrivals, aborted_request_ids = admission_result
        else:
            raise RuntimeError('SpecSLO live control callback returned an invalid tuple')
        if closed and arrivals:
            raise RuntimeError('SpecSLO live admission cannot close with pending requests')
        abort_set = {str(value) for value in aborted_request_ids}
        for index, params in enumerate(request_params):
            if str(params.request_id) not in abort_set or index in completed:
                continue
            for state in (draft_states[index], target_states[index]):
                state.aborted = True
            if index in pending_admission:
                pending_admission.remove(index)
            if index in active:
                active.remove(index)
            for proposal_id, payload in tuple(payloads.items()):
                if payload['ticket'].request_index == index:
                    payloads.pop(proposal_id, None)
                    cache_mappings.pop(proposal_id, None)
            controller.invalidate_request(index)
            completed[index] = local_states[index].clone()
            finished_decode_elapsed[index] = states[index].decode_elapsed_ms
            release_online_cache((index,))
            counters['spec_rhythm_aborted_requests'] += 1
        if not arrivals:
            return bool(closed)
        if len(request_params) + len(arrivals) > self._cache_sequence_capacity:
            raise RuntimeError('SpecSLO live requests exceed the reserved KV sequence capacity')
        existing_signature = (request_params[0].temperature > 0, request_params[0].draft_temperature > 0)
        for prompt, params in arrivals:
            if not isinstance(params, NativeSamplingParams):
                raise TypeError('SpecSLO live admission requires NativeSamplingParams')
            if (params.temperature > 0, params.draft_temperature > 0) != existing_signature:
                raise ValueError('SpecSLO live admission cannot mix greedy and stochastic sampling modes')
            tokens = [int(token_id) for token_id in prompt]
            tail_capacity = self.config.spec_rhythm_tree_width * self.config.spec_rhythm_tree_depth + 1
            if not tokens or len(tokens) + params.max_tokens + tail_capacity > self.config.max_model_len:
                raise ValueError('SpecSLO live prompt plus completion exceeds max_model_len')
            index = len(request_params)
            self._activate_cache_sequence(index, tokens, enable_prefix_caching=self.config.enable_prefix_caching)
            cache_activated.add(index)
            counters['spec_rhythm_online_cache_activated_sequences'] += 1
            draft_state = PearlPipelineState(list(tokens), len(tokens), temperature=params.temperature, draft_temperature=params.draft_temperature, top_p=params.top_p, top_k=params.top_k, draft_top_p=params.draft_top_p, draft_top_k=params.draft_top_k, max_tokens=params.max_tokens, ignore_eos=params.ignore_eos, slo_tpot_ms=params.slo_tpot_ms, slo_class=params.slo_class, request_id=params.request_id, arrival_ts=params.arrival_ts)
            draft_states.append(draft_state)
            target_states.append(draft_state.clone())
            request_params.append(params)
            states[index] = SpecRhythmRuntimeState(request_index=index, home_batch_id=index % 2, slo_tpot_ms=params.slo_tpot_ms, slo_class=params.slo_class, max_gamma=params.spec_rhythm_max_gamma, output_token_budget=params.max_tokens, arrival_debt_cap_tokens=envs.VLLM_ASCEND_SPECRHYTHM_ARRIVAL_DEBT_CAP_TOKENS or None)
            pending_admission.append(index)
            counters['spec_rhythm_live_admitted_requests'] += 1
            if params.slo_tpot_ms is not None or params.slo_class is not None:
                has_slo_constraints = True
                effective_eager_cap = self.config.spec_rhythm_max_eager_tokens or shaper.max_gamma
                self.graph_runner.set_expected_fia_batch_size(None)
        return bool(closed)

    def admit_available(*, poll_live: bool=True, block_live: bool=False) -> bool:
        closed = receive_live_admissions(block=block_live) if poll_live else False
        if not pending_admission or len(active) >= initial_batch_size:
            return closed
        arrival_gated = bool(self.config.spec_rhythm_online_prefill or self.config.spec_rhythm_kv_ready_arrivals)
        now_tensor = torch.tensor([time.time() if self.rank == self.topology.target_leader_rank else 0.0], dtype=torch.float64, device='cpu')
        _dt_cpu_broadcast(self, now_tensor, src=self.topology.target_leader_rank, kind='arrival_clock')
        now = float(now_tensor.cpu().item())
        vacant_capacity = initial_batch_size - len(active)
        admitted: list[int] = []
        packed_prompt_tokens = 0
        token_budget_saturated = False
        admission_candidates = order_spec_rhythm_admission_params(pending_admission, request_params=request_params, enabled=has_slo_constraints and envs.VLLM_ASCEND_SPECRHYTHM_SLO_AWARE_ADMISSION)
        for index in admission_candidates:
            if arrival_gated and request_params[index].arrival_ts is not None and (request_params[index].arrival_ts > now):
                continue
            prompt_tokens = 0 if index in prefetched else int(local_states[index].prompt_length)
            if admitted and (len(admitted) >= vacant_capacity or packed_prompt_tokens + prompt_tokens > self.config.max_num_batched_tokens):
                token_budget_saturated = packed_prompt_tokens + prompt_tokens > self.config.max_num_batched_tokens
                break
            if prompt_tokens > self.config.max_num_batched_tokens:
                raise ValueError('An online PEARL prefill request exceeds max_num_batched_tokens.')
            admitted.append(index)
            packed_prompt_tokens += prompt_tokens
            if len(admitted) >= vacant_capacity:
                break
        if admitted and prefill_coalescing:
            decision = decide_prefill_coalesce(ready_count=len(admitted), remaining_count=len(pending_admission), active_request_count=len(active), request_limit=min(initial_batch_size, self.config.prefill_chunk_size or self.config.max_num_seqs), minimum_requests=self.config.spec_rhythm_prefill_coalesce_min_requests, maximum_wait_ms=self.config.spec_rhythm_prefill_coalesce_max_wait_ms, now=now, ready_arrival_times=[request_params[index].arrival_ts for index in admitted], token_budget_saturated=token_budget_saturated)
            if decision.waited_ms is not None:
                counters['spec_rhythm_prefill_coalesce_max_wait_observed_ms'] = max(counters['spec_rhythm_prefill_coalesce_max_wait_observed_ms'], decision.waited_ms)
            if decision.reason == 'size':
                counters['spec_rhythm_prefill_coalesce_size_releases'] += 1
            elif decision.reason == 'timeout':
                counters['spec_rhythm_prefill_coalesce_timeout_releases'] += 1
            elif decision.release:
                counters['spec_rhythm_prefill_coalesce_forced_releases'] += 1
            else:
                counters['spec_rhythm_prefill_coalesce_deferred_polls'] += 1
            if not decision.release:
                admitted = []
        homes = [sum((states[index].home_batch_id == home for index in active)) for home in (0, 1)]
        for index in admitted:
            home = 0 if homes[0] <= homes[1] else 1
            states[index].home_batch_id = home
            homes[home] += 1
            arrival = request_params[index].arrival_ts
            if arrival_gated and arrival is not None:
                states[index].arrival_wait_ms = max(0.0, (now - arrival) * 1000.0)
        needs_prefill = [index for index in admitted if index not in prefetched]
        cache_activation_indices = admitted if self.config.spec_rhythm_kv_ready_arrivals else needs_prefill
        if cache_activation_indices:
            activate_online_cache(cache_activation_indices)
        if needs_prefill:
            refill_started = time.perf_counter()
            tokens = self._prefill_and_sample_target_batch([local_states[index].token_ids for index in needs_prefill], [local_states[index] for index in needs_prefill], needs_prefill)
            refill_elapsed = torch.tensor([(time.perf_counter() - refill_started) * 1000.0 if self.rank == self.topology.target_leader_rank else 0.0], dtype=torch.float64, device='cpu')
            _dt_cpu_broadcast(self, refill_elapsed, src=self.topology.target_leader_rank, kind='prefill_clock')
            refill_ms = float(refill_elapsed.cpu().item())
            for index in active:
                states[index].add_decode_time(refill_ms)
            counters['spec_rhythm_online_refill_ms'] = counters.get('spec_rhythm_online_refill_ms', 0.0) + refill_ms
            for index, token in zip(needs_prefill, tokens):
                for state in (draft_states[index], target_states[index]):
                    state.token_ids.append(token)
                    state.committed_length += 1
            prefetched.update(needs_prefill)
            counters['spec_rhythm_prefill_batches'] += 1
            counters['spec_rhythm_prefill_requests'] += len(needs_prefill)
            if len(needs_prefill) == 1:
                counters['spec_rhythm_prefill_singleton_batches'] += 1
            elif len(needs_prefill) == 2:
                counters['spec_rhythm_prefill_pair_batches'] += 1
            else:
                counters['spec_rhythm_prefill_multi_batches'] += 1
        for index in admitted:
            pending_admission.remove(index)
            states[index].delivered_tokens = len(local_states[index].committed_completion_token_ids)
            states[index].decode_start_elapsed_ms = states[index].decode_elapsed_ms
            self._deliver_committed_tokens(index, request_params[index], local_states[index])
            if _finished(local_states[index], self.eos_token_ids):
                completed[index] = local_states[index].clone()
                finished_decode_elapsed[index] = 0.0
                release_online_cache((index,))
            else:
                active.append(index)
        counters['spec_rhythm_peak_active_requests'] = max(counters['spec_rhythm_peak_active_requests'], len(active))
        return closed

    def run_profiled_target_only(indices: Sequence[int], *, profile_this_round: bool) -> list[int]:
        """Advance one logical home when the measured candidate roof is zero.

        Both model partitions consume the current committed token so their
        KV prefixes remain aligned if a later batch/context key permits
        speculation again.  Only the target result is committed.
        """
        rows = [int(index) for index in indices]
        if not rows:
            raise RuntimeError('SpecSLO target-only fallback requires one logical-home request')
        input_ids = torch.tensor([local_states[index].token_ids[-1] for index in rows], dtype=torch.long, device=self.device)
        positions = [len(local_states[index].token_ids) - 1 for index in rows]
        target_execution: dict[str, Any] | None = None
        compute_phase = 'draft_compute' if self.is_draft else 'target_compute'
        compute_profile_started = profile_begin(profile_this_round)
        if self.is_draft:
            self._run_device_packed_hidden(input_ids, rows, positions, use_aclgraph=not self.config.enforce_eager, use_fused_infer_attention=False)
            token_ids = torch.zeros(len(rows), dtype=torch.long, device=self.device)
        else:
            packed_positions, metadata = self._prepare_attention_metadata(rows, positions, False)
            temperatures = [target_states[index].temperature for index in rows]
            top_ps = [target_states[index].top_p for index in rows]
            top_ks = [target_states[index].top_k for index in rows]
            target_use_aclgraph = not self.config.enforce_eager and (not envs.VLLM_ASCEND_SPECRHYTHM_DISABLE_TARGET_ACLGRAPH)
            if not any(temperatures):
                if not target_use_aclgraph:
                    hidden = self.model(input_ids, packed_positions, metadata)
                    token_ids = self.model.compute_greedy_tokens(hidden, self.draft_vocab_size)
                else:
                    token_ids = self.graph_runner.run_target_greedy([input_ids], [packed_positions], [metadata], self.draft_vocab_size)[0]
            else:
                if not target_use_aclgraph:
                    hidden = self.model(input_ids, packed_positions, metadata)
                    logits = self.model.compute_logits(hidden)[:, :self.draft_vocab_size]
                else:
                    logits = self.graph_runner.run_target_logits([input_ids], [packed_positions], [metadata], self.draft_vocab_size)[0]
                token_ids = _sample_logits(logits, temperatures, top_ps=top_ps, top_ks=top_ks)
                if dist.is_initialized():
                    dist.broadcast(token_ids, src=self.topology.target_leader_rank, group=self.groups.target_group)
            target_execution = {'attention_backend': attention_backend_identity(metadata), 'used_aclgraph': bool(target_use_aclgraph and self.graph_runner.last_target_execution.used_aclgraph)}
        profile_end(compute_profile_started, compute_phase)
        correction_profile_started = profile_begin(profile_this_round)
        torch.npu.synchronize()
        self._vote_spec_rhythm_prefill_finiteness()
        self._validate_spec_rhythm_target_graph(target_execution, has_target_work=True, target_only=True)
        dist.broadcast(token_ids, src=self.topology.target_leader_rank)
        values = [int(value) for value in token_ids.cpu().tolist()]
        profile_end(correction_profile_started, 'target_to_draft_communication')
        return values

    def complete_target_only_cycle(target_indices: Sequence[int], *, cycle_started: float, unprofiled: bool, target_home: int | None, profile_this_round: bool, profile_round_started: float | None, profile_accounted_before: float) -> None:
        """Finish one AR fallback cycle and keep both model KV states aligned."""
        nonlocal active, live_open, last_cycle_ms, round_count, profiled_decode_steps
        rows = [int(index) for index in target_indices]
        state_profile_started = profile_begin(profile_this_round)
        invalidated_eager = 0
        for index in active:
            invalidated_eager += int(index in controller.staged_eager)
            for ticket in (controller.ready.get(index), controller.staged_eager.get(index)):
                if ticket is not None:
                    payloads.pop(ticket.proposal_id, None)
                    cache_mappings.pop(ticket.proposal_id, None)
            controller.invalidate_request(index)
        counters['spec_rhythm_tree_eager_invalidated'] += invalidated_eager
        profile_end(state_profile_started, 'state_update')
        tokens = run_profiled_target_only(rows, profile_this_round=profile_this_round)
        state_profile_started = profile_begin(profile_this_round)
        finished: list[int] = []
        for index, token in zip(rows, tokens):
            for state in (draft_states[index], target_states[index]):
                state.token_ids.append(token)
                assert state.committed_length is not None
                state.committed_length += 1
            states[index].record_verification(proposed_tokens=0, accepted_tokens=0, delivered_tokens=1, draft_confidence=None, ema_alpha=self.config.spec_rhythm_acceptance_ema_alpha)
            self._deliver_committed_tokens(index, request_params[index], local_states[index])
            if _finished(local_states[index], self.eos_token_ids):
                finished.append(index)
        completed_home = states[rows[0]].home_batch_id if target_home is None else int(target_home)
        controller.next_target_home_batch_id = 1 - completed_home
        elapsed_tensor = torch.tensor([time.perf_counter() - cycle_started if self.rank == self.topology.target_leader_rank else 0.0], dtype=torch.float64, device='cpu')
        _dt_cpu_broadcast(self, elapsed_tensor, src=self.topology.target_leader_rank, kind='cycle_clock')
        last_cycle_ms = float(elapsed_tensor.cpu().item()) * 1000.0
        for index in active:
            states[index].add_decode_time(last_cycle_ms)
        for index in finished:
            completed[index] = local_states[index].clone()
            finished_decode_elapsed[index] = states[index].decode_elapsed_ms
        finished_set = set(finished)
        active = [index for index in active if index not in finished_set]
        release_online_cache(finished)
        if not live_open:
            admit_available(poll_live=False)
        counters['spec_rhythm_last_verification_roof'] = 0
        counters['spec_rhythm_unused_verification_tokens'] = 0
        counters['spec_rhythm_target_query_tokens'] += len(rows)
        counters['spec_rhythm_target_only_fallback_rounds'] = counters.get('spec_rhythm_target_only_fallback_rounds', 0) + 1
        counters['spec_rhythm_target_only_fallback_tokens'] = counters.get('spec_rhythm_target_only_fallback_tokens', 0) + len(tokens)
        if unprofiled:
            counters['spec_rhythm_unprofiled_target_only_fallback_rounds'] = counters.get('spec_rhythm_unprofiled_target_only_fallback_rounds', 0) + 1
            counters['spec_rhythm_unprofiled_target_only_fallback_tokens'] = counters.get('spec_rhythm_unprofiled_target_only_fallback_tokens', 0) + len(tokens)
        counters['spec_rhythm_tree_rounds'] += 1
        profile_end(state_profile_started, 'state_update')
        finish_profile_round(profile_round_started, profile_accounted_before)
        if profile_this_round:
            profiled_decode_steps += 1
        round_count += 1
        if npu_profiler is not None:
            npu_profiler.step()
    admit_available(poll_live=False)
    while (active or pending_admission or live_open) and (max_rounds is None or round_count < max_rounds):
        if not active:
            if live_open and (not pending_admission):
                live_open = not admit_available(block_live=True)
                if not live_open and (not active) and (not pending_admission):
                    break
            else:
                admit_available()
            if active:
                continue
            time.sleep(0.01)
            continue
        if live_open:
            live_open = not admit_available()
            if not active:
                continue
        cycle_started = time.perf_counter()
        host_trace: dict[str, int | float] | None = None
        if round_count < self.config.profile_host_decode_steps:
            host_trace = {'step': round_count, 'rank': self.rank, 'is_draft_rank': int(self.is_draft), 'cycle_start_seconds': cycle_started}
            host_trace['scheduler_start_seconds'] = time.perf_counter()
        profile_this_round = round_count < self.config.profile_decode_steps
        profile_round_started = profile_begin(profile_this_round)
        profile_accounted_before = sum(decode_profile_seconds.values())
        scheduler_profile_started = time.perf_counter() if profile_this_round else None
        projected_wait_ms = max(last_cycle_ms * 2.0, 1e-06)
        roof_context_len = max((len(local_states[index].token_ids) for index in active))
        strict_profile = shaper.roofline if isinstance(shaper.roofline, ProfiledRoofline) else None
        unprofiled_target_only = False
        if strict_profile is not None and strict_profile.metadata.get('unprofiled_execution_policy') == 'target_only':
            measured_roof = strict_profile.lookup_optional(len(active), roof_context_len)
            if measured_roof is None:
                verification_roof = 0
                unprofiled_target_only = True
            else:
                verification_roof = measured_roof
                prospective_home = controller.next_target_home_batch_id
                prospective_rows = [index for index in active if states[index].home_batch_id == prospective_home]
                if not prospective_rows:
                    prospective_home = 1 - prospective_home
                    prospective_rows = [index for index in active if states[index].home_batch_id == prospective_home]
                if not strict_profile.covers_execution(len(active), roof_context_len, len(prospective_rows)):
                    verification_roof = 0
                    unprofiled_target_only = True
        else:
            verification_roof = shaper.verification_roof(len(active), roof_context_len)
        if host_trace is not None:
            host_trace['scheduler_roof_end_seconds'] = time.perf_counter()
        if verification_roof == 0:
            home = controller.next_target_home_batch_id
            target_indices = [index for index in active if states[index].home_batch_id == home]
            if not target_indices:
                home = 1 - home
                target_indices = [index for index in active if states[index].home_batch_id == home]
            if isinstance(shaper.roofline, ProfiledRoofline) and (not unprofiled_target_only):
                shaper.roofline.validate_execution(len(active), roof_context_len, len(target_indices))
            complete_target_only_cycle(target_indices, cycle_started=cycle_started, unprofiled=unprofiled_target_only, target_home=home, profile_this_round=profile_this_round, profile_round_started=profile_round_started, profile_accounted_before=profile_accounted_before)
            continue
        shrink = self._bound_spec_rhythm_tree_ready(controller, payloads, cache_mappings, active, verification_roof)
        counters['spec_rhythm_tree_eager_invalidated'] += shrink['invalidated_eager']
        counters['spec_rhythm_roof_pruned_candidates'] = counters.get('spec_rhythm_roof_pruned_candidates', 0) + shrink['pruned_candidates']
        plan = controller.build_plan(active, active_indices_registered_unique=True, verification_budget=verification_roof, priority=has_slo_constraints, projected_wait_ms=projected_wait_ms, priority_burst=self.config.spec_rhythm_priority_burst, target_slo_first=envs.VLLM_ASCEND_SPECRHYTHM_TARGET_SLO_FIRST, merge_ready_homes=self.config.spec_rhythm_merge_ready_homes, max_target_requests=self.config.spec_rhythm_max_target_batch if has_slo_constraints and self.config.spec_rhythm_max_target_batch > 0 else None)
        if host_trace is not None:
            host_trace['scheduler_plan_end_seconds'] = time.perf_counter()
        target_indices = list(plan.target_request_indices)
        if target_indices and strict_profile is not None and (strict_profile.metadata.get('unprofiled_execution_policy') == 'target_only') and (not strict_profile.covers_execution(len(active), roof_context_len, len(target_indices))):
            complete_target_only_cycle(target_indices, cycle_started=cycle_started, unprofiled=True, target_home=plan.target_home_batch_id, profile_this_round=profile_this_round, profile_round_started=profile_round_started, profile_accounted_before=profile_accounted_before)
            continue
        target_payloads = [payloads[controller.ready[index].proposal_id] for index in target_indices]
        actual_candidates = sum((payload['plan'].candidate_budget for payload in target_payloads))
        if target_indices and isinstance(shaper.roofline, ProfiledRoofline):
            shaper.roofline.validate_execution(len(active), max((len(local_states[index].token_ids) for index in active)), len(target_indices))
        if actual_candidates > shaper.verification_roof(len(active), max((len(local_states[index].token_ids) for index in active))):
            raise RuntimeError('SpecRhythm actual verify input exceeds global B')
        counters['spec_rhythm_peak_verify_candidates'] = max(counters['spec_rhythm_peak_verify_candidates'], actual_candidates)
        for index, payload in zip(target_indices, target_payloads):
            ticket = controller.validate_verification(index)
            if ticket.proposal_id != payload['ticket'].proposal_id:
                raise RuntimeError('SpecRhythm tree proposal identity changed')
            if payload['plan'].prefix_len != len(local_states[index].token_ids) - 1:
                raise RuntimeError('SpecRhythm tree proposal has a stale committed prefix')

        def _dt_submit_current_tree_target():
            target_plans = [payload['plan'] for payload in target_payloads]
            target_rows = [payload['row'] for payload in target_payloads]
            target_roots = [local_states[index].token_ids[-1] for index in target_indices]
            target_request_ids = list(target_indices)
            real_target_count = len(target_plans)
            target_temperatures = [local_states[index].temperature for index in target_indices]
            if native_target_tree_forward and target_plans and (not any(target_temperatures)):
                target_execution_plans, target_execution_roots, target_execution_rows, target_execution_ids, real_target_count = self._pad_spec_rhythm_target_tree_graph(target_plans, target_roots, target_rows, target_request_ids, active, local_states)
            else:
                target_execution_plans = target_plans
                target_execution_roots = target_roots
                target_execution_rows = target_rows
                target_execution_ids = target_request_ids
            target_started = time.perf_counter()
            target_profile_started = profile_begin(profile_this_round and (not self.is_draft) and bool(target_indices))
            target_tree_kwargs: dict[str, Any] = {'sequence_ids': target_execution_ids, 'return_logits': False}
            if native_target_tree_forward:
                target_tree_kwargs['real_tree_count'] = real_target_count
            if any((value > 0 for value in target_temperatures)):
                target_tree_kwargs['temperatures'] = target_temperatures
                target_top_ps = [local_states[index].top_p for index in target_indices]
                target_top_ks = [local_states[index].top_k for index in target_indices]
                if any((value < 1.0 for value in target_top_ps)) or any((value > 0 for value in target_top_ks)):
                    target_tree_kwargs['top_ps'] = target_top_ps
                    target_tree_kwargs['top_ks'] = target_top_ks
            with torch.profiler.record_function('SpecSLO/TargetCompute'):
                target_output = self.target_tree_forward(target_execution_plans, target_execution_roots, target_execution_rows, **target_tree_kwargs) if target_indices else None
            target_submit_ended = time.perf_counter()
            return (target_output, target_started, target_profile_started, target_plans, target_rows, target_roots, target_request_ids, real_target_count, target_temperatures, target_execution_plans, target_execution_roots, target_execution_rows, target_execution_ids, target_tree_kwargs, target_submit_ended)
        self._profile_tree_subphases = profile_this_round
        counters['spec_rhythm_tree_early_target_submit_enabled'] = int(envs.VLLM_ASCEND_SPECRHYTHM_EARLY_CURRENT_TARGET_SUBMIT)
        counters['spec_rhythm_tree_early_target_window_conservative_residual'] = 0
        counters['spec_rhythm_tree_early_target_window_full_worker_wall_predictor'] = 1
        early_target_context = None
        if _dt_tree_early_target_decide(counters, enabled=envs.VLLM_ASCEND_SPECRHYTHM_EARLY_CURRENT_TARGET_SUBMIT, is_draft=self.is_draft, native_target=native_target_tree_forward, has_target=bool(target_indices), profile=profile_this_round, greedy=not any((local_states[index].temperature for index in target_indices)), enforce_eager=self.config.enforce_eager, stable_graphs=self.config.spec_rhythm_stable_graphs):
            early_target_context = _dt_submit_current_tree_target()
        normal = list(plan.normal_draft_request_indices)
        eager = list(plan.eager_candidate_indices)
        if effective_eager_cap <= 0:
            eager = []
        else:
            eager = [index for index in eager if states[index].projected_progress_gap(projected_wait_ms) > 0 and states[index].urgency(projected_wait_ms) >= self.config.spec_rhythm_urgency_threshold and (states[index].expected_acceptance_benefit >= self.config.spec_rhythm_acceptance_floor) and (int(payloads[controller.ready[index].proposal_id]['plan'].cache_positions.max().item()) + 2 + self.config.spec_rhythm_tree_width * self.config.spec_rhythm_tree_depth <= self.config.max_model_len)]
        exploration_cost = self.config.spec_rhythm_tree_width * self.config.spec_rhythm_tree_depth
        window = window_estimator.estimate(normal_tokens=len(normal) * exploration_cost, max_draft_tokens=(len(normal) + len(eager)) * exploration_cost, eager_work=bool(eager))
        eager = sorted(eager, key=lambda index: (states[index].projected_progress_gap(projected_wait_ms) * states[index].expected_acceptance_benefit, states[index].urgency(projected_wait_ms), -index), reverse=True)
        reserve_remaining = self.config.spec_rhythm_eager_reserve_tokens
        reserved_eager_rows = 0
        if reserve_remaining > 0:
            for index in eager:
                cap = min(shaper.max_gamma, states[index].max_gamma or shaper.max_gamma, effective_eager_cap)
                minimum = min(shaper.min_gamma, cap)
                if minimum <= 0 or minimum > reserve_remaining:
                    continue
                reserved_eager_rows += 1
                reserve_remaining -= minimum
        eager_window_tokens = window.eager_token_budget
        if window.calibrated and window.eager_fixed_overhead_hidden:
            eager_window_tokens += reserved_eager_rows * exploration_cost
        eager = eager[:eager_window_tokens // exploration_cost]
        budget_plan = shaper.shape(plan_id=plan.plan_id, normal_request_indices=normal, eager_request_indices=eager, states=states, projected_wait_ms=projected_wait_ms, context_len=max((len(local_states[index].token_ids) for index in active)), draft_token_budget=self.config.spec_rhythm_draft_token_budget, batch_size=len(active), eager_token_cap=effective_eager_cap or None, eager_reserve_tokens=self.config.spec_rhythm_eager_reserve_tokens, verification_roof=verification_roof)
        if host_trace is not None:
            host_trace['scheduler_budget_end_seconds'] = time.perf_counter()
        counters['spec_rhythm_last_verification_roof'] = budget_plan.verification_roof
        counters['spec_rhythm_unused_verification_tokens'] = int(budget_plan.unused_verification_tokens)
        normal_budgets = dict(budget_plan.normal_budgets)
        eager_budgets = dict(budget_plan.eager_budgets)
        work_indices = [*normal_budgets, *eager_budgets]
        work_budgets = [*normal_budgets.values(), *eager_budgets.values()]
        work_eager = [False] * len(normal_budgets) + [True] * len(eager_budgets)
        counters['spec_rhythm_unused_verification_tokens'] = max(0, int(budget_plan.verification_roof) - sum(normal_budgets.values()) - sum(eager_budgets.values()))
        eager_parent_sources: dict[int, tuple[TreeSpeculationPlan, Sequence[int]]] = {}
        work_plans: list[TreeSpeculationPlan] = []
        for index, budget, is_eager in zip(work_indices, work_budgets, work_eager):
            if is_eager:
                parent_ticket = controller.ready.get(index)
                if parent_ticket is None:
                    raise RuntimeError('SpecRhythm eager tree request has no ready parent proposal')
                parent_payload = payloads.get(parent_ticket.proposal_id)
                if parent_payload is None:
                    raise RuntimeError('SpecRhythm eager tree parent payload is missing')
                parent_plan = parent_payload['plan']
                parent_row = parent_payload['row']
                work_plans.append(self._spec_rhythm_tree_eager_plan(parent_plan, local_states[index], budget))
                eager_parent_sources[index] = (parent_plan, parent_row)
            else:
                work_plans.append(self._spec_rhythm_tree_plan(local_states[index], budget))
        tickets = [controller.new_ticket(index, gamma=budget, eager=is_eager) for index, budget, is_eager in zip(work_indices, work_budgets, work_eager)]
        profile_detail_end(scheduler_profile_started, 'scheduler_plan')
        if host_trace is not None:
            host_trace['scheduler_end_seconds'] = time.perf_counter()
            host_trace['target_requests'] = len(target_indices)
            host_trace['draft_requests'] = len(work_plans)
            host_trace['verify_candidates'] = actual_candidates
        self._profile_tree_subphases = profile_this_round
        self._defer_tree_materialization = native_tree_forward
        self._defer_normal_tree_materialization = native_tree_forward
        draft_started = time.perf_counter()
        draft_profile_started = profile_begin(profile_this_round and self.is_draft and bool(work_plans))
        draft_tree_kwargs: dict[str, Any] = {'eager_parent_sources': eager_parent_sources}
        work_temperatures = [local_states[index].draft_temperature for index in work_indices]
        if any((value > 0 for value in work_temperatures)):
            draft_tree_kwargs['draft_temperatures'] = work_temperatures
            work_top_ps = [local_states[index].draft_top_p for index in work_indices]
            work_top_ks = [local_states[index].draft_top_k for index in work_indices]
            if any((value < 1.0 for value in work_top_ps)) or any((value > 0 for value in work_top_ks)):
                draft_tree_kwargs['draft_top_ps'] = work_top_ps
                draft_tree_kwargs['draft_top_ks'] = work_top_ks
        if native_tree_forward:
            draft_tree_kwargs['committed_catchup_token_ids'] = [local_states[index].token_ids[max(0, len(local_states[index].token_ids) - self.config.spec_rhythm_tree_depth - 1):] if not is_eager else () for index, is_eager in zip(work_indices, work_eager)]
            work_index_set = set(work_indices)
            draft_tree_kwargs['graph_padding_catchup_rows'] = [(index, len(local_states[index].token_ids) - 1, local_states[index].token_ids[max(0, len(local_states[index].token_ids) - self.config.spec_rhythm_tree_depth - 1):]) for index in active if index not in work_index_set]
        with torch.profiler.record_function('SpecSLO/DraftCompute'):
            draft_output = self.draft_tree_forward(work_plans, [local_states[index].token_ids[-1] for index in work_indices], work_indices, **draft_tree_kwargs) if work_plans else None
        if draft_output is not None:
            torch.npu.synchronize()
            counters['spec_rhythm_draft_model_calls'] += draft_output.get('model_calls', 0)
            counters['spec_rhythm_draft_graph_calls'] += draft_output.get('graph_calls', 0)
            counters['spec_rhythm_draft_graph_padding_queries'] += draft_output.get('graph_padding_query_count', 0)
            counters['spec_rhythm_draft_linear_full_chain_rounds'] += int(draft_output.get('linear_full_chain', False))
            for phase, seconds in draft_output.get('profile_seconds', {}).items():
                decode_profile_detail_seconds[phase] += float(seconds)
        target_residual_started = time.perf_counter()
        if early_target_context is None:
            target_context = _dt_submit_current_tree_target()
        else:
            target_context = early_target_context
        target_output, target_started, target_profile_started, target_plans, target_rows, target_roots, target_request_ids, real_target_count, target_temperatures, target_execution_plans, target_execution_roots, target_execution_rows, target_execution_ids, target_tree_kwargs, target_submit_ended = target_context
        profile_end(target_profile_started, 'target_compute')
        target_original_host_phase_ended = time.perf_counter()
        if target_output is not None:
            torch.npu.synchronize()
            counters['spec_rhythm_target_query_tokens'] += target_output['query_count']
            counters['spec_rhythm_target_graph_padding_queries'] += target_output.get('graph_padding_query_count', 0)
            if target_output.get('used_tree_attention', False):
                counters['spec_rhythm_target_full_tree_rounds'] += 1
            else:
                counters['spec_rhythm_target_causal_tree_rounds'] += 1
            for phase, seconds in target_output.get('profile_seconds', {}).items():
                decode_profile_detail_seconds[phase] += float(seconds)
        target_ended = time.perf_counter()
        if early_target_context is not None:
            phase_seconds['target'] += target_ended - target_started
        else:
            phase_seconds['target'] += target_original_host_phase_ended - target_started
        target_window_observation_ms = _dt_tree_early_target_record(counters, early=early_target_context is not None, started=target_started, submitted=target_submit_ended, residual_started=target_residual_started, completed=target_ended)
        draft_postprocess_profile_started = profile_begin(profile_this_round)
        self._validate_spec_rhythm_target_graph(target_output, has_target_work=bool(target_indices))
        local_rows = None if draft_output is None else draft_output['draft_token_ids']
        selected_indices = None
        confidences = None
        if draft_output is not None:
            selected_indices = []
            selected_rows = []
            confidences = []
            exploratory = {}
            for row_id, (tree_plan, row) in enumerate(zip(work_plans, local_rows)):
                node_confidence = draft_output['node_confidences'][row_id]
                index = work_indices[row_id]
                runtime = states[index]
                cap = runtime.max_gamma or shaper.max_gamma
                if work_eager[row_id]:
                    cap = min(cap, effective_eager_cap)
                exploratory[index] = TreeCandidateRequest(parents=tree_plan.parent_indices.detach().cpu().tolist(), token_ids=row, conditional_confidences=node_confidence.detach().cpu().tolist(), max_candidates=min(cap, tree_plan.candidate_budget), progress_gap=budget_plan.progress_gaps.get(index, 0), urgency=max(0.0, runtime.urgency(projected_wait_ms)), acceptance_rate=runtime.acceptance_ema, minimum_candidates=0 if work_eager[row_id] else 1)
            with _trace_region(self, 'SpecSLO/GlobalCandidateSelection'):
                refined = _dt_select_backbone(exploratory, budget_plan.verification_roof)
            for row_id, index in enumerate(work_indices):
                indices = refined.selected_indices[index]
                selected_indices.append(list(indices))
                selected_rows.append([local_rows[row_id][node] for node in indices])
                confidences.append(sum((exploratory[index].conditional_confidences[node] for node in indices)) / len(indices) if indices else 0.0)
            if draft_output.get('materialization_deferred', False):
                eager_rows = [] if getattr(self, '_dynamic_tree_canonical_eager', False) else [row for row, is_eager in enumerate(work_eager) if is_eager]
                with _trace_region(self, 'SpecSLO/SelectedDraftKVMaterialize'):
                    materialized = self.materialize_selected_tree_kv([work_plans[row] for row in eager_rows], [draft_output['draft_token_ids'][row] for row in eager_rows], [selected_indices[row] for row in eager_rows], [work_indices[row] for row in eager_rows])
                if materialized is not None:
                    torch.npu.synchronize()
                    counters['spec_rhythm_draft_model_calls'] += materialized.get('model_calls', 0)
                    counters['spec_rhythm_draft_graph_calls'] += materialized.get('graph_calls', 0)
                    counters['spec_rhythm_draft_materialized_nodes'] += int(materialized.get('materialized_nodes', 0))
                    for phase, seconds in materialized.get('profile_seconds', {}).items():
                        decode_profile_detail_seconds[phase] += float(seconds)
            local_rows = selected_rows
        profile_detail_end(draft_postprocess_profile_started, 'draft_postprocess')
        profile_end(draft_profile_started, 'draft_compute')
        draft_ended = time.perf_counter()
        if host_trace is not None:
            host_trace['draft_start_seconds'] = draft_started
            host_trace['draft_end_seconds'] = draft_ended
            host_trace['target_start_seconds'] = target_started
            host_trace['target_end_seconds'] = target_ended
        if self.is_draft:
            phase_seconds['draft'] += draft_ended - draft_started
        exchange_started = time.perf_counter()
        exchange_profile_started = profile_begin(profile_this_round and bool(work_plans))
        frontier_rows = [draft_output['eager_frontier_tokens'].get(index) for index in work_indices] if draft_output is not None else None
        with torch.profiler.record_function('SpecSLO/DraftToTarget'):
            message = self._exchange_spec_rhythm_tree_candidates(local_rows, work_plans, frontier_rows, confidences, selected_indices=selected_indices, draft_compute_ms=(draft_ended - draft_started) * 1000.0 if self.rank == self.topology.draft_leader_rank and work_plans else 0.0)
        profile_end(exchange_profile_started, 'draft_to_target_communication')
        exchange_ended = time.perf_counter()
        phase_seconds['exchange'] += exchange_ended - exchange_started
        if host_trace is not None:
            host_trace['exchange_start_seconds'] = exchange_started
            host_trace['exchange_end_seconds'] = exchange_ended
        draft_ms = 0.0
        candidate_rows = confidence_rows = selected_rows = None
        if work_plans:
            assert message is not None
            split_started = time.perf_counter()
            if self.rank == self.topology.draft_leader_rank:
                assert local_rows is not None
                assert frontier_rows is not None
                assert confidences is not None
                assert selected_indices is not None
                candidate_rows = local_rows
                confidence_rows = confidences
                selected_rows = selected_indices
                draft_ms = (draft_ended - draft_started) * 1000.0
            else:
                candidate_rows, frontier_rows, confidence_rows, selected_rows, draft_ms = self._split_tree_candidates(message, work_plans)
            if host_trace is not None:
                host_trace['publish_candidate_split_seconds'] = time.perf_counter() - split_started
        if profile_this_round:
            timeline_profile_started = profile_begin(True)
            timing_values = [(draft_ended - draft_started) * 1000.0 if self.rank == self.topology.draft_leader_rank and work_plans else 0.0, (target_ended - target_started) * 1000.0 if self.rank == self.topology.target_leader_rank and target_indices else 0.0, draft_started if self.rank == self.topology.draft_leader_rank and work_plans else 0.0, draft_ended if self.rank == self.topology.draft_leader_rank and work_plans else 0.0, target_started if self.rank == self.topology.target_leader_rank and target_indices else 0.0, target_ended if self.rank == self.topology.target_leader_rank and target_indices else 0.0]
            timing = torch.tensor([round(value * (1000 if index < 2 else 1000000)) for index, value in enumerate(timing_values)], dtype=torch.int64, device=self.device)
            dist.all_reduce(timing)
            _, _, ds, de, ts, te = [value / (1000 if index < 2 else 1000000) for index, value in enumerate(timing.cpu().tolist())]
            decode_timeline.append({'step': round_count, 'draft_start': ds, 'draft_end': de, 'target_start': ts, 'target_end': te, 'host_compute_window_overlap_ms': max(0.0, min(de, te) - max(ds, ts)) * 1000.0, 'verify_candidates': actual_candidates, 'verify_requests': len(target_indices), 'draft_requests': len(work_plans), 'residual_draft_window_ms': window.residual_window_ms, 'eager_requests': len(eager_budgets)})
            profile_detail_end(timeline_profile_started, 'profile_timeline_collective')
        publish_profile_started = profile_begin(profile_this_round)
        if work_plans:
            publish_pack_seconds = 0.0
            publish_slot_mapping_seconds = 0.0
            publish_payload_seconds = 0.0
            publish_kv_move_seconds = 0.0
            publish_flat_mapping_seconds = 0.0
            publish_controller_seconds = 0.0
            counters['spec_rhythm_allocated_draft_tokens'] += sum((int(plan.candidate_budget) for plan in work_plans))
            assert candidate_rows is not None
            assert confidence_rows is not None
            assert selected_rows is not None
            mapping = None if draft_output is None else draft_output['cache_slot_mapping']
            cursor = 0
            published_tickets = []
            pending_draft_mappings: list[tuple[int, torch.Tensor]] = []
            normal_publish_sources: list[torch.Tensor] = []
            normal_publish_destinations: list[torch.Tensor] = []
            defer_normal_tree_materialization = bool(getattr(self, '_defer_normal_tree_materialization', False))
            for work_row, (index, ticket, tree_plan, row) in enumerate(zip(work_indices, tickets, work_plans, candidate_rows)):
                expected = int(tree_plan.width) * int(tree_plan.depth) + 1
                if not row:
                    cursor += expected
                    continue
                ticket.gamma = len(row)
                published_tickets.append(ticket)
                selection = selected_rows[work_row]
                publish_part_started = time.perf_counter()
                packed_plan = pack_selected_tree_plan(tree_plan, selection)
                publish_pack_seconds += time.perf_counter() - publish_part_started
                deferred_normal = (not ticket.eager or getattr(self, '_dynamic_tree_canonical_eager', False)) and defer_normal_tree_materialization
                if self.is_draft and deferred_normal:
                    counters['spec_rhythm_draft_publish_rows'] += 1
                    counters['spec_rhythm_draft_publish_skipped_rows'] += 1
                elif self.is_draft and mapping is not None:
                    identity_selection = list(selection) == list(range(len(selection)))
                    if identity_selection:
                        source = mapping[cursor:cursor + len(selection) + 1]
                    else:
                        source = mapping[cursor:cursor + expected].index_select(0, torch.tensor([0, *[node + 1 for node in selection]], dtype=torch.long, device=self.device))
                    if not ticket.eager and identity_selection:
                        counters['spec_rhythm_draft_publish_rows'] += 1
                        counters['spec_rhythm_draft_publish_skipped_rows'] += 1
                    elif not ticket.eager:
                        counters['spec_rhythm_draft_publish_rows'] += 1
                        publish_part_started = time.perf_counter()
                        destination = torch.tensor(self._cache_slot_mapping([index] * (len(selection) + 1), list(range(packed_plan.prefix_len, packed_plan.prefix_len + len(selection) + 1))), dtype=torch.int32, device=self.device)
                        publish_slot_mapping_seconds += time.perf_counter() - publish_part_started
                        normal_publish_sources.append(source)
                        normal_publish_destinations.append(destination)
                        source = destination
                        counters['spec_rhythm_draft_publish_moved_rows'] += 1
                    if source is not None:
                        pending_draft_mappings.append((ticket.proposal_id, source))
                elif self.is_draft:
                    raise RuntimeError('materialized draft tree is missing its KV mapping')
                eager_metadata: dict[str, Any] = {}
                publish_part_started = time.perf_counter()
                if ticket.eager:
                    parent_ticket = controller.ready.get(index)
                    parent_payload = None if parent_ticket is None else payloads.get(parent_ticket.proposal_id)
                    if parent_ticket is None or parent_payload is None:
                        raise RuntimeError('SpecRhythm eager tree parent disappeared before publish')
                    parent_plan = parent_payload['plan']
                    parent_row = parent_payload['row']
                    path_len = self._tree_primary_path_length(parent_plan)
                    eager_metadata = {'eager_parent_proposal_id': parent_ticket.proposal_id, 'eager_dependency_tokens': tuple((int(parent_row[node]) for node in tree_primary_path(parent_plan))), 'eager_frontier_token': frontier_rows[work_row] if frontier_rows is not None else None, 'eager_dependency_length': path_len}
                payloads[ticket.proposal_id] = {'ticket': ticket, 'plan': packed_plan, 'row': row, 'confidence': confidence_rows[work_row], 'draft_kv_materialized': not deferred_normal, **eager_metadata, 'draft_kv_deferred_canonical': bool(ticket.eager and deferred_normal and getattr(self, '_dynamic_tree_canonical_eager', False))}
                publish_payload_seconds += time.perf_counter() - publish_part_started
                cursor += expected
            if normal_publish_sources:
                publish_part_started = time.perf_counter()
                self._move_tree_cache_slots(torch.cat(normal_publish_sources), torch.cat(normal_publish_destinations))
                publish_kv_move_seconds += time.perf_counter() - publish_part_started
            publish_part_started = time.perf_counter()
            cache_mappings.update(pending_draft_mappings)
            controller.publish(published_tickets)
            counters['spec_rhythm_tree_nodes'] += sum((ticket.gamma for ticket in published_tickets))
            counters['spec_rhythm_unused_verification_tokens'] = budget_plan.verification_roof - sum((ticket.gamma for ticket in published_tickets))
            publish_controller_seconds += time.perf_counter() - publish_part_started
            if host_trace is not None:
                host_trace.update({'publish_pack_seconds': publish_pack_seconds, 'publish_slot_mapping_seconds': publish_slot_mapping_seconds, 'publish_payload_seconds': publish_payload_seconds, 'publish_kv_move_seconds': publish_kv_move_seconds, 'publish_flat_mapping_seconds': publish_flat_mapping_seconds, 'publish_controller_seconds': publish_controller_seconds})
        profile_detail_end(publish_profile_started, 'proposal_publish')
        if host_trace is not None:
            host_trace['publish_end_seconds'] = time.perf_counter()
        if not target_indices:
            accounting_profile_started = profile_begin(profile_this_round)
            _dt_cpu_barrier(self)
            elapsed_tensor = torch.tensor([time.perf_counter() - cycle_started if self.rank == self.topology.target_leader_rank else 0.0], dtype=torch.float64, device='cpu')
            _dt_cpu_broadcast(self, elapsed_tensor, src=self.topology.target_leader_rank, kind='cycle_clock')
            last_cycle_ms = float(elapsed_tensor.cpu().item()) * 1000.0
            for index in active:
                states[index].add_decode_time(last_cycle_ms)
            profile_detail_end(accounting_profile_started, 'cycle_accounting')
            if host_trace is not None:
                host_trace['cycle_end_seconds'] = time.perf_counter()
                host_timeline.append(host_trace)
            finish_profile_round(profile_round_started, profile_accounted_before)
            if profile_this_round:
                profiled_decode_steps += 1
            round_count += 1
            if npu_profiler is not None:
                npu_profiler.step()
            continue
        target_verdict: TreeVerificationOutput | None = None
        verdict_host_started = time.perf_counter() if host_trace is not None else None
        verdict_profile_started = profile_begin(profile_this_round and self.rank == self.topology.target_leader_rank)
        if self.rank == self.topology.target_leader_rank:
            assert target_output is not None
            if self.config.spec_rhythm_cpu_verdict:
                target_verdict = self._verify_tree_outputs_host(target_rows, target_plans, target_output['target_query_token_ids'], max((int(plan.depth) for plan in target_plans)), torch.device('cpu'))
            else:
                target_parents = torch.cat([plan.parent_indices[:int(plan.candidate_budget)].to(self.device) for plan in target_plans])
                draft_tokens = torch.tensor([token for row, plan in zip(target_rows, target_plans) for token in row[:int(plan.candidate_budget)]], dtype=torch.long, device=self.device)
                target_verdict = self.verify_tree_outputs(draft_tokens, target_parents, target_output['target_query_token_ids'], target_output['bonus_token_ids'], [int(plan.candidate_budget) for plan in target_plans], max((int(plan.depth) for plan in target_plans)))
        profile_end(verdict_profile_started, 'target_verdict')
        if host_trace is not None:
            host_trace['verdict_start_seconds'] = verdict_host_started or 0.0
            host_trace['verdict_end_seconds'] = time.perf_counter()
        broadcast_started = time.perf_counter()
        correction_profile_started = profile_begin(profile_this_round)
        with torch.profiler.record_function('SpecSLO/TargetToDraft'):
            output_rows, accepted_rows, target_ms = self._broadcast_spec_rhythm_tree_verdict(target_verdict, target_plans, target_window_observation_ms if self.rank == self.topology.target_leader_rank else 0.0)
        profile_end(correction_profile_started, 'target_to_draft_communication')
        broadcast_ended = time.perf_counter()
        phase_seconds['broadcast'] += broadcast_ended - broadcast_started
        if host_trace is not None:
            host_trace['correction_start_seconds'] = broadcast_started
            host_trace['correction_end_seconds'] = broadcast_ended
        window_estimator.observe(draft_compute_ms=draft_ms, drafted_tokens=len(work_plans) * exploration_cost, target_verify_ms=target_ms, eager_work=bool(eager_budgets), eager_rows=len(eager_budgets))
        state_started = time.perf_counter()
        state_profile_started = profile_begin(profile_this_round)
        finished: list[int] = []
        preflight_profile_started = time.perf_counter() if profile_this_round else None
        preflight_error: Exception | None = None
        pending_cache_mappings: dict[int, torch.Tensor] = {}
        try:
            for row, index in enumerate(target_indices):
                ticket = controller.validate_verification(index)
                if ticket.proposal_id != target_payloads[row]['ticket'].proposal_id:
                    raise RuntimeError('SpecRhythm tree verdict references a different proposal')
                plan_for_row = target_plans[row]
                if plan_for_row.prefix_len != len(local_states[index].token_ids) - 1:
                    raise RuntimeError('SpecRhythm tree verdict has a stale prefix')
                accepted_path = [value for value in accepted_rows[row] if value >= 0]
                tokens = [value for value in output_rows[row] if value >= 0]
                if len(tokens) != len(accepted_path) + 1:
                    raise RuntimeError('SpecRhythm tree verdict must include one frontier token')
                parents = plan_for_row.parent_indices.detach().cpu().tolist()
                parent = -1
                for depth, node in enumerate(accepted_path):
                    if node >= len(parents) or parents[node] != parent:
                        raise RuntimeError('SpecRhythm verdict path is not ancestor closed')
                    if tokens[depth] != target_rows[row][node]:
                        raise RuntimeError('SpecRhythm verdict token does not match its proposal')
                    parent = node
                staged = controller.staged_eager.get(index)
                if staged is not None and staged.proposal_id not in payloads:
                    raise RuntimeError('SpecRhythm staged eager tree payload disappeared')
            pending_cache_mappings, preflight_host_cache_mappings = self._preflight_spec_rhythm_tree_cache_commit(controller, payloads, cache_mappings, target_indices, target_plans, target_output, return_host_mappings=True)
        except Exception as error:
            preflight_error = error
        if host_trace is not None:
            host_trace['state_preflight_end_seconds'] = time.perf_counter()
        profile_detail_end(preflight_profile_started, 'commit_preflight')
        consensus_profile_started = time.perf_counter() if profile_this_round else None
        preflight_status = torch.cat((torch.tensor([int(preflight_error is not None)], dtype=torch.int64, device=self.device), self._spec_rhythm_nonfinite_flag(include_layer_cache=False).reshape(1).to(dtype=torch.int64)))
        preflight_status = _dt_cpu_reduce(self, preflight_status, op=dist.ReduceOp.MAX, kind='commit_consensus')
        structural_failed, numerical_failed = (int(value) for value in preflight_status.cpu().tolist())
        if host_trace is not None:
            host_trace['state_consensus_end_seconds'] = time.perf_counter()
        profile_detail_end(consensus_profile_started, 'commit_consensus')
        if structural_failed or numerical_failed or preflight_error is not None:
            numerical_detail = ''
            if numerical_failed:
                model = getattr(self, 'model', None)
                local_flags = []
                for owner, name in ((model, 'output_nonfinite'), (getattr(model, 'lm_head', None), 'logits_nonfinite')):
                    flag = getattr(owner, name, None)
                    local_flags.append(flag.reshape(1).to(dtype=torch.int64) if isinstance(flag, torch.Tensor) else torch.zeros(1, dtype=torch.int64, device=self.device))
                world_size = dist.get_world_size()
                rank_flags = torch.zeros(world_size * 2, dtype=torch.int64, device=self.device)
                rank_flags[self.rank * 2:self.rank * 2 + 2].copy_(torch.cat(local_flags))
                rank_flags = _dt_cpu_reduce(self, rank_flags, op=dist.ReduceOp.MAX, kind='failure_detail')
                detail_values = [int(value) for value in rank_flags.cpu().tolist()]
                numerical_detail = ', rank_flags=' + str({rank: {'output': detail_values[rank * 2], 'logits': detail_values[rank * 2 + 1]} for rank in range(world_size)})
            if preflight_error is not None:
                raise RuntimeError(f'SpecRhythm tree commit preflight failed on rank {self.rank}: {preflight_error}{numerical_detail}') from preflight_error
            raise RuntimeError(f'SpecRhythm tree commit preflight failed on another rank; this step was not committed (structural={structural_failed}, numerical={numerical_failed}{numerical_detail})')
        compaction_profile_started = time.perf_counter() if profile_this_round else None
        cache_mappings.update(pending_cache_mappings)
        compact_proposal_ids = [payload['ticket'].proposal_id for payload in target_payloads]
        rows_to_compact = [row for row, plan in enumerate(target_plans) if not (self.is_draft and getattr(self, '_defer_normal_tree_materialization', False) and (not target_payloads[row]['ticket'].eager or target_payloads[row].get('draft_kv_deferred_canonical', False))) and self._tree_row_requires_kv_compaction(accepted_rows[row], int(plan.candidate_budget))]
        counters['spec_rhythm_kv_compaction_rows'] += len(rows_to_compact)
        counters['spec_rhythm_kv_compaction_skipped_rows'] += len(target_plans) - len(rows_to_compact)
        if rows_to_compact:
            if any((compact_proposal_ids[row] not in preflight_host_cache_mappings for row in rows_to_compact)):
                raise RuntimeError('SpecRhythm compacted tree is missing its preflighted KV mapping')
            self.compact_tree_round(None, [accepted_rows[row] for row in rows_to_compact], [int(target_plans[row].candidate_budget) for row in rows_to_compact], host_slot_rows=[preflight_host_cache_mappings[compact_proposal_ids[row]] for row in rows_to_compact])
            counters['spec_rhythm_kv_compaction_rounds'] += 1
        if host_trace is not None:
            host_trace['state_compaction_end_seconds'] = time.perf_counter()
        profile_detail_end(compaction_profile_started, 'kv_compaction')
        request_commit_profile_started = time.perf_counter() if profile_this_round else None
        promoted_cache_rows: list[tuple[int, int, TreeSpeculationPlan]] = []
        for row, index in enumerate(target_indices):
            tree_plan = target_plans[row]
            was_first_decode_token = states[index].delivered_tokens == 0
            state_for_row = local_states[index]
            committed_tokens = _truncate_completion([value for value in output_rows[row] if value >= 0], self.eos_token_ids, state_for_row.max_tokens - len(state_for_row.committed_completion_token_ids), state_for_row.ignore_eos)
            will_finish = len(state_for_row.committed_completion_token_ids) + len(committed_tokens) >= state_for_row.max_tokens or (not state_for_row.ignore_eos and any((token in self.eos_token_ids for token in committed_tokens)))
            accepted_count = sum((value >= 0 for value in accepted_rows[row]))
            proposed_count = int(tree_plan.candidate_budget)
            fully_accepted = accepted_count >= self._tree_primary_path_length(tree_plan)
            if fully_accepted:
                counters['spec_rhythm_tree_full_accepts'] += 1
            else:
                counters['spec_rhythm_tree_rejections'] += 1
            counters['spec_rhythm_tree_accepted_nodes'] += accepted_count
            counters['spec_rhythm_verified_tokens'] += proposed_count
            current_ticket = target_payloads[row]['ticket']
            controller.validate_verification(index)
            eager_ticket = controller.staged_eager.get(index)
            eager_valid = fully_accepted
            if eager_ticket is not None:
                eager_payload = payloads.get(eager_ticket.proposal_id)
                if eager_payload is None:
                    raise RuntimeError('SpecRhythm staged eager tree payload disappeared')
                dependency_length = int(eager_payload.get('eager_dependency_length', 0))
                dependency_tokens = tuple((int(value) for value in eager_payload.get('eager_dependency_tokens', ())))
                observed_path = tuple((int(value) for value in output_rows[row][:dependency_length]))
                observed_frontier = int(output_rows[row][dependency_length]) if dependency_length < len(output_rows[row]) else -1
                eager_frontier = eager_payload.get('eager_frontier_token')
                eager_valid = eager_valid and eager_payload.get('eager_parent_proposal_id') == current_ticket.proposal_id and (dependency_length > 0) and (observed_path == dependency_tokens) and (eager_frontier is not None) and (observed_frontier == int(eager_frontier))
                if not eager_valid:
                    counters['spec_rhythm_tree_eager_dependency_rejections'] += 1
            promoted = controller.finish_verification(index, fully_accepted=eager_valid and (not will_finish), proposed_tokens=proposed_count, accepted_tokens=accepted_count, delivered_tokens=len(committed_tokens), draft_confidence=target_payloads[row]['confidence'], ema_alpha=self.config.spec_rhythm_acceptance_ema_alpha)
            for state in (draft_states[index], target_states[index]):
                state.apply_tree_verification(output_token_ids=committed_tokens, accepted=accepted_count, proposed_tokens=proposed_count)
            delivered = len(committed_tokens)
            if was_first_decode_token and delivered > 0:
                first_decode_indices.add(index)
            payloads.pop(current_ticket.proposal_id, None)
            if current_ticket.proposal_id in cache_mappings:
                cache_mappings.pop(current_ticket.proposal_id)
            if eager_ticket is not None and promoted is None:
                counters['spec_rhythm_tree_eager_invalidated'] += 1
                payloads.pop(eager_ticket.proposal_id, None)
                cache_mappings.pop(eager_ticket.proposal_id, None)
            elif eager_ticket is not None:
                counters['spec_rhythm_tree_eager_promoted'] += 1
                if self.is_draft:
                    eager_plan = payloads[eager_ticket.proposal_id]['plan']
                    if not payloads[eager_ticket.proposal_id].get('draft_kv_deferred_canonical', False):
                        promoted_cache_rows.append((eager_ticket.proposal_id, index, eager_plan))
            if _finished(local_states[index], self.eos_token_ids):
                finished.append(index)
            self._deliver_committed_tokens(index, request_params[index], local_states[index])
        if promoted_cache_rows:
            moved_slots = self._move_promoted_tree_cache_slots(promoted_cache_rows, cache_mappings, preflight_host_cache_mappings)
            counters['spec_rhythm_tree_eager_promotion_kv_move_rounds'] += 1
            counters['spec_rhythm_tree_eager_promotion_kv_move_rows'] += len(promoted_cache_rows)
            counters['spec_rhythm_tree_eager_promotion_kv_move_slots'] += moved_slots
        controller.finish_cycle(plan.target_home_batch_id)
        if host_trace is not None:
            host_trace['state_request_commit_end_seconds'] = time.perf_counter()
        profile_detail_end(request_commit_profile_started, 'request_commit')
        phase_seconds['state_update'] += time.perf_counter() - state_started
        profile_end(state_profile_started, 'state_update')
        if host_trace is not None:
            host_trace['state_end_seconds'] = time.perf_counter()
        accounting_profile_started = time.perf_counter() if profile_this_round else None
        elapsed_tensor = torch.tensor([time.perf_counter() - cycle_started if self.rank == self.topology.target_leader_rank else 0.0], dtype=torch.float64, device='cpu')
        _dt_cpu_broadcast(self, elapsed_tensor, src=self.topology.target_leader_rank, kind='cycle_clock')
        last_cycle_ms = float(elapsed_tensor.cpu().item()) * 1000.0
        for index in active:
            states[index].add_decode_time(last_cycle_ms)
        first_decode_indices.clear()
        for index in finished:
            completed[index] = local_states[index].clone()
            finished_decode_elapsed[index] = states[index].decode_elapsed_ms
            for ticket in (controller.ready.get(index), controller.staged_eager.get(index)):
                if ticket is not None:
                    payloads.pop(ticket.proposal_id, None)
                    cache_mappings.pop(ticket.proposal_id, None)
            controller.invalidate_request(index)
        release_online_cache(finished)
        active = [index for index in active if index not in set(finished)]
        if not live_open:
            admit_available(poll_live=False)
        profile_detail_end(accounting_profile_started, 'cycle_accounting')
        counters['spec_rhythm_tree_rounds'] += 1
        if host_trace is not None:
            host_trace['state_start_seconds'] = state_started
            host_trace['cycle_end_seconds'] = time.perf_counter()
            host_timeline.append(host_trace)
        finish_profile_round(profile_round_started, profile_accounted_before)
        if profile_this_round:
            profiled_decode_steps += 1
        round_count += 1
        if npu_profiler is not None:
            npu_profiler.step()
        if self.config.stop_after_profiled_decode_steps and round_count >= self.config.profile_decode_steps:
            break
    torch.npu.synchronize()
    if npu_profiler is not None:
        _stop_active_npu_profiler(self)
    self._profile_tree_subphases = False
    self._defer_tree_materialization = False
    decode_elapsed = time.perf_counter() - started
    self.last_worker_decode_phase_seconds = phase_seconds
    self.last_worker_decode_profile_seconds = dict(decode_profile_seconds)
    self.last_worker_decode_profile_detail_seconds = dict(decode_profile_detail_seconds)
    self.last_worker_decode_host_timeline = host_timeline
    self.last_worker_profiled_decode_steps = profiled_decode_steps
    self.last_worker_decode_counters = counters
    self._release_cache()
    if self.rank != self.topology.target_leader_rank:
        return None
    results: list[dict[str, Any]] = []
    for index, state in enumerate(target_states):
        state = completed.get(index, state)
        completion = _truncate_completion(state.committed_completion_token_ids, self.eos_token_ids, state.max_tokens, state.ignore_eos)
        request_decode_elapsed_ms = finished_decode_elapsed.get(index, states[index].decode_elapsed_ms)
        measured_decode_elapsed_ms = max(0.0, request_decode_elapsed_ms - (states[index].decode_start_elapsed_ms or 0.0))
        observed_tpot_ms = measured_decode_elapsed_ms / max(1, len(completion) - 1)
        paper_tpot_ms = measured_decode_elapsed_ms / max(1, len(completion))
        slo_attained = None if state.slo_tpot_ms is None else observed_tpot_ms <= state.slo_tpot_ms
        finish_reason = 'abort' if state.aborted else 'length' if len(completion) >= state.max_tokens else 'stop'
        results.append({'completion_token_ids': completion, 'request_id': request_params[index].request_id, 'accepted_draft_tokens': state.accepted_draft_tokens, 'verified_draft_tokens': state.verified_draft_tokens, 'verification_rounds': state.verification_rounds, 'num_acc_tokens': state.acceptance_lengths, 'mean_accept_tokens': sum(state.acceptance_lengths) / len(state.acceptance_lengths) if state.acceptance_lengths else 0.0, 'acceptance_rate': state.accepted_draft_tokens / state.verified_draft_tokens if state.verified_draft_tokens else 0.0, 'temperature': state.temperature, 'draft_temperature': state.draft_temperature, 'top_p': state.top_p, 'top_k': state.top_k, 'draft_top_p': state.draft_top_p, 'draft_top_k': state.draft_top_k, 'max_tokens': state.max_tokens, 'ignore_eos': state.ignore_eos, 'slo_tpot_ms': state.slo_tpot_ms, 'slo_class': state.slo_class, 'arrival_ts': request_params[index].arrival_ts, 'finish_reason': finish_reason, 'observed_tpot_ms': observed_tpot_ms, 'tpot_definition': 'decode_after_first_token_ms / (output_tokens - 1)', 'paper_tpot_ms': paper_tpot_ms, 'paper_tpot_definition': 'same_decode_elapsed_ms / output_tokens', 'paper_slo_attained': None if state.slo_tpot_ms is None else paper_tpot_ms <= state.slo_tpot_ms, 'paper_slo_goodput_tokens': len(completion) if state.slo_tpot_ms is None or paper_tpot_ms <= state.slo_tpot_ms else 0, 'slo_attained': slo_attained, 'slo_goodput_tokens': len(completion) if slo_attained is not False else 0, 'round_count': round_count, 'decode_phase_seconds': dict(phase_seconds), 'spec_rhythm': dict(counters), 'elapsed_seconds': prefill_elapsed + decode_elapsed, 'prefill_elapsed_seconds': prefill_elapsed, 'decode_elapsed_seconds': decode_elapsed, 'tree_mode': True, 'gamma': self.gamma, 'decode_timeline': decode_timeline, 'tree_width': self.config.spec_rhythm_tree_width, 'tree_depth': self.config.spec_rhythm_tree_depth, **self._result_graph_metrics()})
    return results
