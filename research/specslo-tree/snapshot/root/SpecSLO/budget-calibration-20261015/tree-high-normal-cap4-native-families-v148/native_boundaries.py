"""Concrete original Native/Gloo bindings for the BLOCKED V75 prototype.

Nothing imports torch, initializes a model or installs hooks on module import.
These routines are unexecuted source. Gloo can coordinate ranks that reach a
boundary; it cannot recover a rank hung inside a failed device/TP collective.
The original supervisor timeout and four-card cleanup remain necessary.
"""
from dataclasses import asdict
import hashlib
import uuid

from atomic_prefill import TreeCommitFence, prompt_digest
from diagnostic_authorization import require_bounded_scope


class BoundaryRejected(RuntimeError):
    pass


class NativeBoundaries:
    def __init__(self, engine):
        require_bounded_scope()
        import torch.distributed as dist
        self.engine = engine
        self.group = engine.groups.verification_coordination_group
        if (self.group is None or str(dist.get_backend(self.group)) != 'gloo'
                or dist.get_world_size(self.group) != 4
                or tuple(sorted(engine.topology.verification_ranks)) != tuple(range(4))
                or len(engine.topology.draft_ranks) != 1):
            raise BoundaryRejected('V75 requires the ORIGINAL complete four-rank Gloo group')
        self.events = []  # RAM only, never a disk write in a timed call.
        self.sequence = 0
        self.activation_epochs = {}
        self.allocation_object = engine.cache_allocation
        self._original_activate = engine._activate_cache_sequence
        self._activation_override_existed = '_activate_cache_sequence' in engine.__dict__
        self._activation_override = engine.__dict__.get('_activate_cache_sequence')
        self.prefix_before = None
        self.prefix_lengths = None
        self.released_counts = {}
        self.last_draft_prompt = None

    def install_activation_tracking(self):
        engine = self.engine
        if engine.cache_allocation is None or engine.cache_block_tables is None:
            raise BoundaryRejected('Original allocation must exist before source lease binding')
        for index, table in enumerate(engine.cache_allocation.block_tables):
            if any(page >= 0 for page in table):
                self.activation_epochs[index] = 1
        def observed_activation(sequence_id, prompt, **kwargs):
            # An epoch advances only AFTER an actual original allocator call.
            result = self._original_activate(sequence_id, prompt, **kwargs)
            self.activation_epochs[sequence_id] = self.activation_epochs.get(sequence_id, 0) + 1
            return result
        engine._activate_cache_sequence = observed_activation

    def stage(self, name, callback):
        """Catch reachable local errors before every rank enters the same vote."""
        import torch.distributed as dist
        value = None
        error = None
        try:
            value = callback()
            # Untimed diagnostic stage: no target projection may remain in
            # flight when the shared CPU/Gloo status rendezvous begins.
            self._synchronize()
        except BaseException as exc:
            error = type(exc).__name__+': '+str(exc)
        rows = [None] * 4
        dist.all_gather_object(rows, {
            'rank': self.engine.rank, 'sequence': self.sequence, 'stage': name,
            'error': error}, group=self.group)
        sequence = self.sequence
        self.sequence += 1
        self.events.append({'sequence': sequence, 'stage': name, 'ranks': rows})
        if (any(not isinstance(row, dict) or row.get('stage') != name
                or row.get('sequence') != sequence or row.get('error') is not None for row in rows)
                or sorted(row.get('rank') for row in rows) != list(range(4))):
            raise BoundaryRejected('All-rank original boundary rejected '+name+': '+repr(rows))
        return value

    def _gather(self, value):
        import torch.distributed as dist
        rows = [None] * 4
        dist.all_gather_object(rows, value, group=self.group)
        return rows

    def _synchronize(self):
        import torch
        if self.engine.device.type != 'npu':
            raise BoundaryRejected('A CPU fake is not actual Native stream completion')
        torch.npu.synchronize()

    def cycle_preflight_all4(self, descriptor):
        rows = self._gather(descriptor)
        self.stage('mixed-cycle-shape-consensus', lambda: self._require(
            all(row == rows[0] for row in rows), 'Mixed geometry/cursor disagrees across ranks'))

    def draft_prompt_local(self, row, chunk):
        def preflight():
            engine = self.engine
            if (engine.config.enable_prefix_caching
                    or min(128, engine.config.max_num_batched_tokens) < chunk.end-chunk.start
                    or engine.cache_allocation.num_cached_tokens[row.lease.sequence_id] != 0
                    or tuple(page for page in engine.cache_allocation.block_tables[row.lease.sequence_id]
                             if page >= 0) != row.lease.pages):
                raise BoundaryRejected('Original native chunk config/physical lease changed')
        self.stage('original-draft-chunk-preflight', preflight)
        def forward():
            if self.engine.is_draft:
                import torch
                tokens = list(row.prompt[chunk.start:chunk.end])
                hidden = self.engine._run_packed_hidden(tokens, [row.lease.sequence_id]*len(tokens),
                    list(range(chunk.start, chunk.end)), use_aclgraph=False,
                    use_fused_infer_attention=True)
                saved = hidden.detach().clone()
                _, metadata = self.engine._prepare_attention_metadata(
                    [row.lease.sequence_id]*len(tokens), list(range(chunk.start, chunk.end)), True)
                from independent_history_v75 import query_KV
                values = query_KV(self.engine, metadata)
                if not bool(torch.isfinite(saved).all().item()) or any(
                        not bool(torch.isfinite(value).all().item()) for value in values):
                    raise BoundaryRejected('Actual draft prompt chunk hidden/query KV is nonfinite')
                self.last_draft_prompt = {'hidden': saved, 'query_KV': values,
                    'owner': row.lease.sequence_id, 'start': chunk.start, 'end': chunk.end}
        return self.stage('original-draft-prompt-local-forward', forward)

    def reservation_context_all4(self, request_index, active, state, pending, capacity):
        def observe():
            engine = self.engine
            allocation = engine.cache_allocation
            if allocation is not self.allocation_object or engine.config.enable_prefix_caching:
                raise BoundaryRejected('Reservation allocation changed or uses prefix caching')
            if allocation.num_cached_tokens[request_index] != 0:
                raise BoundaryRejected('Chunk absolute cursor cannot reuse a prefix-cache hit')
            table = allocation.block_tables[request_index]
            pages = tuple(page for page in table if page >= 0)
            # Include ALL other allocated rows, not only active service rows:
            # future pending prompts, graph scratch and mixed scratch all own KV.
            live = tuple(page for index, row in enumerate(allocation.block_tables)
                         if index != request_index for page in row if page >= 0)
            if (not pages or len(pages) != len(set(pages)) or set(pages) & set(live)
                    or self.activation_epochs.get(request_index, 0) <= 0):
                raise BoundaryRejected('Actual allocator has no distinct observed owner epoch')
            return {'rank': engine.rank, 'request_index': request_index,
                    'active': tuple(active), 'allocation_epoch': self.activation_epochs[request_index],
                    'pending': tuple(pending), 'capacity': capacity,
                    'prompt_sha256': prompt_digest(state.token_ids),
                    'prompt_length': len(state.token_ids),
                    'committed_length': state.committed_length,
                    'eligible_greedy': state.temperature == 0 and state.top_p >= 1.0 and state.top_k <= 0,
                    'reserved_pages': pages, 'live_pages': live}
        local = self.stage('reservation-original-allocator', observe)
        packets = self._gather(local)
        logical = {(p['request_index'], p['active'], p['allocation_epoch'], p['pending'], p['capacity'],
                    p['prompt_sha256'], p['prompt_length'], p['committed_length'], p['eligible_greedy'])
                   for p in packets}
        self.stage('reservation-logical-consensus', lambda: self._require(len(logical) == 1,
            'Owner/active/observed allocation epoch disagrees across ranks'))
        # Physical page IDs are role-local and are deliberately NOT equated.
        import torch.distributed as dist
        nonce = [uuid.uuid4().hex if self.engine.rank == self.engine.topology.target_leader_rank else None]
        dist.broadcast_object_list(nonce, src=self.engine.topology.target_leader_rank, group=self.group)
        self.events.append({'stage': 'reservation-lease-packets', 'ranks': packets, 'nonce': nonce[0]})
        return dict(local, nonce=nonce[0])

    @staticmethod
    def _require(condition, message):
        if condition is not True:
            raise BoundaryRejected(message)

    def _prefix_hashes(self, lengths):
        """Actual all-layer used-prefix bytes + separate finite gate (debug cost)."""
        import torch
        engine = self.engine
        allocation = engine.cache_allocation
        block_size = int(engine.config.kvcache_block_size)
        hashes = {}
        for layer_id, layer in enumerate(engine.model.layers):
            for name in ('key_cache', 'value_cache'):
                storage = getattr(layer.self_attn, name).flatten(0, 1)
                for owner, length in sorted(lengths.items()):
                    if length == 0:
                        continue
                    table = allocation.block_tables[owner]
                    pages = table[:(length+block_size-1)//block_size]
                    if any(page < 0 for page in pages):
                        raise BoundaryRejected('Protected prefix has an unallocated real page')
                    slots = [table[pos//block_size]*block_size+pos%block_size for pos in range(length)]
                    value = storage.index_select(0, torch.tensor(slots, dtype=torch.long, device=storage.device))
                    if not bool(torch.isfinite(value).all().item()):
                        raise BoundaryRejected('Protected used-prefix KV contains a nonfinite element')
                    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
                    hashes[f'{layer_id}:{name}:{owner}'] = hashlib.sha256(raw).hexdigest()
        return hashes

    def begin_prefix_protection(self, plans, sequence_ids, reservation):
        lengths = {owner: int(plan.prefix_len) for owner, plan in zip(sequence_ids, plans, strict=True)}
        lengths[reservation.lease.sequence_id] = reservation.committed_cursor
        self.prefix_lengths = lengths
        self.events.append({'stage': 'declared-prefix-protection-scope',
            'materialized_prefix_lengths': dict(lengths),
            'scope': 'current target owners before root plus reserved committed cursor',
            'all_active_full_KV_hash_claimed': False,
            'actual_model_layer_count': len(self.engine.model.layers)})
        self.prefix_before = self.stage('mixed-original-prefix-before', lambda: self._prefix_hashes(lengths))

    def protected_prefix_finite_all4(self, reservation):
        def compare():
            if self.prefix_before is None or self.prefix_lengths is None:
                raise BoundaryRejected('Missing actual pre-forward source-prefix snapshot')
            after = self._prefix_hashes(self.prefix_lengths)
            self._require(after == self.prefix_before, 'Mixed work changed protected original prefix bytes')
            return {'prefix_byte_hashes_equal': True, 'all_used_prefix_finite': True}
        self.stage('mixed-original-prefix-after', compare)
        return True  # Produced ONLY after the actual comparison + all4 vote.

    def commit_fence_all4(self, chunk, cycle_id, *, structural_failed, numerical_failed,
                          preflight_error, controller, target_home):
        def verify_and_fence():
            if (type(structural_failed) is not int or structural_failed != 0
                    or type(numerical_failed) is not int or numerical_failed != 0
                    or preflight_error is not None or chunk.cycle_id != cycle_id
                    or target_home not in (0, 1)
                    or controller.next_target_home_batch_id != 1-target_home):
                raise BoundaryRejected('Actual original consensus/finish-cycle evidence is incomplete')
            # Reaching the finish_cycle call alone does not prove KV completion.
            self._synchronize()
            return {'cycle_id': cycle_id, 'structural_failed': structural_failed,
                    'numerical_failed': numerical_failed, 'finished_home': target_home,
                    'next_home': controller.next_target_home_batch_id,
                    'local_compaction_stream_fenced': True}
        local = self.stage('original-finish-cycle-physical-fence', verify_and_fence)
        rows = self._gather(local)
        self.stage('original-finish-cycle-logical-consensus', lambda: self._require(
            all(row == rows[0] for row in rows), 'Original cycle commit differs across ranks'))
        self.events.append({'stage': 'original-tree-commit-evidence', 'ranks': rows})
        return TreeCommitFence(cycle_id, True, True, True, True)

    def publication_preflight_all4(self, row, active, pending, capacity, local_states):
        def validate():
            index = row.lease.identity.request_index
            state = local_states[index]
            if (index in active or index not in pending or len(active) >= capacity
                    or tuple(state.token_ids) != row.prompt
                    or state.committed_length != len(row.prompt)
                    or row.committed_cursor != len(row.prompt)
                    or row.prepared_token is None or getattr(state, 'aborted', False)):
                raise BoundaryRejected('Original admission publication preflight failed')
            self._synchronize()
            return {'identity': asdict(row.lease.identity), 'first_token': row.prepared_token,
                    'prompt_sha256': hashlib.sha256(repr(row.prompt).encode()).hexdigest()}
        value = self.stage('original-publication-preflight', validate)
        values = self._gather(value)
        self.stage('original-publication-token-consensus', lambda: self._require(
            all(v == values[0] for v in values), 'Final prompt token/identity differs across ranks'))

    def original_cache_invalidation(self, disposition, release_online_cache=None):
        def invalidate():
            self._synchronize()
            index = disposition.identity.request_index
            if self.activation_epochs.get(index) != disposition.identity.allocation_epoch:
                raise BoundaryRejected('Cannot release a newer physical allocation epoch')
            table = self.engine.cache_allocation.block_tables[index]
            if any(page >= 0 for page in table):
                if release_online_cache is None:
                    released = self.engine._release_cache_sequence(index)
                else:
                    before = sum(page >= 0 for page in table)
                    release_online_cache((index,))
                    released = before
            else:
                released = 0
            self._synchronize()
            if any(page != -1 for page in table):
                raise BoundaryRejected('Original host page row remained allocated after invalidation')
            import torch
            if not bool((self.engine.cache_block_tables[index] == -1).all().item()):
                raise BoundaryRejected('Original device page row remained allocated after invalidation')
            ledger = getattr(self.engine, '_dynamic_tree_known_KV', None)
            old_claim = None
            if self.engine.is_draft:
                if not isinstance(ledger, dict):
                    raise BoundaryRejected('Draft invalidation lacks the actual known-KV ledger')
                old_claim = ledger.pop(index, None)
            self.released_counts[index] = released
            return {'owner': index, 'released_blocks': released, 'physical_row_invalidated': True,
                    'invalidated_draft_claim': old_claim,
                    'other_owner_claims_modified': False}
        return self.stage('original-reservation-physical-invalidation', invalidate)

    def close(self):
        # Restore instance/class binding exactly; never leave activation tracing
        # attached to a future unrelated generation.
        if self._activation_override_existed:
            self.engine._activate_cache_sequence = self._activation_override
        else:
            self.engine.__dict__.pop('_activate_cache_sequence', None)
        if getattr(self.engine, '_v75_atomic_boundaries', None) is self:
            del self.engine._v75_atomic_boundaries
        active = getattr(self.engine, '_v78_atomic_hooks', None)
        if active is not None and getattr(active.original.close_original_bindings, '__self__', None) is self:
            del self.engine._v78_atomic_hooks

    def fatal_release_all(self):
        def release():
            self._synchronize()
            self.engine._release_cache()
            self._synchronize()
            if self.engine.cache_allocation is not None or self.engine.cache_block_tables is not None:
                raise BoundaryRejected('Original generation cache release did not complete')
        return self.stage('fatal-original-generation-cache-release', release)
