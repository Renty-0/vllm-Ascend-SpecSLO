"""Live host boundaries; original Native chunk/commit/publication fences remain.

Host status agreement does not imply stream completion. Prefix protection is
an allocation/slot write-footprint proof, never a comparison of KV contents or
a numerical qualification receipt. No observer or runtime installs on import.
"""
from live_allocation_tracker import LiveAllocationTracker
from native_boundaries import BoundaryRejected, NativeBoundaries


def _protected_page_limits(prefix_lengths, current):
    """Exact union of protected [0, limit) offsets on each 128-token page.

    Different logical pages/owners may name the same physical page: union
    then means maximum limit, never replacement or an owner-qualified key.
    Allocation/epoch/alias validation remains the caller's responsibility.
    """
    protected = {}
    for owner, length in prefix_lengths.items():
        pages = current[owner]
        if length > len(pages)*128:
            raise BoundaryRejected('Protected committed prefix lacks actual allocated pages')
        for index in range((length+127)//128):
            page = pages[index]
            limit = min(128, length-index*128)
            protected[page] = max(protected.get(page, 0), limit)
    return protected


class LiveNativeBoundaries(NativeBoundaries):
    def __init__(self, engine, *, allocation_tracker=None):
        super().__init__(engine)
        tracker = allocation_tracker if allocation_tracker is not None else LiveAllocationTracker(engine)
        if not isinstance(tracker, LiveAllocationTracker) or tracker.engine is not engine:
            raise BoundaryRejected('Live boundaries require this engine\'s actual allocation tracker')
        self.allocation_tracker = tracker
        self._tracking_started = False
        self._closed = False
        self._prefix_reservation = None
        self._prefix_identity = None
        self._prefix_prompt = None
        self._live_descriptor = None
        self._write_footprint = None

    def install_activation_tracking(self):
        self.allocation_tracker.install()
        self.activation_epochs = self.allocation_tracker.activation_epochs
        self._tracking_started = True

    def stage(self, name, callback):
        """Agree reachable host errors without an unconditional device fence."""
        import torch.distributed as dist
        value, error = None, None
        try:
            value = callback()
        except BaseException as exc:
            error = type(exc).__name__+': '+str(exc)
        rows = [None]*4
        sequence = self.sequence
        dist.all_gather_object(rows, dict(rank=self.engine.rank, sequence=sequence,
            stage=name, error=error), group=self.group)
        self.sequence += 1
        self.events.append(dict(sequence=sequence, stage=name, ranks=rows,
            boundary_kind='host-status-agreement'))
        if (any(not isinstance(row, dict) or row.get('stage') != name
                or row.get('sequence') != sequence or row.get('error') is not None for row in rows)
                or sorted(row.get('rank') for row in rows) != list(range(4))):
            raise BoundaryRejected('Live all-rank host boundary rejected '+name+': '+repr(rows))
        return value

    def reservation_context_all4(self, request_index, active, state, pending, capacity):
        # Keep the original source/greedy/capacity/nonce consensus. The added
        # stage checks the real tracker owner before the original table reader.
        self.stage('live-reservation-observed-owner', lambda:
            self.allocation_tracker.snapshot(request_index))
        return super().reservation_context_all4(request_index, active, state, pending, capacity)

    @staticmethod
    def _integer(value, name, maximum):
        if type(value) is not int or not 0 <= value <= maximum:
            raise BoundaryRejected('Invalid live '+name)
        return value

    @staticmethod
    def _positions(plan):
        values = getattr(plan, 'cache_positions', None)
        if (getattr(getattr(values, 'device', None), 'type', None) != 'cpu'
                or getattr(values, 'ndim', None) != 1):
            raise BoundaryRejected('Write-footprint positions must already be a CPU vector')
        positions = tuple(values.tolist())
        if (not positions or len(set(positions)) != len(positions)
                or any(type(position) is not int or not 0 <= position < 4096 for position in positions)):
            raise BoundaryRejected('Invalid or repeated live physical query position')
        return positions

    def begin_prefix_protection(self, plans, sequence_ids, reservation):
        def declare():
            plans_, ids = tuple(plans), tuple(sequence_ids)
            if (not plans_ or len(plans_) != len(ids) or len(ids) != len(set(ids))
                    or any(type(owner) is not int or owner < 0 for owner in ids)):
                raise BoundaryRejected('Protected normal plans/owners are not aligned')
            owner = reservation.lease.sequence_id
            if owner in ids or reservation.lease.identity.request_index != owner:
                raise BoundaryRejected('Protected prompt and normal owners collide')
            prompt = self.allocation_tracker.snapshot(owner)
            if (prompt.allocation_epoch != reservation.lease.identity.allocation_epoch
                    or prompt.pages != tuple(reservation.lease.pages)):
                raise BoundaryRejected('Protected prompt lease is stale')
            lengths = {index: self._integer(plan.prefix_len, 'normal committed prefix', 4095)
                       for index, plan in zip(ids, plans_, strict=True)}
            lengths[owner] = self._integer(reservation.committed_cursor, 'prompt committed cursor', len(reservation.prompt))
            self.prefix_lengths = lengths
            self._prefix_reservation = reservation
            self._prefix_identity = reservation.lease.identity
            self._prefix_prompt = tuple(reservation.prompt)
            self._normal_ids = ids
            self._live_descriptor = self._write_footprint = None
            self.events.append(dict(stage='live-prefix-write-scope',
                protected_prefix_lengths=dict(lengths),
                scope='current target prefixes and reserved committed prompt cursor',
                protection_kind='structural-write-footprint', runtime_KV_bytes_compared=False))
        return self.stage('live-prefix-write-declaration', declare)

    def _footprint(self, descriptor):
        from live_mixed_metadata import LiveMixedDescriptor, OwnerLease
        if not isinstance(descriptor, LiveMixedDescriptor) or self.prefix_lengths is None:
            raise BoundaryRejected('Actual live descriptor and declared prefixes are required')
        row = self._current_prefix_reservation(descriptor)
        layout = descriptor.flexible.layout
        n = len(descriptor.target_ids)
        segments=19 if n<=16 else 35
        if (descriptor.allocation_object is not self.engine.cache_allocation
                or type(descriptor.generation) is not int
                or descriptor.generation != self.engine._dynamic_tree_generation
                or descriptor.identity != self._prefix_identity
                or row.lease.identity != self._prefix_identity or tuple(row.prompt) != self._prefix_prompt
                or descriptor.chunk.identity != descriptor.identity
                or descriptor.chunk.cycle_id != descriptor.cycle_id
                or descriptor.previous_committed_cycle != row.last_committed_cycle_id
                or descriptor.target_ids != self._normal_ids
                or not 1 <= n <= 32 or descriptor.flexible.envelope.physical_S != segments
                or descriptor.chunk.start != self.prefix_lengths[row.lease.sequence_id]
                or row.committed_cursor != descriptor.chunk.start
                or not descriptor.chunk.start < descriptor.chunk.end <= len(row.prompt)
                or len(layout.plans) != segments or len(layout.sequence_ids) != segments
                or len(descriptor.owner_leases) != segments or len(set(layout.sequence_ids)) != segments
                or tuple(layout.sequence_ids[:n]) != self._normal_ids
                or layout.sequence_ids[n] != row.lease.sequence_id or layout.total_queries != 197):
            raise BoundaryRejected('Stale live descriptor/reservation or write owner geometry')
        size = self.engine.config.kvcache_block_size
        if type(size) is not int or size != 128:
            raise BoundaryRejected('Live write proof requires original 128-token pages')
        current = {}
        for index, lease in enumerate(descriptor.owner_leases):
            if (not isinstance(lease, OwnerLease) or not lease.pages
                    or lease.sequence_id != layout.sequence_ids[index]):
                raise BoundaryRejected('Descriptor lease order differs from actual query owners')
            observed = self.allocation_tracker.snapshot(lease.sequence_id)
            append_only_draft = self.engine.is_draft and index < n
            pages_match = (observed.pages[:len(lease.pages)] == lease.pages
                if append_only_draft else observed.pages == lease.pages)
            if observed.allocation_epoch != lease.allocation_epoch or not pages_match:
                raise BoundaryRejected('Live write owner activation epoch or lease is stale')
            current[lease.sequence_id] = observed.pages
        protected = _protected_page_limits(self.prefix_lengths, current)
        writes, physical_count = [], 0
        for index, (owner, plan) in enumerate(zip(layout.sequence_ids, layout.plans, strict=True)):
            prefix = self.prefix_lengths.get(owner, 0)
            if index <= n and (type(plan.prefix_len) is not int or plan.prefix_len != prefix):
                raise BoundaryRejected('Actual query plan changed its protected committed prefix')
            positions = self._positions(plan)
            physical_count += len(positions)
            if any(position < prefix for position in positions):
                raise BoundaryRejected('Actual query writes below a protected committed prefix')
            if index == n and positions != tuple(range(descriptor.chunk.start, descriptor.chunk.end)):
                raise BoundaryRejected('Draft/target prompt writes differ from its exact live Chunk')
            # The draft does not execute the mixed target's normal or scratch
            # queries. Its one eager prompt chunk is the only write in scope.
            if self.engine.is_draft and index != n:
                continue
            pages = current[owner]
            for position in positions:
                if position//size >= len(pages):
                    raise BoundaryRejected('Actual query write lacks an allocated page')
                page, offset = pages[position//size], position%size
                slot = page*size+offset
                if offset < protected.get(page, 0):
                    raise BoundaryRejected('Query write aliases a protected committed prefix slot')
                writes.append((owner, position, slot))
        slots = [value[2] for value in writes]
        if physical_count != 197 or len(slots) != len(set(slots)) or not writes:
            raise BoundaryRejected('Actual query writes alias one another or are absent')
        return tuple(writes)

    def _current_prefix_reservation(self, descriptor):
        """Accept only the declared reservation's exact immutable transition.

        AtomicPrefill.submitted replaces the RESERVED value before any KV
        write. The manager's current value must preserve the original lease,
        prompt and committed history; its only permitted changes here are the
        SUBMITTED phase, this descriptor's chunk and the physical-write bit.
        The metadata predicate also rejects prepared/adopted/stale states.
        """
        from dataclasses import replace
        from atomic_prefill import Phase, Reservation
        from live_mixed_metadata import _reservation
        protected = self._prefix_reservation
        if (not isinstance(protected, Reservation) or protected.phase is not Phase.RESERVED
                or protected.chunk is not None):
            raise BoundaryRejected('Stale live descriptor/reservation declaration')
        try:
            row = _reservation(descriptor.transactions, descriptor.chunk, allow_submitted=True)
        except (ValueError, TypeError, AttributeError) as error:
            raise BoundaryRejected('Stale live descriptor/reservation: '+str(error)) from error
        if row.phase is Phase.RESERVED:
            expected = protected
        elif row.phase is Phase.SUBMITTED:
            expected = replace(protected, phase=Phase.SUBMITTED,
                chunk=descriptor.chunk, has_physical_writes=True)
        else:
            raise BoundaryRejected('Stale live descriptor/reservation phase')
        if row != expected:
            raise BoundaryRejected('Stale live descriptor/reservation preserved fields changed')
        return row

    def bind_live_write_footprint(self, descriptor):
        """Host-only: call inside common preflight after real descriptor binding.

        No Gloo/WORLD/device operation occurs here. The caller's existing host
        stage agrees errors before asymmetric target submission begins.
        """
        writes = self._footprint(descriptor)
        self._live_descriptor, self._write_footprint = descriptor, writes
        self.events.append(dict(stage='live-query-write-footprint-bound',
            cycle_id=descriptor.cycle_id, query_write_count=len(writes),
            protected_prefix_lengths=dict(self.prefix_lengths),
            runtime_KV_bytes_compared=False))
        return writes

    def _revalidate_write_footprint(self, reservation):
        from atomic_prefill import Phase
        if self._live_descriptor is None or self._write_footprint is None:
            raise BoundaryRejected('Missing actual pre-forward live write-footprint binding')
        if reservation is not self._live_descriptor.transactions.current:
            raise BoundaryRejected('Live write proof requires the descriptor manager current reservation')
        if reservation is None or reservation.phase is not Phase.SUBMITTED:
            raise BoundaryRejected('Original live chunk join requires the current SUBMITTED reservation')
        current = self._footprint(self._live_descriptor)
        if current != self._write_footprint:
            raise BoundaryRejected('Live query write footprint changed after preflight')
        return current

    def protected_prefix_finite_all4(self, reservation):
        """Legacy callback name; Native's original chunk vote owns finiteness.

        This callback proves only structural non-overwrite from current host
        leases/slots. It neither inspects bytes nor claims a runtime KV match.
        """
        writes = self.stage('live-prefix-structural-write-proof-at-original-join', lambda:
            self._revalidate_write_footprint(reservation))
        return len(writes) > 0

    def draft_prompt_local(self, row, chunk):
        def preflight():
            self._revalidate_write_footprint(row)
            lease = self.allocation_tracker.snapshot(row.lease.sequence_id)
            if (self.engine.config.enable_prefix_caching
                    or min(128, self.engine.config.max_num_batched_tokens) < chunk.end-chunk.start
                    or self.engine.cache_allocation.num_cached_tokens[row.lease.sequence_id] != 0
                    or lease.pages != tuple(row.lease.pages)
                    or lease.allocation_epoch != row.lease.identity.allocation_epoch
                    or chunk != self._live_descriptor.chunk):
                raise BoundaryRejected('Original live draft chunk config/lease changed')
        self.stage('original-draft-chunk-preflight', preflight)
        def forward():
            if self.engine.is_draft:
                tokens = list(row.prompt[chunk.start:chunk.end])
                self.engine._run_packed_hidden(tokens, [row.lease.sequence_id]*len(tokens),
                    list(range(chunk.start, chunk.end)), use_aclgraph=False,
                    use_fused_infer_attention=True)
                # Original Native sticky finite flags and the subsequent exact
                # chunk finite/fence helper own completion and numerical checks.
                self.last_draft_prompt = dict(owner=row.lease.sequence_id, start=chunk.start, end=chunk.end)
        return self.stage('original-draft-prompt-local-forward', forward)

    def close(self):
        self.allocation_tracker.close()
        self._closed = True
        self._prefix_reservation = self._live_descriptor = self._write_footprint = None
        if getattr(self.engine, '_v75_atomic_boundaries', None) is self:
            del self.engine._v75_atomic_boundaries
        for name in ('_v78_atomic_hooks', '_v81_live_hooks'):
            active = getattr(self.engine, name, None)
            original = getattr(active, 'original', None)
            if getattr(getattr(original, 'close_original_bindings', None), '__self__', None) is self:
                delattr(self.engine, name)

    def original_cache_invalidation(self, disposition, release_online_cache=None):
        def observed_lease():
            lease = self.allocation_tracker.snapshot(disposition.identity.request_index)
            if lease.allocation_epoch != disposition.identity.allocation_epoch:
                raise BoundaryRejected('Cannot invalidate a different observed allocation epoch')
        self.stage('live-reservation-lease-before-original-invalidation', observed_lease)
        return super().original_cache_invalidation(disposition, release_online_cache)
