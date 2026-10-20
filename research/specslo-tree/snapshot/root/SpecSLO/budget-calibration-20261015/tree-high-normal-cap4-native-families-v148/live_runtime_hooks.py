"""Source-only live descriptor adapter for the existing original boundaries.

Nothing installs on import. Actual entry still requires the source-bound local
bounded permit; an explicit partition allowlist is only an input restriction,
not a numerical/online qualification certificate. The inherited original join,
finish/cursor transition, cycle clock and publication remain authoritative.
"""
import traceback

from atomic_prefill import Phase
from diagnostic_authorization import require_bounded_scope
from flexible_scope import envelope, integer
from live_mixed_metadata import (bind_live_descriptor, prepare_live_call,
    prepare_target_capacity_at_common_boundary, validate_live_descriptor,
    _plan_signature)
from original_hook_adapter import OriginalHooks, OriginalHookMissing, RuntimeHooks


class LiveRuntimeHooks(RuntimeHooks):
    def __init__(self, engine, hooks, *, allocation_tracker, allowed_partitions, qualified_s35_scope=None):
        if not isinstance(hooks, OriginalHooks):
            raise TypeError('Live hooks require the explicit OriginalHooks callback contract')
        if type(allowed_partitions) is not frozenset:
            raise TypeError('An explicit frozen set of (N, Btuple) partitions is required')
        for entry in allowed_partitions:
            if (type(entry) is not tuple or len(entry) != 2 or type(entry[1]) is not tuple):
                raise ValueError('Each allowed partition must be an exact (N, Btuple) pair')
            envelope(entry[0], entry[1], 1)
        super().__init__(engine, hooks)
        self.allocation_tracker = allocation_tracker
        self.allowed_partitions = allowed_partitions
        self.qualified_s35_scope=qualified_s35_scope
        if qualified_s35_scope is not None:qualified_s35_scope.require(engine)
        self.descriptor = None
        self.prepared_call = None
        self._cycle_inputs = None
        self._cycle_tickets = None
        self._execution_attempted = False

    def _clear_live_references(self):
        self.descriptor = None
        self.prepared_call = None
        self._planned = None
        self._flexible = None
        self._target_samples = None
        self._cycle_inputs = None
        self._cycle_tickets = None
        self._execution_attempted = False
        self._eligible_cycle = None

    def request_original_fallback(self):
        if self.transactions.current is None:
            return False
        self._fallback_requested = True
        return True

    def _host_inputs(self, target_payloads, target_indices, local_states, cycle_id):
        row = self.transactions.current
        integer(cycle_id, 'current live cycle', 0)
        if (row is None or row.phase != Phase.RESERVED or row.chunk is not None
                or row.initial_fixture_cursor is not None or row.initial_fixture_scope is not None
                or row.committed_cursor > 0 and row.last_committed_cycle_id < 0
                or cycle_id <= row.last_committed_cycle_id or self._planned is not None
                or self.descriptor is not None or self._fatal):
            raise OriginalHookMissing('Live begin requires an unused current non-adopted reservation/cycle')
        ids = tuple(target_indices)
        payloads = tuple(target_payloads)
        if (len(ids) != len(payloads) or any(type(i) is not int or i < 0 for i in ids)
                or len(ids) != len(set(ids))):
            raise OriginalHookMissing('Actual target payloads and request IDs are not aligned')
        plans = tuple(payload['plan'] for payload in payloads)
        rows = tuple(tuple(payload['row']) for payload in payloads)
        tickets = tuple(payload['ticket'] for payload in payloads)
        proposals = tuple(ticket.proposal_id for ticket in tickets)
        if (any(type(i) is not int or i < 0 for i in proposals)
                or len(proposals) != len(set(proposals))):
            raise OriginalHookMissing('Actual target tickets need unique integer proposal IDs')
        budgets = tuple(plan.candidate_budget for plan in plans)
        roots = tuple(local_states[index].token_ids[-1] for index in ids)
        temperatures = tuple(local_states[index].temperature for index in ids)
        pending = local_states[row.lease.sequence_id]
        if (tuple(pending.token_ids) != row.prompt or pending.temperature != 0
                or pending.top_p < 1.0 or pending.top_k > 0):
            raise OriginalHookMissing('Live reserved prompt/state or greedy admission identity changed')
        remaining = len(row.prompt)-row.committed_cursor
        supported = (1 <= len(plans) <= 32 and all(type(b) is int and 1 <= b <= 5 for b in budgets)
            and sum(budgets) <= 96 and sum(b+1 for b in budgets) <= 128
            and remaining > 0 and all(t == 0 for t in temperatures)
            and not self.engine.config.enforce_eager and not self._fallback_requested
            and ((len(plans),budgets) in self.allowed_partitions if len(plans)<=16 else
                self.qualified_s35_scope is not None and len(plans) in self.qualified_s35_scope.require(self.engine)))
        shape = envelope(len(plans), budgets, remaining) if supported else None
        from atomic_prefill import prompt_digest
        packet = dict(generation=self.engine._dynamic_tree_generation, cycle_id=cycle_id,
            identity=(row.lease.identity.request_index, row.lease.identity.nonce,
                      row.lease.identity.allocation_epoch), owner=row.lease.sequence_id,
            cursor=row.committed_cursor, remaining=remaining, prompt_length=len(row.prompt),
            prompt_sha256=prompt_digest(row.prompt), target_ids=ids, proposal_ids=proposals,
            budgets=budgets, supported=supported,
            query_partition=None if shape is None else shape.query_partition,
            end=None if shape is None else row.committed_cursor+shape.prefill_queries,
            final=None if shape is None else shape.final)
        return dict(plans=plans, rows=rows, roots=roots, ids=ids, tickets=tickets,
                    proposals=proposals, temperatures=temperatures, shape=shape, packet=packet)

    def begin_target_cycle(self, *, target_payloads, target_indices, local_states, cycle_id):
        require_bounded_scope()
        if self.transactions.current is None:
            return None
        # Every reachable rank reports input errors before entering the logical
        # preflight. No physical page IDs or rank-local allocation objects cross
        # this consensus callback.
        inputs = self.original.stage_all4('live-mixed-cycle-host-inputs', lambda:
            self._host_inputs(target_payloads, target_indices, local_states, cycle_id))
        self.original.cycle_preflight_all4(inputs['packet'])
        # Original proposal-only rounds have no current target to combine.
        # Keep the real reservation/cursor while the original draft path makes
        # its next proposal. A later nonempty unsupported target still falls
        # back; explicit target-only branches request their own reset.
        if not inputs['plans'] and not self._fallback_requested:
            self._eligible_cycle = None
            return None
        if not inputs['packet']['supported'] or self._fallback_requested:
            self._eligible_cycle = None
            self.request_original_fallback()
            return None
        row = self.transactions.current
        self.original.begin_prefix_protection(inputs['plans'], inputs['ids'], row)
        def prepare_at_common_boundary():
            prepare_target_capacity_at_common_boundary(self.engine,
                plans=inputs['plans'], target_ids=inputs['ids'])
            chunk = self.transactions.plan(row.lease.identity, cycle_id=cycle_id,
                end=row.committed_cursor+inputs['shape'].prefill_queries)
            epochs = self.allocation_tracker.activation_epochs
            descriptor = bind_live_descriptor(self.engine, self.transactions, chunk,
                plans=inputs['plans'], roots=inputs['roots'], rows=inputs['rows'],
                target_ids=inputs['ids'], proposal_ids=inputs['proposals'],
                observed_allocation_epochs=epochs)
            prepared = None
            if not self.engine.is_draft:
                prepared, same = prepare_live_call(self.engine, descriptor,
                    transactions=self.transactions, cycle_id=cycle_id,
                    target_ids=inputs['ids'], proposal_ids=inputs['proposals'],
                    observed_allocation_epochs=epochs)
                if same is not descriptor:
                    raise OriginalHookMissing('Prepared call lost its original live descriptor')
            self.descriptor, self.prepared_call = descriptor, prepared
            boundaries = getattr(self.engine, '_v75_atomic_boundaries', None)
            bind_footprint = getattr(boundaries, 'bind_live_write_footprint', None)
            if not callable(bind_footprint):
                raise OriginalHookMissing('Actual postcapacity live write-footprint binding is absent')
            bind_footprint(descriptor)
            self._flexible = descriptor.flexible
            self._planned = (chunk, descriptor.flexible.layout)
            self._cycle_inputs, self._cycle_tickets = inputs, inputs['tickets']
            self._eligible_cycle = cycle_id
            self._execution_attempted = False
            self._target_submit_error = None
            self._target_samples = None
            return descriptor.flexible.layout
        return self.original.stage_all4('live-mixed-layout-and-metadata-preflight', prepare_at_common_boundary)

    def _validate_cached(self):
        if self.descriptor is None or self._planned is None or self._cycle_inputs is None:
            raise OriginalHookMissing('No current live mixed descriptor/prepared layout')
        descriptor = self.descriptor
        proposals = tuple(ticket.proposal_id for ticket in self._cycle_tickets)
        validate_live_descriptor(self.engine, descriptor, transactions=self.transactions,
            cycle_id=self._eligible_cycle, target_ids=self._cycle_inputs['ids'],
            proposal_ids=proposals,
            observed_allocation_epochs=self.allocation_tracker.activation_epochs)
        if (self._planned[0] != descriptor.chunk or self._planned[1] is not descriptor.flexible.layout
                or not self.engine.is_draft and (self.prepared_call is None
                    or self.prepared_call.layout is not descriptor.flexible.layout)):
            raise OriginalHookMissing('Cached target call differs from its exact prepared live layout')
        return descriptor

    def plan_target(self, *, plans, roots, rows, sequence_ids, temperatures, cycle_id,
                    scratch_sequence_ids=None):
        """Only verify/reuse the call built at common begin; never build here.

        This is called again under asymmetric early submission. A reachable
        local mismatch is retained for the inherited common join, while the
        exact cached layout still routes execution through execute_target.
        """
        if self._planned is None:
            return None
        try:
            descriptor = self._validate_cached()
            count = len(descriptor.target_ids)
            if (type(cycle_id) is not int or cycle_id != descriptor.cycle_id
                    or tuple(sequence_ids) != descriptor.target_ids
                    or tuple(roots) != descriptor.flexible.layout.roots[:count]
                    or tuple(tuple(row) for row in rows) != descriptor.flexible.layout.rows[:count]
                    or len(temperatures) != count or any(t != 0 for t in temperatures)
                    or _plan_signature(plans) != descriptor.plan_signature[:count]
                    or scratch_sequence_ids is not None and tuple(scratch_sequence_ids) != tuple(
                        lease.sequence_id for lease in descriptor.flexible.scratch_leases)):
                raise OriginalHookMissing('Repeated target planning changed live cycle/IDs/roots/rows/plans')
        except BaseException:
            self._target_submit_error = traceback.format_exc()
        return self._planned[1]

    def execute_target(self):
        # Every local validation/accounting/authorization failure is reported at
        # the ORIGINAL common join. No WORLD/Gloo operation is permitted here.
        try:
            require_bounded_scope()
            if self._target_submit_error is not None:
                raise OriginalHookMissing(self._target_submit_error)
            descriptor = self._validate_cached()
            if self._execution_attempted:
                raise OriginalHookMissing('Duplicate live mixed execution in the same cycle')
            self._execution_attempted = True
            account = getattr(self.engine, '_specslo_account_real_target_plans', None)
            if not callable(account):
                raise OriginalHookMissing('Original real-target accounting helper is not bound')
            account(descriptor.flexible.layout.plans[:len(descriptor.target_ids)])
            self.transactions.submitted(descriptor.chunk)  # Before any possible KV write.
            if self.engine.is_draft:
                self._target_samples = None
                return None
            result = self.original.mixed_target_forward(self.engine,
                descriptor.flexible.layout, full_query_capacity=128)
            self._target_samples = result['prefill_token_ids'].detach().clone()
            return result['tree']
        except BaseException:
            self._target_submit_error = traceback.format_exc()
            return None

    def commit_after_original_finish_cycle(self, *args, **kwargs):
        try:
            return super().commit_after_original_finish_cycle(*args, **kwargs)
        finally:
            self._clear_live_references()

    def abort_reserved(self, request_index, *, release_online_cache=None):
        row = self.transactions.current
        affected = row is not None and row.lease.identity.request_index == request_index
        try:
            return super().abort_reserved(request_index, release_online_cache=release_online_cache)
        finally:
            if affected:
                self._clear_live_references()

    def fatal_cleanup_all4(self, original_error):
        try:
            return super().fatal_cleanup_all4(original_error)
        finally:
            self._clear_live_references()

    def close_scope_before_original_release(self):
        try:
            return super().close_scope_before_original_release()
        finally:
            self._clear_live_references()
