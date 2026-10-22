"""Host contracts only: actual Native metadata builder, never model/graph/NPU.

Activation epochs and fences below are explicit CPU lifecycle test inputs,
not numerical or allocator qualification receipts.
"""
from dataclasses import fields, replace
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest

import atomic_prefill as atomic
import flexible_layout as flexible
from flexible_scope import envelope
import live_mixed_metadata as live
from test_flexible_layout_cpu import fixture as native_fixture
from test_flexible_layout_cpu import MODEL_FORWARD_ATTEMPTS, NATIVE_BUILDER_OBSERVATIONS


def fixture(N=16, budgets=None, prompt_length=102, *, remap=False):
    """Real source token geometry, real AtomicPrefill state, CPU allocator stub."""
    import torch
    budgets = (5,)*N if budgets is None else tuple(budgets)
    engine, old, _, plans, roots, rows = native_fixture(N, budgets, prompt_length)
    full_prompt = json.loads((Path(__file__).parent/'inputs-v45-reference.json').read_text())['prompt']['token_ids']
    prompt = tuple(full_prompt[-prompt_length:])
    tables = engine.cache_allocation.block_tables
    tables.extend([[-1]*32 for _ in range(80-len(tables))])
    used = {page for row in tables for page in row if page >= 0}
    def page():
        result = next(p for p in range(544) if p not in used)
        used.add(result)
        return result
    # Admission reserves the COMPLETE immutable prompt before cursor0 begins.
    current = [p for p in tables[N] if p >= 0]
    while len(current) < (len(prompt)+127)//128:
        current.append(page())
    tables[N] = current+[-1]*(32-len(current))
    # Original Native generation owns32 dedicated scratch rows. All allocated
    # rows, including the ones not selected this cycle, participate in checks.
    pool = tuple(range(N+1, N+33))
    for owner in pool:
        if tables[owner][0] == -1:
            tables[owner][0] = page()
    ids = tuple(range(N))
    owner = N
    if remap:
        mapping = {index:(index+23)%80 for index in range(80)}
        moved = [[-1]*32 for _ in tables]
        for index, row in enumerate(tables):
            moved[mapping[index]] = row
        tables = moved
        engine.cache_allocation.block_tables = tables
        ids, owner = tuple(mapping[v] for v in ids), mapping[owner]
        pool = tuple(mapping[v] for v in pool)
    epochs = {index:100+index for index, row in enumerate(tables) if row[0] >= 0}
    engine._dynamic_tree_generation = 7
    engine.config.enforce_eager = False
    engine._tree_cache_capacity = 544*128
    engine._tree_graph_scratch_sequence_ids = pool
    engine.cache_block_tables = torch.tensor(tables, dtype=torch.int32)
    engine.capacity_calls = []
    def ensure_capacity(packed, positions):
        engine.capacity_calls.append((tuple(packed), tuple(positions)))
        for index, position in zip(packed, positions, strict=True):
            logical = position//128
            for block in range(logical+1):
                if tables[index][block] == -1:
                    tables[index][block] = page()
        engine.cache_block_tables = torch.tensor(tables, dtype=torch.int32)
    engine._ensure_cache_capacity = ensure_capacity
    identity = atomic.Identity(owner, 'cpu-live-reservation', epochs[owner])
    lease = atomic.Lease(identity, owner, tuple(p for p in tables[owner] if p >= 0))
    manager = atomic.AtomicPrefill()
    manager.reserve(identity, prompt, lease, active=ids, pending=(owner,), capacity=32,
        live_pages={p for index, row in enumerate(tables) if index != owner for p in row if p >= 0})
    return NS(engine=engine, transactions=manager, identity=identity, plans=plans,
        roots=roots, rows=rows, ids=ids, proposals=tuple(reversed(ids)), epochs=epochs)


def bind(case, cycle_id=0):
    row = case.transactions.current
    shape = envelope(len(case.plans), tuple(p.candidate_budget for p in case.plans),
                     len(row.prompt)-row.committed_cursor)
    chunk = case.transactions.plan(case.identity, cycle_id=cycle_id,
                                   end=row.committed_cursor+shape.prefill_queries)
    return live.bind_live_descriptor(case.engine, case.transactions, chunk,
        plans=case.plans, roots=case.roots, rows=case.rows, target_ids=case.ids,
        proposal_ids=case.proposals, observed_allocation_epochs=case.epochs)


def prepare(case, descriptor, **overrides):
    args = dict(transactions=case.transactions, cycle_id=descriptor.cycle_id,
        target_ids=case.ids, proposal_ids=case.proposals, observed_allocation_epochs=case.epochs)
    args.update(overrides)
    return live.prepare_live_call(case.engine, descriptor, **args)


def commit_partial(case, descriptor):
    case.transactions.submitted(descriptor.chunk)
    case.transactions.prepared(descriptor.chunk, {}, atomic.PreparedFence(*([True]*6)))
    case.transactions.commit_after_tree(case.identity,
        atomic.TreeCommitFence(descriptor.cycle_id, *([True]*4)))


class LiveMixedMetadataCPU(unittest.TestCase):
    def assert_native_equivalent(self, case, descriptor):
        import torch
        before = case.transactions.current
        calls = len(case.engine.capacity_calls)
        prepared, same = prepare(case, descriptor)
        self.assertIs(same, descriptor)
        self.assertIs(case.transactions.current, before)
        self.assertEqual(len(case.engine.capacity_calls), calls)
        # This oracle invokes the ACTUAL retained Native builder, with its
        # source path asserted by the existing CPU fixture; no model forward.
        reference = flexible.prepare_metadata(case.engine, descriptor.flexible,
            observed_allocation_epochs=case.epochs)
        self.assertTrue(torch.equal(prepared.input_ids, reference.input_ids))
        self.assertTrue(torch.equal(prepared.positions, reference.positions))
        for item in fields(reference.metadata):
            expected, observed = getattr(reference.metadata, item.name), getattr(prepared.metadata, item.name)
            if isinstance(expected, torch.Tensor):
                self.assertEqual((observed.shape, observed.dtype, observed.device),
                                 (expected.shape, expected.dtype, expected.device), item.name)
                self.assertTrue(torch.equal(observed, expected), item.name)
            else:
                self.assertEqual(observed, expected, item.name)
        self.assertEqual(case.engine.model.builder_flags, [False])
        self.assertEqual(len(prepared.allocation_epoch_evidence), 19)
        self.assertFalse(prepared.qualified)
        self.assertFalse(descriptor.qualified)
        self.assertFalse(descriptor.numerically_qualified)
        self.assertFalse(descriptor.online_qualified)
        self.assertFalse(prepared.metadata.use_fused_infer_attention)
        self.assertFalse(torch.npu.is_initialized())
        self.assertEqual(MODEL_FORWARD_ATTEMPTS, [])
        return prepared

    def test_host_builder_matches_actual_native_from_cursor0_and_changed_owner_ids(self):
        for N, budgets, length, remap in [
                (16, (5,)*16, 2301, False), (15, (1,)*15, 2301, True),
                (1, (1,), 2, True), (16, (4,)*16, 113, False)]:
            with self.subTest(N=N, remap=remap):
                case = fixture(N, budgets, length, remap=remap)
                before = case.transactions.current
                descriptor = bind(case)
                self.assertIs(case.transactions.current, before)
                self.assertEqual(descriptor.chunk.start, 0)
                self.assertEqual(descriptor.target_ids, case.ids)
                self.assertEqual(descriptor.proposal_ids, case.proposals)
                self.assertEqual(case.engine.capacity_calls, [])
                prepared = self.assert_native_equivalent(case, descriptor)
                q = descriptor.flexible.envelope.verification_queries
                end = descriptor.chunk.end
                self.assertEqual(prepared.input_ids[q:q+end].tolist(), list(before.prompt[:end]))
                self.assertEqual(prepared.positions[q:q+end].tolist(), list(range(end)))
                self.assertEqual(descriptor.flexible.layout.head_indices,
                    tuple(range(q))+((q+end-1,) if descriptor.chunk.final else ()))
        self.assertTrue(NATIVE_BUILDER_OBSERVATIONS)

    def test_real_partial_commit_then_genuine_root_only_final_at_cursor101(self):
        case = fixture(15, (5,)*15, 102)
        first = bind(case, 3)
        self.assertEqual((first.chunk.start, first.chunk.end, first.chunk.final), (0, 101, False))
        prepare(case, first)
        commit_partial(case, first)
        with self.assertRaisesRegex(ValueError, 'cursor|phase|cycle'):
            prepare(case, first)
        second = bind(case, 4)
        self.assertEqual((second.chunk.start, second.chunk.end, second.chunk.final), (101, 102, True))
        plan = second.flexible.layout.plans[15]
        self.assertIsInstance(plan, flexible.RootOnlyPromptPlan)
        self.assertEqual((plan.parent_indices.numel(), plan.positions.tolist(), plan.cache_positions.tolist()),
                         (0, [101], [101]))
        case.transactions.submitted(second.chunk)
        prepared = self.assert_native_equivalent(case, second)
        q = second.flexible.envelope.verification_queries
        self.assertEqual(prepared.input_ids[q].item(), case.transactions.current.prompt[101])
        self.assertEqual(prepared.metadata.context_lens[q].item(), 102)
        self.assertEqual(case.transactions.current.committed_cursor, 101)
        self.assertEqual(case.transactions.current.phase, atomic.Phase.SUBMITTED)

    def test_capacity_is_separate_target_only_and_uses_original_call_shape(self):
        case = fixture(16, (5,)*16, 102)
        tables = case.engine.cache_allocation.block_tables
        # Keep the actual activated root page; missing tail pages must fail
        # before preparation, and only the explicitly called capacity step may
        # restore normal-tree pages. Real plans have long existing prefixes.
        owner, plan = case.ids[0], case.plans[0]
        logical = max(plan.cache_positions.tolist())//128
        self.assertGreater(logical, 0)
        tables[owner][logical:] = [-1]*(32-logical)
        with self.assertRaisesRegex(RuntimeError, 'unallocated'):
            bind(case)
        self.assertEqual(case.engine.capacity_calls, [])
        pairs = live.prepare_target_capacity_at_common_boundary(case.engine,
            plans=case.plans, target_ids=case.ids)
        expected = tuple((owner, position) for owner, p in zip(case.ids, case.plans, strict=True)
                         for position in p.cache_positions.tolist())
        self.assertEqual(pairs, expected)
        descriptor = bind(case)
        prepare(case, descriptor)
        self.assertEqual(len(case.engine.capacity_calls), 1)
        case.engine.is_draft = True
        self.assertEqual(live.prepare_target_capacity_at_common_boundary(case.engine,
            plans=case.plans, target_ids=case.ids), ())
        self.assertEqual(len(case.engine.capacity_calls), 1)
        with self.assertRaisesRegex(ValueError, 'target roles'):
            prepare(case, descriptor)

    def test_stale_generation_cycle_manager_allocation_and_id_sets_are_rejected(self):
        for change in ('generation', 'cycle', 'manager', 'allocation', 'target', 'proposal'):
            with self.subTest(change=change):
                case = fixture()
                descriptor = bind(case)
                args = {}
                if change == 'generation': case.engine._dynamic_tree_generation += 1
                if change == 'cycle': args['cycle_id'] = True
                if change == 'manager': args['transactions'] = atomic.AtomicPrefill()
                if change == 'allocation':
                    current = case.engine.cache_allocation
                    case.engine.cache_allocation = NS(block_tables=current.block_tables,
                                                      num_cached_tokens=current.num_cached_tokens)
                if change == 'target': args['target_ids'] = tuple(reversed(case.ids))
                if change == 'proposal': args['proposal_ids'] = case.ids
                with self.assertRaisesRegex(ValueError, 'Stale live'):
                    prepare(case, descriptor, **args)
                self.assertEqual(case.engine.model.builder_flags, [])

    def test_draft_descriptor_does_not_claim_target_normal_query_materialization(self):
        case = fixture()
        owner, plan = case.ids[0], case.plans[0]
        first_query_page = min(plan.cache_positions.tolist())//128
        case.engine.cache_allocation.block_tables[owner][first_query_page:] = [-1]*(32-first_query_page)
        self.assertGreater(first_query_page, 0)
        with self.assertRaisesRegex(RuntimeError, 'unallocated'):
            bind(case)
        case.engine.is_draft = True
        descriptor = bind(case)
        self.assertEqual(len(descriptor.owner_leases), 19)
        self.assertEqual(case.engine.capacity_calls, [])
        self.assertEqual(case.engine.model.builder_flags, [])
        with self.assertRaisesRegex(ValueError, 'target roles'):
            prepare(case, descriptor)

    def test_real_lease_epoch_page_and_nonce_replacement_reject_old_descriptor(self):
        for change in ('epoch', 'bool-epoch', 'missing-epoch', 'page', 'nonce', 'prompt-lease', 'digest'):
            with self.subTest(change=change):
                case = fixture()
                descriptor = bind(case)
                owner = descriptor.owner_leases[-1].sequence_id
                if change == 'epoch': case.epochs[owner] += 1
                if change == 'bool-epoch': case.epochs[owner] = True
                if change == 'missing-epoch': del case.epochs[owner]
                if change == 'page':
                    table = case.engine.cache_allocation.block_tables
                    used = {p for row in table for p in row if p >= 0}
                    table[owner][0] = next(p for p in range(544) if p not in used)
                row = case.transactions.current
                if change == 'nonce':
                    identity = replace(row.lease.identity, nonce='different-reservation')
                    case.transactions._current = replace(row, lease=replace(row.lease, identity=identity))
                if change == 'prompt-lease':
                    case.transactions._current = replace(row, lease=replace(row.lease, pages=(543,)))
                if change == 'digest':
                    case.transactions._current = replace(row, prompt=(row.prompt[0]+1,)+row.prompt[1:])
                with self.assertRaises(ValueError): prepare(case, descriptor)
                self.assertEqual(case.engine.model.builder_flags, [])

    def test_alias_against_unselected_pending_or_scratch_row_is_rejected(self):
        for kind in ('pending', 'unselected-scratch', 'same-owner'):
            with self.subTest(kind=kind):
                case = fixture()
                descriptor = bind(case)
                tables = case.engine.cache_allocation.block_tables
                selected = descriptor.owner_leases[-1].sequence_id
                other = {'pending':79, 'unselected-scratch':case.engine._tree_graph_scratch_sequence_ids[-1],
                         'same-owner':selected}[kind]
                tables[other][1 if other == selected else 0] = tables[selected][0]
                with self.assertRaisesRegex(ValueError, 'alias'):
                    prepare(case, descriptor)
                self.assertEqual(case.engine.model.builder_flags, [])

    def test_bind_rejects_diagnostic_cursor_missing_prompt_lease_and_wrong_chunk(self):
        for kind in ('fixture', 'uncommitted-cursor', 'prompt-pages', 'epoch', 'chunk-end', 'scratch-owner'):
            with self.subTest(kind=kind):
                case = fixture()
                row = case.transactions.current
                if kind == 'fixture':
                    case.transactions._current = replace(row, initial_fixture_cursor=1,
                                                         initial_fixture_scope='diagnostic')
                if kind == 'uncommitted-cursor':
                    case.transactions._current = replace(row, committed_cursor=1)
                if kind == 'prompt-pages':
                    case.transactions._current = replace(row, lease=replace(row.lease, pages=(543,)))
                if kind == 'epoch': case.epochs[case.identity.request_index] += 1
                if kind == 'scratch-owner': case.engine._tree_graph_scratch_sequence_ids = case.ids
                if kind == 'chunk-end':
                    chunk = case.transactions.plan(case.identity, cycle_id=0, end=1)
                    with self.assertRaisesRegex(ValueError, 'partition'):
                        live.bind_live_descriptor(case.engine, case.transactions, chunk,
                            plans=case.plans, roots=case.roots, rows=case.rows, target_ids=case.ids,
                            proposal_ids=case.proposals, observed_allocation_epochs=case.epochs)
                else:
                    with self.assertRaises(ValueError): bind(case)

    def test_device_plans_rejected_before_readback_and_cpu_plan_mutation_detected(self):
        import torch
        case = fixture()
        original = case.plans[0]
        bad = NS(**{name:getattr(original, name) for name in (
            'width','depth','candidate_budget','prefix_len','max_model_len',
            'parent_indices','positions','cache_positions','attention_mask')})
        bad.positions = torch.empty_like(original.positions, device='meta')
        case.plans = [bad, *case.plans[1:]]
        with self.assertRaisesRegex(ValueError, 'CPU tensor'): bind(case)
        self.assertEqual(case.engine.model.builder_flags, [])
        case = fixture()
        # Scratch and prompt plans can be cached shared objects. Change a clone
        # on this normal plan, then restore the original field in finally.
        plan = case.plans[0]
        old = plan.attention_mask
        try:
            object.__setattr__(plan, 'attention_mask', old.clone())
            descriptor = bind(case)
            plan.attention_mask[0, -1].logical_not_()
            with self.assertRaisesRegex(ValueError, 'CPU plan tensors'):
                prepare(case, descriptor)
        finally:
            object.__setattr__(plan, 'attention_mask', old)
        self.assertFalse(torch.npu.is_initialized())


if __name__ == '__main__':
    unittest.main()
