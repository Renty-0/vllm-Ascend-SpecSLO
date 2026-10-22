"""Host/CPU contracts only: no Native model or NPU is constructed by this suite."""
from dataclasses import replace
import copy
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import independent_history_v75 as h
from atomic_prefill import Identity
from local_qualification_scope import SCOPES, SOURCE_INPUT_SHA256, fixture_for


def stage(_label, callback):
    return callback()


def certificate(scope=SCOPES[0]):
    identity = Identity(scope.target_requests, 'actual-local-test-nonce', 1)
    prompt = h.immutable_source(fixture_for(scope)).prompt_tokens
    records = tuple(h.InitialRankEvidence(rank, 'draft' if rank == 0 else 'target',
        28 if rank == 0 else 64, 1, 1, scope.initial_prompt_cursor,
        h._sha(prompt), SOURCE_INPUT_SHA256,
        ('a'*64,)*(56 if rank == 0 else 128), True, True, tuple(range(18)))
        for rank in range(4))
    evidence = h.InitialPrefixEvidence(scope.name, SOURCE_INPUT_SHA256, h._sha(prompt),
        identity.request_index, identity.nonce, identity.allocation_epoch,
        scope.initial_prompt_cursor, 2301, 1, records)
    return identity, evidence


class IndependentHistoryCPU(unittest.TestCase):
    def test_original_immutable_owners_prompt_slices_and_no_probe_import(self):
        import sys
        before = set(sys.modules)
        for scope in SCOPES:
            fixture = fixture_for(scope)
            original = copy.deepcopy(fixture)
            source = h.immutable_source(fixture)
            self.assertEqual(len(source.tree_tokens), scope.target_requests)
            self.assertEqual(len(set(source.tree_ids)), scope.target_requests)
            self.assertEqual(len(source.prompt_tokens), 2301)
            start = source.initial_cursor
            for end, chunk in zip(source.chunk_ends, fixture['chunks'], strict=True):
                self.assertEqual(source.prompt_tokens[start:end], chunk)
                start = end
            self.assertEqual(fixture, original)
            with self.assertRaises(Exception):
                source.initial_cursor = 2299
        self.assertFalse(any(name.rsplit('.',1)[-1] in ('paired_progressive','paired_full_repair')
            for name in set(sys.modules)-before))

    def test_source_edit_owner_reorder_and_dense_intermediate_source_rejected(self):
        fixture = fixture_for(SCOPES[0])
        for mutate in (
            lambda f: f['prompt']['token_ids'].__setitem__(2200, f['prompt']['token_ids'][2200]+1),
            lambda f: f['trees'].reverse(),
            lambda f: f.__setitem__('initial_prompt_tokens', tuple(f['prompt']['token_ids'][:2299])),
            lambda f: f['provenance'].__setitem__('source_input_sha256', '0'*64),
        ):
            bad = copy.deepcopy(fixture)
            mutate(bad)
            with self.assertRaises(h.HistoryRejected):
                h.immutable_source(bad)

    def test_typed_certificate_requires_actual_four_roles_hashes_layers_and_lease(self):
        for scope in SCOPES:
            identity, evidence = certificate(scope)
            self.assertIs(h.validate_initial_prefix_evidence(evidence, identity), evidence)
            rows = evidence.rank_records
            bad_rows = (
                rows[:3],
                (replace(rows[0], role='target'),)+rows[1:],
                (rows[0], replace(rows[1], layer_count=28))+rows[2:],
                (replace(rows[0], all_prefix_finite=False),)+rows[1:],
                (rows[0], replace(rows[1], local_stream_fenced=False))+rows[2:],
                (rows[0], replace(rows[1], prefix_sha256=('g'*64,)*128))+rows[2:],
                (rows[0], replace(rows[1], pages=(0,)*18))+rows[2:],
            )
            for bad in bad_rows:
                with self.assertRaises(h.HistoryRejected):
                    h.validate_initial_prefix_evidence(replace(evidence, rank_records=bad), identity)
            for bad in (replace(evidence, initial_cursor=2299),
                        replace(evidence, nonce='different'),
                        replace(evidence, adoption_kind='online-tree-commit'),
                        replace(evidence, source_input_sha256='0'*64)):
                with self.assertRaises(h.HistoryRejected):
                    h.validate_initial_prefix_evidence(bad, identity)

    def test_true_fields_are_not_a_permit_or_an_issued_execution_capability(self):
        identity, evidence = certificate()
        with self.assertRaises(Exception):
            h.consume_initial_prefix_evidence(evidence, identity, expected_pages=tuple(range(18)))
        with patch.object(h, 'require_bounded_scope', return_value=None):
            with self.assertRaises(h.HistoryRejected):
                h.consume_initial_prefix_evidence(evidence, identity, expected_pages=tuple(range(18)))
            with self.assertRaises(h.HistoryRejected):
                h.consume_initial_prefix_evidence(dict(vars(evidence)), identity,
                    expected_pages=tuple(range(18)))
        self.assertNotIn(id(evidence), h._ISSUED_INITIAL_PREFIXES)

    def test_issued_capability_single_use_and_live_epoch_pages(self):
        # Directly seeded registry is a HOST contract fixture, not an issuer or
        # a permitted actual actor path. The manager never accepts such a dict.
        identity, evidence = certificate()
        allocation = NS(block_tables=[[-1]*32 for _ in range(16)])
        allocation.block_tables[15][:18] = list(range(18))
        engine = NS(rank=0, cache_allocation=allocation)
        engine._v75_atomic_boundaries = NS(allocation_object=allocation, activation_epochs={15:1})
        epoch = dict(engine_epoch=1, scope=SCOPES[0],
            immutable_source=h.immutable_source(fixture_for(SCOPES[0])))
        entry = dict(evidence=evidence, engine=engine, epoch=epoch, identity=identity, consumed=False)
        with patch.dict(h._ISSUED_INITIAL_PREFIXES, {id(evidence):entry}, clear=True), \
                patch.object(h,'require_bounded_scope'), patch.object(h,'_epoch'), \
                patch.object(h,'prefix_evidence',return_value={15:evidence.rank_records[0].prefix_sha256}):
            with self.assertRaises(h.HistoryRejected):
                h.consume_initial_prefix_evidence(evidence, identity, expected_pages=tuple(range(1,19)))
            self.assertFalse(entry['consumed'])
            self.assertIs(h.consume_initial_prefix_evidence(evidence, identity,
                expected_pages=tuple(range(18))), evidence)
            with self.assertRaises(h.HistoryRejected):
                h.consume_initial_prefix_evidence(evidence, identity, expected_pages=tuple(range(18)))

    def test_independent_normal_missing_ledger_and_observed_append_nonmutation(self):
        source = h.immutable_source(fixture_for(SCOPES[0]))
        n = len(source.tree_tokens)
        refs = list(range(20,20+n))
        ledger = {owner:dict(safe_before=len(t)-1,tokens={})
            for owner,t in zip(refs,source.tree_tokens,strict=True)}
        epoch = dict(immutable_source=source, reference_tree_ids=refs,
            draft_reference_ledger=ledger,candidate_prefixes=tuple(t[:-1] for t in source.tree_tokens)
                +(source.prompt_tokens[:source.initial_cursor],)+((),)*(18-n))
        self.assertTrue(all(len(row[3]) == 1 for row in h._normal_entries(epoch)))
        for owner, tokens in zip(refs,source.tree_tokens,strict=True):
            p = len(tokens)-1
            ledger[owner] = dict(safe_before=p,tokens={p:tokens[-1],p+1:777,p+2:778,p+3:779})
        emitted = [[777,999,123] for _ in range(n)]
        record = dict(original_finish_cycle_executed=True,
            extracted_structural_and_numerical_lanes_unchanged=True,cycle_id=7,
            actual_prefix_epochs={i:1 for i in range(n)},committed_tokens=emitted,
            actual_logical_tokens=[list(t)+out for t,out in zip(source.tree_tokens,emitted,strict=True)])
        saved = copy.deepcopy(record)
        with patch.object(h,'_epoch',return_value=source):
            h.observe_original_commit(NS(),epoch,record,stage_all4=stage)
            with self.assertRaises(h.HistoryRejected):
                h.observe_original_commit(NS(),epoch,record,stage_all4=stage)
        self.assertEqual(record,saved)
        for entry,tokens in zip(h._normal_entries(epoch),source.tree_tokens,strict=True):
            self.assertEqual(entry[2],len(tokens)+1)
            self.assertEqual(entry[3],(999,123))
        self.assertEqual(source,h.immutable_source(fixture_for(SCOPES[0])))

    def test_independent_target_compaction_all64_layers_uses_own_clones_only(self):
        import torch
        n = 15
        block_size = 128
        layers = [NS(self_attn=NS(key_cache=torch.zeros((38,block_size,1)),
                    value_cache=torch.zeros((38,block_size,1)))) for _ in range(64)]
        engine = NS(is_draft=False,device=torch.device('cpu'),model=NS(layers=layers),
            config=NS(kvcache_block_size=128),
            cache_allocation=NS(block_tables=[[i]+[-1]*31 for i in range(38)]))
        prefix = tuple((100,101,102) for _ in range(16))+((),)*3
        source = NS(tree_tokens=tuple((100,101,102,103+i) for i in range(n)))
        epoch = dict(immutable_source=source,reference_span_ids=list(range(19,38)),
            reference_tree_ids=list(range(19,34)),reference_prefixes=prefix)
        plans = [NS(prefix_len=3,candidate_budget=5,
            parent_indices=torch.tensor([-1,0,1,2,0])) for _ in range(n)]
        query = tuple(torch.arange(197,dtype=torch.float32).reshape(197,1)+1000*index
            for index in range(128))
        candidate_before = [a[:19].clone() for layer in layers for a in
            (layer.self_attn.key_cache,layer.self_attn.value_cache)]
        query_before = [v.clone() for v in query]
        with patch.object(h,'_epoch',return_value=source),patch.object(h,'_device_fence'):
            result = h.commit_reference_tree(engine,epoch,plans,[[10,11,12]]*n,
                [[0,1]]*n,query,stage_all4=stage)
        self.assertIs(result['copied_candidate_KV'],False)
        for array_index,cache in enumerate(h._arrays(engine)):
            self.assertTrue(torch.equal(cache[:19*128],candidate_before[array_index].flatten(0,1)))
            for owner in range(19,34):
                expected = query[array_index][6*(owner-19):6*(owner-19)+3]
                self.assertTrue(torch.equal(cache[owner*128+3:owner*128+6],expected))
            self.assertTrue(torch.equal(query[array_index],query_before[array_index]))
        for i in range(n):
            self.assertEqual(epoch['reference_prefixes'][i],(100,101,102,103+i,10,11))
        before = [v.clone() for v in h._arrays(engine)]
        with patch.object(h,'_epoch',return_value=source):
            with self.assertRaises(h.HistoryRejected):
                h.commit_reference_tree(engine,epoch,plans,[[10,11,12]]*n,
                    [[0,2]]*n,query,stage_all4=stage)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,h._arrays(engine))))

    def test_query_clones_and_nonfinite_prefix_are_real_tensor_checks(self):
        import torch
        layers = [NS(self_attn=NS(key_cache=torch.ones((1,128,1)),
            value_cache=torch.ones((1,128,1))*2)) for _ in range(28)]
        engine = NS(is_draft=True,device=torch.device('cpu'),model=NS(layers=layers),
            config=NS(kvcache_block_size=128),cache_allocation=NS(block_tables=[[0]+[-1]*31]))
        meta = NS(slot_mapping=torch.tensor([3,4,5]))
        snapshot = h.query_KV(engine,meta)
        self.assertEqual(len(snapshot),56)
        layers[0].self_attn.key_cache[0,3] = 999
        self.assertEqual(snapshot[0][0].item(),1)
        with self.assertRaises(h.HistoryRejected):
            h.query_KV(engine,NS(slot_mapping=torch.tensor([-1])))
        h.prefix_evidence(engine,{0:(1,2,3)})
        layers[-1].self_attn.value_cache[0,2] = float('nan')
        with self.assertRaises(h.HistoryRejected):
            h.prefix_evidence(engine,{0:(1,2,3)})

    def test_progressive_consumer_strict_sequence_no_dense_callback_or_manager_mutation(self):
        import torch
        for scope in SCOPES:
            source = h.immutable_source(fixture_for(scope))
            n = len(source.tree_tokens)
            refs = list(range(40,59))
            histories = tuple(t[:-1] for t in source.tree_tokens)+(source.prompt_tokens[:source.initial_cursor],)+((),)*(18-n)
            def make_epoch():
                return dict(scope=scope,immutable_source=source,reference_span_ids=refs,
                    reference_prompt_id=refs[n],reference_prefixes=histories,
                    next_prompt_phase=0,independent_prompt_cursor=source.initial_cursor,
                    failed=False,history_events=[])
            engine = NS(is_draft=False,rank=1)
            epoch = make_epoch()
            calls = []
            def evaluator(_engine,current,index):
                phase = scope.phases()[index]
                calls.append((index,current['independent_prompt_cursor']))
                return dict(passed=True,source_prefix_finite=True,
                    all_physical_hidden_original_matcher=True,
                    all64_query_KV_original_matcher=True,exact_target_heads=True,
                    mirror_recomputed=True,copied_candidate_KV=False,phase_index=index,
                    prompt_start=phase['prompt_start'],prompt_end=phase['prompt_end'],
                    reference_span_ids=refs,reference_prompt_id=refs[n],physical_M=197,
                    segments=19,mask_capacity=128,source_input_sha256=SOURCE_INPUT_SHA256,
                    input_prompt_tokens=list(source.prompt_tokens[phase['prompt_start']:phase['prompt_end']]),
                    first_token=77 if phase['final'] else None,
                    reference_query_KV=tuple(torch.zeros((197,1)) for _ in range(128)))
            def gather(_engine,packet):
                return tuple(dict(rank=r,is_draft=r==0,
                    first_token=None if r==0 else packet['first_token']) for r in range(4))
            with patch.object(h,'_epoch',return_value=source), \
                    patch.object(h,'prefix_evidence',return_value={refs[n]:('a'*64,)*128}), \
                    patch.object(h,'_gather_all4',side_effect=gather):
                with self.assertRaises(h.HistoryRejected):
                    h.advance_prompt_reference(engine,epoch,1,recompute_same_m=evaluator,stage_all4=stage)
                self.assertEqual(calls,[])
                for index in range(len(scope.phases())):
                    record = h.advance_prompt_reference(engine,epoch,index,
                        recompute_same_m=evaluator,stage_all4=stage)
                    self.assertEqual(record['phase_index'],index)
                self.assertEqual(calls,list(enumerate((scope.initial_prompt_cursor,)+scope.chunk_ends[:-1])))
                self.assertEqual(epoch['independent_prompt_cursor'],2301)
                self.assertEqual(epoch['reference_prefixes'][n],source.prompt_tokens)
                self.assertTrue(all(not e['original_WORLD_broadcast_claimed'] for e in epoch['history_events']))
                bad_epoch = make_epoch()
                def false_evaluator(*args):
                    value = evaluator(*args)
                    value['copied_candidate_KV'] = True
                    return value
                with self.assertRaises(h.HistoryRejected):
                    h.advance_prompt_reference(engine,bad_epoch,0,recompute_same_m=false_evaluator,stage_all4=stage)
                self.assertEqual(bad_epoch['independent_prompt_cursor'],source.initial_cursor)
                self.assertEqual(bad_epoch['reference_prefixes'],histories)
                self.assertTrue(bad_epoch['failed'])
                self.assertNotIn('manager',epoch)

    def test_original_manager_adoption_and_real_commit_precede_descriptor_advance(self):
        from atomic_prefill import AtomicPrefill, Lease, PreparedFence, TreeCommitFence
        for scope in SCOPES:
            identity, evidence = certificate(scope)
            source = h.immutable_source(fixture_for(scope))
            n = scope.target_requests
            manager = AtomicPrefill()
            manager.reserve(identity,source.prompt_tokens,
                Lease(identity,n,tuple(range(18))),active=tuple(range(n)),pending=(n,),
                capacity=32,live_pages=tuple(range(18,18+n)))
            # This exercises the real HOST manager; the consumer alone is
            # mocked. It does not issue an execution certificate or NPU proof.
            with patch('diagnostic_authorization.require_bounded_scope'), \
                    patch.object(h,'consume_initial_prefix_evidence',return_value=evidence) as consumer:
                row = manager.adopt_local_initial_prefix(identity,evidence=evidence)
            self.assertEqual(row.committed_cursor,scope.initial_prompt_cursor)
            self.assertEqual(row.last_committed_cycle_id,-1)
            consumer.assert_called_once_with(evidence,identity,expected_pages=tuple(range(18)))
            epoch = dict(scope=scope,immutable_source=source,_manager=manager,
                prompt_lease_identity=identity,last_observed_cycle=-1,
                candidate_prefixes=tuple(t[:-1] for t in source.tree_tokens)
                    +(source.prompt_tokens[:source.initial_cursor],)+((),)*(18-n))
            with patch.object(h,'_epoch',return_value=source):
                for index,phase in enumerate(scope.phases()):
                    epoch['last_observed_cycle'] = index
                    chunk = manager.plan(identity,cycle_id=index,end=phase['prompt_end'])
                    manager.submitted(chunk)
                    manager.prepared(chunk,[77] if phase['final'] else [],
                        PreparedFence(True,True,True,True,True,True))
                    with self.assertRaises(h.HistoryRejected):
                        h.observe_atomic_prompt_commit(NS(),epoch,manager.current,stage_all4=stage)
                    self.assertEqual(len(epoch['candidate_prefixes'][n]),phase['prompt_start'])
                    manager.commit_after_tree(identity,TreeCommitFence(index,True,True,True,True))
                    actual = manager.current
                    result = h.observe_atomic_prompt_commit(NS(),epoch,actual,stage_all4=stage)
                    self.assertIs(manager.current,actual)
                    self.assertEqual(len(epoch['candidate_prefixes'][n]),phase['prompt_end'])
                    self.assertEqual(epoch['candidate_prefixes'][n],source.prompt_tokens[:phase['prompt_end']])
                    self.assertIs(result['manager_cursor_assigned'],False)
                    with self.assertRaises(h.HistoryRejected):
                        h.observe_atomic_prompt_commit(NS(),epoch,actual,stage_all4=stage)
                self.assertEqual(manager.current.committed_cursor,2301)

    def test_plural_owner_partition_is_disjoint_bounded_and_nonmutating(self):
        for n,used in ((15,22),(16,21)):
            pool = list(range(463,495));before = pool.copy()
            owners = h.epoch_owner_partition(n,pool)
            self.assertEqual(pool,before)
            self.assertEqual(len(owners['candidate_span_ids']),19)
            self.assertEqual(len(owners['reference_span_ids']),19)
            self.assertEqual(len(set(owners['candidate_span_ids']+owners['reference_span_ids'])),38)
            self.assertEqual(owners['actual_padding_ids'],tuple(pool[:18-n]))
            self.assertEqual(owners['reference_prompt_id'],pool[18])
            self.assertEqual(owners['reference_padding_ids'],tuple(pool[19:used]))
            self.assertEqual(len(set(owners['actual_padding_ids'])|set(owners['reference_span_ids'])),used)
            self.assertFalse(owners['new_NPU_scope_qualified'])
            self.assertNotIn('actual_padding_id',owners)
            self.assertNotIn('reference_padding_id',owners)
            for bad in (pool[:used-1],[0]+pool[1:],pool[:-1]+[pool[0]],pool[:-1]+[-1]):
                with self.assertRaises(h.HistoryRejected):h.epoch_owner_partition(n,bad)
        for n in (0,14,17,True):
            with self.assertRaises(h.HistoryRejected):h.epoch_owner_partition(n,range(463,495))

    def test_original_CPU_allocator_reserves_38_private_spans_and_no_forward(self):
        import torch
        from vllm_ascend.spec_decode.pearl.native_cache import NativePrefixCache
        from vllm_ascend.spec_decode.pearl.native_engine import NativePearlEngine
        for scope in SCOPES:
            fixture=fixture_for(scope);source=h.immutable_source(fixture)
            engine=NativePearlEngine.__new__(NativePearlEngine)
            engine.device=torch.device('cpu');engine.is_draft=True
            engine.config=NS(max_num_queued_seqs=463,max_num_seqs=32,
                enable_spec_rhythm=True,spec_rhythm_stable_graphs=True,enforce_eager=False,
                spec_rhythm_tree_width=2,spec_rhythm_tree_depth=4,
                enable_prefix_caching=False,num_kvcache_blocks=544,kvcache_block_size=128,
                max_model_len=4096)
            engine.prefix_cache=NativePrefixCache(544,32,128)
            engine.cache_allocation=None;engine.cache_block_tables=None
            writes=[]
            def record_write(_engine,owner,tokens,*,stage_all4):
                writes.append((owner,tuple(tokens)))
            # Real Native HOST allocate/activate/ensure/release methods run.
            # Device/model gates and actual forwards are intentionally mocked;
            # this cannot emit a valid initial-prefix certificate.
            with patch.object(h,'_require_engine'),patch.object(h,'_write_initial_prefix',side_effect=record_write), \
                    patch.object(h,'_device_fence'),patch.object(h,'prefix_evidence',return_value={}):
                epoch=h.allocate_independent_epoch(engine,fixture,stage_all4=stage)
            try:
                ids=epoch['candidate_span_ids']+epoch['reference_span_ids']
                self.assertEqual(len(ids),38)
                self.assertEqual(len(set(ids)),38)
                self.assertEqual(writes,list(zip(ids,epoch['source_prefixes']*2,strict=True)))
                self.assertEqual(len(epoch['source_prefixes']),19)
                self.assertEqual(epoch['source_prefixes'][scope.target_requests],source.prompt_tokens[:source.initial_cursor])
                self.assertEqual(epoch['source_prefixes'][scope.target_requests+1:],((),)*(18-scope.target_requests))
                allpages=[]
                for owner in ids:
                    pages=epoch['allocations'][owner];allpages.extend(pages)
                    self.assertEqual(engine.cache_block_tables[owner].tolist(),engine.cache_allocation.block_tables[owner])
                    self.assertEqual(engine.cache_allocation.num_cached_tokens[owner],0)
                self.assertEqual(len(allpages),len(set(allpages)))
                self.assertLessEqual(len(allpages),544)
                for owner in (epoch['prompt_id'],epoch['reference_prompt_id']):
                    self.assertEqual(len(epoch['allocations'][owner]),18)
                    self.assertEqual(engine._dynamic_tree_known_KV[owner]['safe_before'],source.initial_cursor)
                self.assertFalse(epoch['initial_fixture_adoption_is_online_commit'])
                self.assertEqual(len(h._ISSUED_INITIAL_PREFIXES),0)
                self.assertFalse(torch.npu.is_initialized())
            finally:
                engine._release_cache()
                self.assertIsNone(engine.cache_allocation)
                self.assertIsNone(engine.cache_block_tables)


if __name__ == '__main__':
    unittest.main()
