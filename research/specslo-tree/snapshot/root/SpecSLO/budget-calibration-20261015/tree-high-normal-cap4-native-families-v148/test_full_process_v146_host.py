"""New full-process boundaries only; no Torch, model or device operations."""
import ast
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace,ModuleType
import unittest
from budget_policy import TreeBudgetPolicy
from split_shape_v132 import shape
from normal_admission_v140 import reset
from scope_v146 import cases,VECTORS,target_key,validate_full_configuration
from full_process_v146 import (capture_inventory,finish_inventory,validate_inventory_delta,
    capture_fixture_state,finish_fixture_state,track_actual_target_key)
from variable_actual_gate_v146 import validate_documents,require_actual_variable_component

P=Path(__file__).resolve().parent
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl/spec_rhythm.py')


def actual_native_types():
    tree=ast.parse(NATIVE.read_text());imports=[n for n in tree.body if isinstance(n,ast.ImportFrom)
        and n.module=='vllm_ascend.spec_decode.pearl.roofline']
    assert len(imports)==1
    tree.body.remove(imports[0])
    module=ModuleType('_V146_actual_Native_HOST');sys.modules[module.__name__]=module
    module.ProfiledRoofline=type('UnusedProfiledRoofline',(),{})
    exec(compile(tree,str(NATIVE)+':V146-HOST', 'exec'),module.__dict__)
    return module


def entry(validated=True):return SimpleNamespace(runtime_validated=validated,validated_real_row_count=32)


def fake_engine(*,draft=False,after_structures=True):
    # Use original key producers and actually pass the unchanged original validator.
    from startup_receipts_v75 import (_explicit_draft_keys,original_draft_family_keys,
        expected_prefill_keys,_prefill_queries,validate_native_inventory,NATIVE_COUNTERS)
    from finite_inventory_plan import targets
    from shape_scope_v101 import target_key as structure_key
    queries=_prefill_queries()
    if draft:
        keys=_explicit_draft_keys()
        keys['target_entries']|=original_draft_family_keys()
    else:
        keys=dict(entries=set(),draft_entries=set(),target_entries=expected_prefill_keys(queries)
            |{ast.literal_eval(row['expected_key']) for row in targets()})
    runner=SimpleNamespace(**{family:{key:entry() for key in values} for family,values in keys.items()},
        disabled_entry_keys=set(),graph_cache_sealed=False)
    engine=SimpleNamespace(is_draft=draft,cache_allocation=None,cache_block_tables=None,
        _mixed_target_graph_enabled=False,_mixed_target_scratch_sequence_ids=(),
        _tree_graph_scratch_sequence_ids=tuple(range(463,527)),_cache_storage_sequence_capacity=527,
        _dynamic_exact_prefill_qualified={(n,1) for n in queries},
        graph_metrics=lambda:dict.fromkeys(NATIVE_COUNTERS,0),
        prefix_cache=SimpleNamespace(num_blocks=544),
        config=SimpleNamespace(max_num_seqs=64,max_num_queued_seqs=463,kvcache_block_size=128),
        graph_runner=runner)
    validate_native_inventory(engine,queries,original_draft_prepared=True)
    if after_structures and not draft:
        runner.target_entries.update({structure_key(n):entry() for n in (19,35)})
    return engine


def actual_live_plan(native,*,observed=False,eager=False):
    policy=TreeBudgetPolicy(json.loads((P/'tree-gain-model.json').read_text()));reset(policy,1)
    classes=['tight']*19+['normal']*6+['loose']*7
    states={i:native.SpecRhythmRuntimeState(request_index=i,home_batch_id=i%2,slo_tpot_ms=40.,
        slo_class=cls,max_gamma=5,output_token_budget=2 if observed and cls=='loose' else 256)
        for i,cls in enumerate(classes+(['normal']*8 if eager else []))}
    for i,state in states.items():
        state.delivered_tokens=1
        state.decode_elapsed_ms=40.*({'tight':3.5,'normal':2.2,'loose':.5}[state.slo_class] if observed else (1. if i>=32 else 0.))
    if observed:policy.observed[('normal',3)]=(12,23)
    shaper=native.SpecRhythmBudgetShaper(min_gamma=1,max_gamma=8,verification_budget=96)
    plan=shape(policy,shaper,plan_id=7,normal_request_indices=tuple(range(32)),
        eager_request_indices=tuple(range(32,40)) if eager else (),states=states,
        projected_wait_ms=0.,context_len=128,draft_token_budget=192,batch_size=64,
        eager_token_cap=3,eager_reserve_tokens=32,verification_roof=96)
    return plan,policy


class FullProcessHOST(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.native=actual_native_types()

    def test_actual_live_global192_normal96_binding_and_two_vectors(self):
        for observed,vector in zip((False,True),VECTORS,strict=True):
            plan,_=actual_live_plan(self.native,observed=observed)
            self.assertEqual(plan.available_selected_candidates,192)
            self.assertEqual(plan.work_limit.normal_candidates,96)
            self.assertEqual((plan.target_limit.candidates,plan.target_limit.queries,plan.target_limit.rows),(96,128,32))
            self.assertEqual(tuple(plan.normal_budgets.values()),vector)
            self.assertEqual(len(plan.normal_budgets),32)
            self.assertLessEqual(sum(plan.normal_budgets.values()),96)

    def test_actual_live_shared4_defers_all_original_eager_grants(self):
        plan,policy=actual_live_plan(self.native,eager=True)
        self.assertEqual(sum(plan.normal_budgets.values()),96)
        self.assertEqual(plan.eager_budgets,{})
        self.assertGreater(policy.counts['eager_deferred_shared_steps4_rows'],0)
        self.assertEqual(plan.reserve_used,0)
        self.assertEqual(plan.work_limit.global_selected_candidates,192)

    def test_full_configuration_rejects_local96_or_wrong_scratch(self):
        engine=fake_engine();self.assertEqual(validate_full_configuration(engine)['storage'],527)
        engine._cache_storage_sequence_capacity=96
        with self.assertRaises(RuntimeError):validate_full_configuration(engine)
        engine._cache_storage_sequence_capacity=527;engine._tree_graph_scratch_sequence_ids=tuple(range(64,128))
        with self.assertRaises(RuntimeError):validate_full_configuration(engine)

    def test_old247_to249_only_two_new_hidden_and_native_upgrade(self):
        engine=fake_engine();before=capture_inventory(engine)
        old=next(iter(engine.graph_runner.target_entries.values()));old.runtime_validated=False
        # Capture an actually unvalidated inherited entry; only Native-style upgrade is accepted.
        before=capture_inventory(engine);old.runtime_validated=True
        engine.graph_runner.target_entries.update({target_key(c,hidden=True):entry() for c in cases()})
        proof=finish_inventory(engine,before)
        self.assertEqual(sum(map(len,proof['before'].values())),247)
        self.assertEqual(sum(map(len,proof['after'].values())),249)
        canonical=json.loads(json.dumps(proof))
        validate_inventory_delta(canonical['before'],canonical['after'],is_draft=False)
        self.assertTrue(proof['inherited_entry_identity_preserved'])

    def test_inventory_rejects_preexisting_hidden_extra_key_and_replaced_entry(self):
        engine=fake_engine();engine.graph_runner.target_entries[target_key(cases()[0],hidden=True)]=entry()
        with self.assertRaises(RuntimeError):capture_inventory(engine)
        engine=fake_engine();before=capture_inventory(engine)
        engine.graph_runner.target_entries.update({target_key(c,hidden=True):entry() for c in cases()})
        key=next(iter(engine.graph_runner.target_entries));engine.graph_runner.target_entries[key]=entry()
        with self.assertRaises(RuntimeError):finish_inventory(engine,before)
        engine=fake_engine(draft=True);before=capture_inventory(engine)
        engine.graph_runner.target_entries[('new-root',128)]=entry()
        with self.assertRaises(RuntimeError):finish_inventory(engine,before)

    def test_fixture_ledger_and_defer_flags_do_not_cross_fresh_boundary(self):
        engine=fake_engine();engine._defer_tree_materialization=False;engine._dynamic_tree_generation=23
        state=capture_fixture_state(engine);engine._defer_tree_materialization=True
        engine._defer_normal_tree_materialization=True;engine.graph_runner._dynamic_tree_top2=True
        engine._dynamic_tree_known_KV={0:dict(safe_before=128,tokens={128:1})}
        original_entries=engine.graph_runner.entries
        proof=finish_fixture_state(engine,state)
        self.assertEqual(engine._dynamic_tree_known_KV,{})
        self.assertFalse(engine._defer_tree_materialization)
        self.assertFalse(hasattr(engine,'_defer_normal_tree_materialization'))
        self.assertFalse(hasattr(engine.graph_runner,'_dynamic_tree_top2'))
        self.assertIs(engine.graph_runner.entries,original_entries)
        self.assertEqual(proof['source_generation_unchanged'],23)
        engine.cache_allocation=object()
        with self.assertRaises(RuntimeError):finish_fixture_state(engine,state)

    def test_instance_provenance_restores_original_method_even_on_error(self):
        class Runner:
            def _record_execution(self,*args,**kwargs):return args,kwargs
        engine=SimpleNamespace(graph_runner=Runner());runner=engine.graph_runner
        with self.assertRaisesRegex(ValueError,'test failure'):
            with track_actual_target_key(engine):
                self.assertEqual(runner._record_execution('target',1,2,entry_key='actual-key')[0],('target',1,2))
                self.assertEqual(runner._v146_last_target_key,'actual-key')
                raise ValueError('test failure')
        self.assertNotIn('_record_execution',runner.__dict__)
        self.assertNotIn('_v146_last_target_key',runner.__dict__)

    def test_original_native_producer_and_retained_V143_mapping_labels_are_distinct(self):
        import gzip,importlib.util
        from startup_receipts_v75 import validate_native_inventory,_prefill_queries
        from shape_scope_v101 import target_key as structure_key
        old=P.parent/'tree-high-normal-cap4-full-v146/full_process_v146.py'
        spec=importlib.util.spec_from_file_location('_V146_frozen_wrong_namespace_HOST',old)
        wrong=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrong)
        for rank in range(4):
            engine=fake_engine(draft=rank==0,after_structures=False)
            native=validate_native_inventory(engine,_prefill_queries(),original_draft_prepared=True)
            directory=P.parent/'tree-standalone-prefill-control-v143/local-output/standalone-prefill-v143-one-full463-v1'
            report=json.loads((directory/f'rank{rank}-local-diagnostic.json').read_text())
            logical=report['inventory_counts'];actual=report['startup']['actual_native_entry_counts']
            self.assertEqual(logical,dict(draft=86,generic=28,target=0) if rank==0 else dict(draft=0,generic=201,target=44))
            self.assertEqual(actual,dict(entries=0,draft_entries=60,target_entries=54) if rank==0 else dict(entries=0,draft_entries=0,target_entries=245))
            self.assertEqual(native['counts'],actual)
            self.assertEqual({name:sorted(map(repr,values)) for name,values in native['keys'].items()},report['startup']['actual_native_keys'])
            chain=json.loads(gzip.decompress(Path(report['mixed_same_M_qualification']['artifact']['path']).read_bytes()))
            self.assertEqual(chain['actual_inventory']['before'],report['startup']['actual_native_keys'])
            if rank:engine.graph_runner.target_entries.update({structure_key(n):entry() for n in (19,35)})
            self.assertEqual({name:sorted(map(repr,getattr(engine.graph_runner,name))) for name in ('entries','draft_entries','target_entries')},chain['actual_inventory']['after'])
            with self.assertRaises(RuntimeError):wrong.capture_inventory(engine)
            self.assertEqual(capture_inventory(engine)['record'],chain['actual_inventory']['after'])
        engine=fake_engine(after_structures=False)
        keys=list(engine.graph_runner.target_entries)[:201]
        for key in keys:engine.graph_runner.entries[key]=engine.graph_runner.target_entries.pop(key)
        with self.assertRaises(RuntimeError):validate_native_inventory(engine,_prefill_queries(),original_draft_prepared=True)

    def test_real_prior_local_actual_gate_and_pending_or_tampered_rejected(self):
        cfg=json.loads((P/'CANDIDATE_V132.json').read_text())
        result=require_actual_variable_component(cfg)
        self.assertFalse(result['grants_current_full527_qualification'])
        cfg['component_actual_evidence']['variable_B4_local']=None
        with self.assertRaises(RuntimeError):require_actual_variable_component(cfg)
        cfg=json.loads((P/'CANDIDATE_V132.json').read_text());proof=cfg['component_actual_evidence']['variable_B4_local']
        receipt=json.loads(Path(proof['receipt']).read_text());audit=json.loads(Path(proof['external_audit']).read_text())
        audit['natural_protocol_repair_coverage']['1']=False
        with self.assertRaises(RuntimeError):validate_documents(receipt,audit,proof)

    def test_retained_actual_producer_schema_rebased_as_synthetic_full_scope(self):
        """Schema regression only: transformed local raw is never full527 evidence."""
        import gzip,tempfile,shutil
        from audit_v146 import audit_reports
        actual=P.parent/'tree-variable-B4-local-qualification-v144/actual/variable-B4-local-v144-1'
        with tempfile.TemporaryDirectory(prefix='V146-HOST-schema-') as directory:
            root=Path(directory)
            for path in actual.glob('rank*-pass*-case*-*.json.gz'):shutil.copyfile(path,root/path.name)
            for rank in range(4):
                report=json.loads(gzip.decompress((actual/f'rank{rank}-component.json.gz').read_bytes()))
                report['sealed']=False
                report['construction'].update(maxseq=64,maxqueued=463,storage=527)
                report['actual_config'].update(max_num_seqs=64,max_num_queued_seqs=463)
                engine=fake_engine(draft=rank==0);before=capture_inventory(engine)
                if rank:engine.graph_runner.target_entries.update({target_key(c,hidden=True):entry() for c in cases()})
                report['inventory_delta']=finish_inventory(engine,before)
                report['gain_reset']={v:dict(calibrated_prior_unchanged=True,observed_after=[]) for v in ('before','after')}
                report['production_fresh_boundary']=dict(original_cache_release_completed=True,known_KV_ledger_empty=True,
                    defer_flags_restored=True,runner_top2_restored=True,Native_entries_flags_and_graph_owned_buffers_cleared=False)
                for trace in report['traces']:trace['lease']['lease']['storage_rows']=527
                events=[e for e in report['boundary_events'] if 'unified-seal' not in e['label']]
                for sequence,event in enumerate(events):
                    event['sequence']=sequence
                    for row in event['ranks']:row['sequence']=sequence
                report['boundary_events']=events
                report['retained_boundary_prefix']=dict(stage_count=len(events),last_label=events[-1]['label'],
                    saving_stage_and_later_audit_not_retained=True)
                (root/f'rank{rank}-component.json.gz').write_bytes(gzip.compress(json.dumps(report).encode()))
            proof=audit_reports(root)
            self.assertTrue(proof['component_numerical_qualified'])
            self.assertEqual(proof,json.loads(json.dumps(proof)))
            path=root/'rank1-component.json.gz';report=json.loads(gzip.decompress(path.read_bytes()))
            report['retained_boundary_prefix']['stage_count']+=1
            path.write_bytes(gzip.compress(json.dumps(report).encode()))
            with self.assertRaises(RuntimeError):audit_reports(root)


if __name__=='__main__':unittest.main()
