"""Small new installed-branch and exact-source regressions, no device imports."""
import ast
from collections import Counter
from dataclasses import dataclass
from functools import partial
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types
import unittest
from unittest.mock import patch

P=Path(__file__).resolve().parent
NATIVE=Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/source/vllm-ascend-hust/vllm_ascend/spec_decode/pearl')
from split_shape_v132 import shape,filter_eager_parents
from split_budget_v123 import TargetAdmissionLimit

class State:
    max_gamma=5;slo_class='tight';decode_elapsed_ms=100.;decode_start_elapsed_ms=0.;delivered_tokens=1
    slo_tpot_ms=40.;expected_acceptance_benefit=.9;acceptance_ema=.9
    def projected_progress_gap(self,wait):return 2.5
    def urgency(self,wait):return 3.
class Policy:
    candidate_roof=96;target_queries=128;max_target_requests=32
    def __init__(self):self.counts=Counter()
    def gain(self,state,budget):return float(budget)
    def allocate(self,*args):raise AssertionError("Original allocator must not execute under typed candidate")

class CandidateHOST(unittest.TestCase):
    def allocate(self,N,E=0,**changes):
        policy=Policy();shaper=types.SimpleNamespace(min_gamma=1,max_gamma=8)
        kwargs=dict(plan_id=1,normal_request_indices=tuple(range(N)),eager_request_indices=tuple(range(N,N+E)),
            states={i:State() for i in range(N+E)},projected_wait_ms=0.,context_len=128,batch_size=N+E,
            draft_token_budget=192,eager_token_cap=3,eager_reserve_tokens=32,verification_roof=96)
        kwargs.update(changes);return shape(policy,shaper,**kwargs),policy

    def test_actual_large_normal_cap3_with_exact_future3_and_192_conservation(self):
        plan,policy=self.allocate(32,32)
        self.assertEqual(set(plan.normal_budgets.values()),{3});self.assertEqual(set(plan.eager_budgets.values()),{3})
        self.assertEqual(plan.allocated_selected_candidates,192);self.assertEqual(plan.unused_draft_candidates,0)
        self.assertEqual(plan.target_limit.check_actual_call(tuple(plan.normal_budgets.values())),128)
        plans=[types.SimpleNamespace(candidate_budget=b) for b in (*plan.normal_budgets.values(),*plan.eager_budgets.values())]
        self.assertTrue(plan.require_work(tuple(range(64)),[3]*64,[False]*32+[True]*32,plans))
        plans[-1].candidate_budget=2
        with self.assertRaises(RuntimeError):plan.require_work(tuple(range(64)),[3]*64,[False]*32+[True]*32,plans)

    def test_small_normal_cap5_preserves_genuine_branch_budget(self):
        plan,_=self.allocate(8,16);self.assertEqual(set(plan.normal_budgets.values()),{5})
        self.assertEqual(sum(plan.normal_budgets.values()),40);self.assertEqual(sum(plan.eager_budgets.values()),0)

    def test_normal_threshold_is_actual_draft_normal_count(self):
        a,policy=self.allocate(8,32);b,large=self.allocate(9,0)
        self.assertEqual(set(a.normal_budgets.values()),{5});self.assertEqual(set(b.normal_budgets.values()),{3})
        self.assertEqual(policy.counts['draft_normal_N8_shape_calls'],1);self.assertEqual(large.counts['draft_normal_N9_shape_calls'],1)

    def test_explicit_caller_zero_never_becomes_default192(self):
        p,_=self.allocate(32,32,draft_token_budget=0)
        self.assertEqual(p.allocated_selected_candidates,0);self.assertEqual(p.available_selected_candidates,0)

    def test_eager_zero_cap_and_reserve_do_not_create_work(self):
        p,_=self.allocate(16,16,eager_token_cap=0);self.assertEqual(dict(p.eager_budgets),{});self.assertEqual(p.reserve_used,0)
        self.assertEqual(sum(p.normal_budgets.values()),48)

    def test_partial_eager_grants_defer_preserving_admitted_set_and_normal(self):
        p,policy=self.allocate(1,2,draft_token_budget=2,eager_reserve_tokens=1)
        self.assertEqual(p.admitted_eager,(1,2));self.assertEqual(dict(p.eager_budgets),{})
        self.assertGreater(policy.counts['partial_eager_grant_deferred_B1']+policy.counts['partial_eager_grant_deferred_B2'],0)
        self.assertEqual(p.reserve_used,0);self.assertGreaterEqual(p.unused_draft_candidates,1)
        self.assertEqual(dict(p.reserved_by_owner),{});self.assertEqual(sum(p.raw_reserved_by_owner.values()),1)
        self.assertTrue(p.raw_eager_budgets);self.assertNotEqual(dict(p.raw_eager_budgets),dict(p.eager_budgets))

    def test_parent_filter_only_after_existing_eligibility_never_adds_owner(self):
        ready={i:types.SimpleNamespace(proposal_id=100+i) for i in range(5)}
        payload={100+i:dict(plan=types.SimpleNamespace(candidate_budget=i+1)) for i in range(5)};counts={}
        result=filter_eager_parents([0,1,2,3,4],types.SimpleNamespace(ready=ready),payload,counts)
        self.assertEqual(result,[2])
        self.assertEqual(counts['spec_rhythm_eager_parent_B5_deferred_first_candidate'],1);self.assertEqual(counts['spec_rhythm_eager_parent_B1_excluded_unqualified'],1)
        self.assertEqual(filter_eager_parents([],types.SimpleNamespace(ready=ready),payload,counts),[])
        self.assertEqual(counts['spec_rhythm_eager_parent_filter_calls'],2)
        with self.assertRaises(RuntimeError):filter_eager_parents([7],types.SimpleNamespace(ready=ready),payload,{})

    def test_real_original_selector_twice_retains_normal96_plus_future96(self):
        plan,_=self.allocate(32,32)
        native_tree=ast.parse((NATIVE/'tree_budget.py').read_text());names=('TreeCandidateRequest','GlobalTreeCandidateSelection')
        classes=[n for n in native_tree.body if isinstance(n,ast.ClassDef) and n.name in names]
        scope=dict(dataclass=dataclass,Mapping=__import__('typing').Mapping,Sequence=__import__('typing').Sequence,math=__import__('math'))
        exec(compile(ast.Module(body=classes,type_ignores=[]),'<V132 actual Native records>','exec',dont_inherit=True),scope)
        tree=ast.parse((P/'backbone_selection.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='select')
        fn.body=[n for n in fn.body if not isinstance(n,ast.ImportFrom)]
        scope.update(PARENTS=(-1,0,1,2,-1,0,1,2),prod=__import__('math').prod)
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(P/'backbone_selection.py')+':HOST import-only substitution','exec'),scope)
        request=scope['TreeCandidateRequest'];requests={i:request(parents=scope['PARENTS'],token_ids=list(range(8)),conditional_confidences=[.9]*8,max_candidates=3) for i in range(64)}
        from split_shape_v132 import refine_draft_groups
        selected=refine_draft_groups(requests,plan,scope['select'])
        self.assertEqual((selected.normal_candidates,selected.future_eager_candidates),(96,96))
        self.assertEqual(len(selected.original_group_calls),2)
        error=RuntimeError('original selector identity')
        with self.assertRaises(RuntimeError) as caught:refine_draft_groups(requests,plan,lambda *args:(_ for _ in ()).throw(error))
        self.assertIs(caught.exception,error)

    def test_actual_live_builder_seven_rules_extras_and_no_legacy_budget_alias(self):
        from live_decode_patch import build_live_runtime
        source=build_live_runtime();tree=ast.parse(source)
        self.assertEqual(source.count('budget_plan.require_work('),1);self.assertEqual(source.count('budget_plan.require_published_tickets('),1)
        self.assertEqual(source.count('_v123_refine_draft_groups('),1);self.assertEqual(source.count('_v132_filter_eager_parents('),1)
        self.assertNotIn('budget_plan.verification_roof',source);self.assertNotIn('budget_plan.unused_verification_tokens',source)
        calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='shaper.shape']
        self.assertEqual(ast.unparse(next(k.value for k in calls[0].keywords if k.arg=='eager_token_cap')),'effective_eager_cap')
        self.assertLess(source.index('_v132_filter_eager_parents('),source.index('window_estimator.estimate('))
        self.assertIn('_v75_hooks.commit_after_original_finish_cycle',source)

    def test_unknown_live_input_and_removed_real_anchor_rejected(self):
        from runtime_adapter_v132 import adapt_live_source
        with self.assertRaises(RuntimeError):adapt_live_source('pass')
        from ast_adapter_v123 import prepare_runtime_adapter
        raw=(P/'runtime_tree_decode.py').read_text().replace("controller.publish(published_tickets)","controller.publish(other)")
        with self.assertRaises(RuntimeError):prepare_runtime_adapter(raw,hashlib.sha256(raw.encode()).hexdigest())

    def test_original_fresh_decode_policy_target_math_and_clock_bytes_unchanged(self):
        parent=json.loads((P/'PARENT_V132.json').read_text())
        for name in ('tree_entry.py','runtime_tree_decode.py','v63_runtime_reference.py','RUNTIME_SOURCE_PIN.json','budget_policy.py',
          'production_probe_v101.py','poisson_coverage_probe.py','atomic_decode_patch.py','backbone_selection.py',
          'fast_tree_draft_v5.py','root_seed.py','host_target_metadata.py','mixed_target.py','live_runtime_hooks.py','shape_scope_v101.py'):
            self.assertEqual(hashlib.sha256((P/name).read_bytes()).hexdigest(),parent['cloned_static'][name],name)

    def test_mapper_only_exact_actual_six_and_original_exception_identity(self):
        from candidate_controls_v132 import MAPPER_PARTITIONS,BASE_MAPPER_PARTITIONS,OPTIONAL_LARGE_PARTITIONS,mapper_partitions
        from complete_search import envelope
        cfg=json.loads((P/'CANDIDATE_V132.json').read_text())
        self.assertEqual(MAPPER_PARTITIONS,mapper_partitions(cfg))
        self.assertEqual(mapper_partitions({'mapper_extra_large_partitions_enabled':False}),BASE_MAPPER_PARTITIONS)
        self.assertEqual(mapper_partitions({'mapper_extra_large_partitions_enabled':True}),BASE_MAPPER_PARTITIONS|OPTIONAL_LARGE_PARTITIONS)
        with self.assertRaises(RuntimeError):mapper_partitions({'mapper_extra_large_partitions_enabled':1})
        for n,q,f in OPTIONAL_LARGE_PARTITIONS:
            m,physical_s,padding=envelope(n,q,f);self.assertEqual((m,physical_s),(128,32));self.assertEqual(q+sum(padding),m)
        for n,q,f in BASE_MAPPER_PARTITIONS:
            m,s,padding=envelope(n,q,f);self.assertEqual((m,s),(96,32));self.assertEqual(q+sum(padding),m)

    def test_import_guard_checks_actual_existing_origin_and_local_first(self):
        from import_origins_v132 import local_first
        original=list(sys.path)
        try:
            sys.path.insert(0,'/unrelated/extension/examples');local_first();self.assertEqual(sys.path[0],str(P))
            bad=types.ModuleType('chain_metadata');bad.__file__='/root/old-wrong-epoch/chain_metadata.py'
            with patch.dict(sys.modules,{'chain_metadata':bad}):
                with self.assertRaisesRegex(RuntimeError,'actual local helper origin'):local_first()
        finally:sys.path[:]=original

    def test_actual_candidate_shape_install_rollback_policy_identity_and_reserved_exact3(self):
        import candidate_controls_v132 as module
        import target_layout
        policy=Policy()
        def original_shape(instance,**kwargs):return policy
        rhythm=types.SimpleNamespace(SpecRhythmBudgetShaper=type('HostShaper',(),{'shape':original_shape}))
        native=types.SimpleNamespace();active={'clean':False}
        def observe(*args,redundant_cpu):return dict(passed=redundant_cpu is not active['clean'])
        def clean(*args):active['clean']=True;return dict(status='cleaned')
        def restore(*args):active['clean']=False;return dict(status='reinstalled')
        @dataclass
        class Config:
            spec_rhythm_draft_token_budget:object=None
            spec_rhythm_max_eager_tokens:int=5
            spec_rhythm_eager_reserve_tokens:int=0
        original_config=Config()
        engine=types.SimpleNamespace(cache_allocation=None,cache_block_tables=None,config=original_config)
        old_envelope=target_layout.envelope
        with (patch.object(module,'observe_cpu_device_context',side_effect=observe),
              patch.object(module,'restore_redundant_cpu_device_context',side_effect=clean),
              patch.object(module,'reinstall_redundant_cpu_device_context',side_effect=restore)):
            control=module.CandidateControls(native=native,rhythm=rhythm,policy=policy,torch=object(),
                constructor_state=object(),is_draft=True,configuration=dict(target_cpu_context_clean=True,draft_cpu_context_clean=True))
            try:
                selected=control.select('candidate',engine=engine)
                self.assertTrue(selected['typed_draft192']);self.assertTrue(selected['CPU_clean'])
                p=rhythm.SpecRhythmBudgetShaper.shape(types.SimpleNamespace(min_gamma=1,max_gamma=8),plan_id=1,
                    normal_request_indices=tuple(range(32)),eager_request_indices=(),states={i:State() for i in range(32)},
                    projected_wait_ms=0.,context_len=128,batch_size=32,draft_token_budget=192,eager_token_cap=3,
                    eager_reserve_tokens=32,verification_roof=96)
                self.assertEqual(sum(p.normal_budgets.values()),96);self.assertIs(type(policy).allocate if hasattr(type(policy),'allocate') else control.original_allocate,control.original_allocate)
                self.assertIs(target_layout.envelope,control.mapper)
            finally:control.select('original',engine=engine)
            self.assertIs(rhythm.SpecRhythmBudgetShaper.shape,original_shape);self.assertIs(target_layout.envelope,old_envelope)
            self.assertFalse(active['clean']);self.assertFalse(hasattr(native,'_v132_filter_eager_parents'))
            self.assertIs(engine.config,original_config);self.assertEqual(control.config_snapshots['candidate']['spec_rhythm_eager_reserve_tokens'],32)

    def test_real_installed_live_delegate_matches_original_identity_guard_and_old_rejected(self):
        import torch
        import live_decode_patch as live
        import live_poisson_probe as probe
        from typing import Any,Sequence
        shell=probe.original._expected_code(P/'tree_entry.py','fresh_decode')
        old=probe.original._expected_code(P/'runtime_tree_decode.py','_generate_spec_rhythm_tree_decode')
        def cell(value):return (lambda:value).__closure__[0]
        policy=object();oldraw=types.FunctionType(old,{})
        cells=dict(policy=policy,decode_impl=torch.inference_mode()(oldraw))
        original=types.FunctionType(shell,{'P':P},closure=tuple(cell(cells[k]) for k in shell.co_freevars))
        cls=type('HostNative',(),{'_generate_spec_rhythm_tree_decode':original})
        native=types.SimpleNamespace(NativePearlEngine=cls,torch=torch,Any=Any,Sequence=Sequence,
            PearlPipelineState=type('State',(),{}),NativeSamplingParams=type('Params',(),{}))
        with patch.object(live,'require_bounded_scope',return_value=None):live.install(native,lambda engine:None)
        installed=cls._generate_spec_rhythm_tree_decode
        self.assertIs(installed.__code__,original.__code__);self.assertIs(installed.__globals__,original.__globals__)
        self.assertIs(dict(zip(installed.__code__.co_freevars,installed.__closure__))['policy'].cell_contents,policy)
        probe._assert_live_decode(installed.__get__(object()))
        with self.assertRaises(probe.original.CoverageRejected):probe._assert_live_decode(original.__get__(object()))
        cls._generate_spec_rhythm_tree_decode=original
        self.assertIs(cls._generate_spec_rhythm_tree_decode,original)

    def test_full_caller_factory_scope_and_unknown_Tensor_factory_reject(self):
        from factory_scope_v132 import collect
        report=collect();self.assertFalse(report['global_factory_equivalence_proved'])
        self.assertTrue(report['source_dataflow_closed_for_V132_selected_greedy_callers'])
        self.assertGreater(report['default_device_sites'],30)
        self.assertEqual(report['Native_source_file_count'],33)
        self.assertEqual(set(report['role_caller_scope']),{'draft','target'})
        self.assertTrue(any(r['path'].endswith('fast_tree_draft_v5.py') and r['default_device_applies'] for r in report['factory_sites']))
        with self.assertRaisesRegex(RuntimeError,'unreviewed omitted-device factory'):
            collect([(P/'future_unreviewed_tensor_factory.py','def bad(input):\n    return torch.as_tensor(input)\n')])

    def test_new_source_contract_executes_actual_S35_and_full_manifest_before_constructor(self):
        from check_v132 import source_contract
        value=source_contract();inputs=value['pre_model_actual_input_contract']
        self.assertEqual((inputs['S35_scopes'],inputs['S35_phases']),(21,26))
        self.assertEqual(inputs['source_workload']['463']['request_count'],463)
        self.assertTrue(inputs['actual_fixture_helpers_executed'])
        self.assertFalse(value['component_device_qualification_inherited'])

    def test_true_per_owner_reservation_branch_and_filtered_final_map_are_immutable(self):
        from dataclasses import replace
        p,_=self.allocate(32,32)
        self.assertEqual(dict(p.reserved_by_owner),{i:1 for i in range(32,64)})
        self.assertEqual(dict(p.raw_reserved_by_owner),dict(p.reserved_by_owner));self.assertEqual(p.reserve_used,32)
        with self.assertRaises(TypeError):p.reserved_by_owner[32]=2
        with self.assertRaises(ValueError):replace(p,reserved_by_owner={32:1},reserve_used=1)
        with self.assertRaises(ValueError):replace(p,reserved_by_owner={64:1},reserve_used=1)

    def test_actual_extension_runpy_entry_places_private_source_first_and_refuses_existing_wrong_module(self):
        evidence=[]
        for mode in ('actual-runpy','wrong-existing-origin'):
            argv=[sys.executable,'-I','-S','-B',str(P/'host_entry_v132.py'),mode]
            done=subprocess.run(argv,text=True,capture_output=True,timeout=5)
            self.assertEqual(done.returncode,0,done.stdout+done.stderr)
            record=json.loads(done.stdout);self.assertTrue(record['passed']);self.assertFalse(record['runtime_main_executed'])
            self.assertEqual(record['forbidden_import_attempts'],[]);evidence.append(dict(argv=argv,exit_code=done.returncode,record=record))
        global ACTUAL_ENTRY_EVIDENCE
        ACTUAL_ENTRY_EVIDENCE=evidence

    def test_new_config_exact_three_draft_fields_and_no_qualified_flag_write(self):
        tree=ast.parse((P/'candidate_controls_v132.py').read_text());call=next(n for n in ast.walk(tree) if isinstance(n,ast.Call) and ast.unparse(n.func)=='replace')
        self.assertEqual({k.arg:ast.literal_eval(k.value) for k in call.keywords},dict(spec_rhythm_draft_token_budget=192,spec_rhythm_max_eager_tokens=3,spec_rhythm_eager_reserve_tokens=32))
        from component_source_binding_v132 import bind
        cfg=json.loads((P/'CANDIDATE_V132.json').read_text())
        proof=cfg['component_actual_evidence']['parent_B3_eager']
        component=bind('parent_B3_eager',json.loads(Path(proof['receipt']).read_text()))
        self.assertEqual(len(component['actual_shared_helpers']),17)
        self.assertGreaterEqual(len(component['Native_source_files']),6)
        for name in ('candidate_controls_v132.py','candidate_runtime_v132.py','split_shape_v132.py'):
            tree=ast.parse((P/name).read_text())
            self.assertFalse(any(isinstance(n,(ast.Assign,ast.AnnAssign)) and any(isinstance(v,ast.Attribute) and v.attr=='runtime_validated' for v in ast.walk(n)) for n in ast.walk(tree)))

    def test_tag_private_startup_paths_preserve_original_writers_and_prior_evidence(self):
        import tempfile
        from startup_progress_v132 import prepare,require_link
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            first=prepare('trial1',root)
            (root/'startup-progress'/'original-stage.json').write_text('old evidence')
            second=prepare('trial2',root)
            self.assertEqual((Path(first['actual_directory'])/'original-stage.json').read_text(),'old evidence')
            self.assertNotEqual(first['actual_directory'],second['actual_directory'])
            from private_storage_v132 import snapshot as storage_snapshot,require_limits,MAX_ATTEMPT_BYTES,MAX_AGGREGATE_BYTES
            (root/'local-output'/'trial2').mkdir(parents=True)
            (root/'local-output'/'trial2'/'trace.json').write_bytes(b'12345')
            (root/'prepared-static.json').write_bytes(b'prepared')
            (root/'startup-progress'/'new-stage.json').write_bytes(b'new')
            storage=storage_snapshot('trial2',root);require_limits(storage)
            self.assertEqual(storage['attempt_private_bytes'],8)
            self.assertEqual(storage['private_bytes'],8)
            self.assertEqual(storage['aggregate_private_bytes'],len(b'old evidence')+8+len(b'prepared'))
            with self.assertRaisesRegex(RuntimeError,'current attempt'):require_limits(dict(storage,private_bytes=MAX_ATTEMPT_BYTES,attempt_private_bytes=MAX_ATTEMPT_BYTES))
            with self.assertRaisesRegex(RuntimeError,'aggregate'):require_limits(dict(storage,aggregate_private_bytes=MAX_AGGREGATE_BYTES))
            self.assertEqual(require_link('trial2',root),second)
            with self.assertRaises(RuntimeError):require_link('trial1',root)
            with self.assertRaises(FileExistsError):prepare('trial2',root)
        runtime=ast.parse((P/'candidate_runtime_v132.py').read_text())
        fn=next(n for n in runtime.body if isinstance(n,ast.FunctionDef) and n.name=='worker')
        source=ast.unparse(fn)
        self.assertLess(source.index('startup_progress = require_link('),source.index('engine = native.NativePearlEngine('))
        self.assertIn("receipt['startup_progress'] = prepare(args.tag)",ast.unparse(ast.parse((P/'supervise_candidate_v132.py').read_text())))

    def test_unmanaged_or_legacy_startup_paths_and_bad_tags_are_refused(self):
        import tempfile
        from startup_progress_v132 import prepare
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);legacy=root/'startup-progress';legacy.mkdir()
            evidence=legacy/'original.json';evidence.write_text('retain')
            with self.assertRaises(RuntimeError):prepare('trial1',root)
            self.assertEqual(evidence.read_text(),'retain')
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);other=root/'unmanaged';other.mkdir()
            (root/'startup-progress').symlink_to(other)
            with self.assertRaises(RuntimeError):prepare('trial1',root)
            for tag in ('../escape','a/b','',None):
                with self.assertRaises((RuntimeError,TypeError)):prepare(tag,root)

    def test_original_window_boundary_snapshot_is_lossless_pure_and_explicit_missing(self):
        from window_cost_state_v132 import snapshot,validate
        from copy import deepcopy
        scope={'math':__import__('math')}
        tree=ast.parse((P/'packed_window.py').read_text());klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PackedTreeWindowEstimator')
        exec(compile(ast.Module(body=[klass],type_ignores=[]),'<actual original window class>','exec'),scope)
        cls=scope['PackedTreeWindowEstimator'];window=cls()
        window.normal_ms={16:12.25,32:20.5};window.overhead_ms={1:3.5};window.target_verify_ms=18.75
        window.communication_ms=.125;window.eager_fixed_overhead_ms=3.5;window.probe_offers=7;window.eager_observations=2
        before=deepcopy(vars(window));engine=types.SimpleNamespace(_dynamic_tree_window_estimator=window,_dynamic_tree_generation=4)
        record=snapshot(engine);self.assertEqual(vars(window),before);self.assertEqual(record,json.loads(json.dumps(record,allow_nan=False)))
        self.assertEqual(record['target_verify_ms'],18.75);self.assertEqual(record['normal_ms'],[dict(bucket=16,ms=12.25),dict(bucket=32,ms=20.5)])
        self.assertFalse(snapshot(types.SimpleNamespace())['available'])
        bad=deepcopy(record);bad['target_verify_ms']=float('nan')
        with self.assertRaises(RuntimeError):validate(bad)
        bad=deepcopy(record);bad['summary']['normal_bucket16_ms']=1.
        with self.assertRaises(RuntimeError):validate(bad)
        source=(P/'candidate_phases_v132.py').read_text()
        self.assertLess(source.index("window-state-before"),source.index('trace=run_trace('))
        self.assertGreater(source.index("window-state-after"),source.index('trace=run_trace('))
        self.assertNotIn('.estimate(', (P/'window_cost_state_v132.py').read_text())

if __name__=='__main__':unittest.main()
