"""Four bounded HOST checks of actual source adapters and entry scheduling.

Tensor/backend imports are forbidden by host160. Fake execution below tests
orchestration only; no model, numerical or live performance proof is inferred.
"""
import ast
import copy
import functools
import gzip
import json
import math
import statistics
import sys
import tempfile
import types
import typing
import unittest
from unittest.mock import patch

from window160 import BASE, P, FILENAME, adapt_source, parse_windows, runtime_bindings, sha


def parent_modules():
    if str(BASE) not in sys.path:
        sys.path.insert(1, str(BASE))
    import live_decode_patch
    import production_probe_v101
    return live_decode_patch, production_probe_v101


def cell(value):
    return (lambda: value).__closure__[0]


class Host(unittest.TestCase):
    def test_actual_live_builder_cap_inverse_and_argument_execution(self):
        live, original = parent_modules()
        source = live.build_live_runtime()
        original_tree = ast.parse(source)
        for cap in (32, 24, 16):
            tree = ast.parse(adapt_source(source, cap))
            fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
            inserted = fn.body.pop(1)
            self.assertEqual(ast.unparse(inserted), 'initial_batch_size = min(initial_batch_size, '+str(cap)+')')
            self.assertEqual(ast.dump(tree), ast.dump(original_tree))
            # Execute the inserted actual statement, not a parallel cap formula.
            for incoming, expected in ((64, cap), (8, 8)):
                ns = dict(initial_batch_size=incoming)
                exec(compile(ast.fix_missing_locations(ast.Module(body=[inserted], type_ignores=[])),
                             'HOST160-actual-cap-statement', 'exec'), ns)
                self.assertEqual(ns['initial_batch_size'], expected)
            with self.assertRaises(RuntimeError):
                adapt_source(adapt_source(source, cap), cap)
        self.assertEqual(parse_windows('32,24,16'), (32, 24, 16))
        for invalid in ('32,32', '64', '0', ''):
            with self.assertRaises((RuntimeError, ValueError)):
                parse_windows(invalid)
        self.assertEqual(original.source_scope(463)['request_count'], 463)

    def test_actual_isolated_installer_shell_and_trace_AST(self):
        from probe160 import trace_source, make_trace
        live, original = parent_modules()
        import live_poisson_probe as observation
        from types import FunctionType
        # Execute the original installer with a HOST inference decorator and
        # explicit permit stub in a private module copy; original globals stay.
        fake_live = types.ModuleType('live_decode_patch')
        fake_live.__dict__.update(vars(live))
        permits = []
        fake_live.require_bounded_scope = lambda: permits.append('HOST-only')
        decorator_ns = {}
        exec(compile('def factory(func):\n def decorate_context(*a, **k):\n  return func(*a, **k)\n return decorate_context\n',
                     '/HOST/torch/utils/_contextlib.py', 'exec'), decorator_ns)
        def inference_mode():
            return lambda fn: functools.update_wrapper(decorator_ns['factory'](fn), fn)
        native = types.ModuleType('HOST160_native')
        native.__dict__.update(torch=types.SimpleNamespace(inference_mode=inference_mode),
            PearlPipelineState=object, NativeSamplingParams=object, Sequence=typing.Sequence, Any=typing.Any)
        code = original.original._expected_code(BASE/'tree_entry.py', 'fresh_decode')
        closure = tuple(cell(None) for _ in code.co_freevars)
        shell = FunctionType(code, dict(P=BASE), 'fresh_decode', closure=closure)
        cls = type('NativePearlEngine', (), dict(_generate_spec_rhythm_tree_decode=shell))
        native.NativePearlEngine = cls
        old_live_install = live.install
        with patch.dict(sys.modules, {'live_decode_patch': fake_live}):
            install, assertion, state = runtime_bindings(24)
            install(native, lambda e: None)
            assertion(cls()._generate_spec_rhythm_tree_decode)
            self.assertIn('initial_batch_size = min(initial_batch_size, 24)', state['source'])
            trace, trace_state = make_trace(24)
            self.assertIs(trace.__globals__['install'].__globals__['P'], P)
            self.assertIsNot(trace.__globals__, vars(original))
        self.assertEqual(permits, ['HOST-only'])
        self.assertIs(live.install, old_live_install)
        self.assertEqual(FILENAME, '<V160-original-live-with-active-window>')
        original_fn = copy.deepcopy(next(n for n in ast.parse((BASE/'production_probe_v101.py').read_text()).body
                                            if isinstance(n, ast.FunctionDef) and n.name == 'run_trace'))
        for cap in (32, 24, 16):
            fn = next(n for n in trace_source(cap).body if isinstance(n, ast.FunctionDef))
            capture = next(n for n in fn.body if isinstance(n, ast.FunctionDef) and n.name == '_v160_capture_event')
            ns = dict(stream=[], _v160_first=set(), time=types.SimpleNamespace(time=lambda: 101.5))
            exec(compile(ast.fix_missing_locations(ast.Module(body=[capture], type_ignores=[])),
                         'HOST160-actual-callback', 'exec'), ns)
            event = dict(request_index=7, elapsed_seconds=2.0, token_ids=[1])
            ns['_v160_capture_event'](event);ns['_v160_capture_event'](event)
            self.assertEqual(event, dict(request_index=7, elapsed_seconds=2.0, token_ids=[1]))
            self.assertEqual(ns['stream'][0]['first_token_wall_seconds160'], 101.5)
            self.assertNotIn('first_token_wall_seconds160', ns['stream'][1])
            fn.body = [n for n in fn.body if not (isinstance(n, ast.FunctionDef) and n.name == '_v160_capture_event')
                       and not (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_v160_first' for t in n.targets))]
            for n in ast.walk(fn):
                if isinstance(n, ast.Compare) and ast.unparse(n) == '0 < peak <= '+str(cap):
                    n.comparators[-1] = ast.Constant(64)
                if isinstance(n, ast.keyword) and n.arg == 'actual_capacity64_scope_completed':
                    n.value = ast.parse('peak > 32', mode='eval').body
                if isinstance(n, ast.keyword) and n.arg == 'on_token_commit':
                    n.value = ast.parse('lambda event: stream.append(dict(event))', mode='eval').body
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'dict':
                    n.keywords = [k for k in n.keywords if k.arg not in
                        ('configured_capacity', 'admission_window160', 'actual_window_covered160')]
            location = next(i for i,n in enumerate(original_fn.body) if isinstance(n, ast.ImportFrom) and n.module == 'live_decode_patch')
            fn.body.insert(location, copy.deepcopy(original_fn.body[location]))
            self.assertEqual(ast.dump(fn), ast.dump(original_fn))

    def test_same_engine_schedule_generation_reset_release_and_failure_restore(self):
        import pair160
        actions, resets, calls, seals = [], [], [], []
        engine = types.SimpleNamespace(rank=0, topology=types.SimpleNamespace(draft_leader_rank=0),
            config=types.SimpleNamespace(max_num_seqs=64, max_num_queued_seqs=463,
                spec_rhythm_max_target_batch=32, spec_rhythm_verification_budget=96,
                spec_rhythm_online_prefill=True, enable_continuous_batching=True,
                enable_prefix_caching=False, num_kvcache_blocks=544),
            _cache_storage_sequence_capacity=527, _tree_graph_scratch_sequence_ids=tuple(range(463,527)),
            cache_allocation=None, cache_block_tables=None,
            graph_runner=types.SimpleNamespace(graph_cache_sealed=False), _dynamic_tree_generation=0)
        def seal():
            seals.append(engine._dynamic_tree_generation);engine.graph_runner.graph_cache_sealed=True
        engine.seal_graph_cache=seal
        controls=types.SimpleNamespace(select=lambda mode,**kw:actions.append(mode), observe=lambda:dict(mode=actions[-1] if actions else 'original'))
        snapshots=types.ModuleType('candidate_phases_v132')
        def snapshot(e,reset_normal_admission=False):
            if reset_normal_admission:resets.append(e._dynamic_tree_generation+1)
            return dict(normal_admission=dict(generation=e._dynamic_tree_generation+int(reset_normal_admission)),
                        gain_state=dict(generation=e._dynamic_tree_generation), policy={},FastV5={})
        snapshots.split_snapshot=snapshot;snapshots.split_delta=lambda a,b:dict(policy={},FastV5={});snapshots.update_snapshot=lambda e:dict(tasks=0)
        proposals=types.ModuleType('combined_controls_v113');proposals.proposal_counter_snapshot=lambda e:dict(proposals=0)
        window=types.ModuleType('window_cost_state_v132');window.snapshot=lambda e:dict(generation=e._dynamic_tree_generation)
        fail=[]
        def maker(cap):
            def trace(e,*,count,role,directory,qualified_s35_scope):
                calls.append((id(e),cap,role,count,e.graph_runner.graph_cache_sealed,qualified_s35_scope))
                if fail:raise RuntimeError('HOST explicit trace failure')
                e._dynamic_tree_generation+=1
                return dict(source_generation=e._dynamic_tree_generation,cache_released=True,
                            admission_window160=cap,actual_capacity64_scope_completed=False,artifact=dict(HOST=True))
            return trace,{}
        scope=object();stage=lambda name,fn:fn()
        with tempfile.TemporaryDirectory(prefix='HOST160-pair-',dir=P) as temp:
            from pathlib import Path
            with patch.dict(sys.modules,{'candidate_phases_v132':snapshots,'combined_controls_v113':proposals,'window_cost_state_v132':window}),\
                 patch.object(pair160,'require_ready',lambda:dict(HOST=True)),patch.object(pair160,'make_trace',maker),\
                 patch.object(pair160,'full_metrics',lambda x:dict(HOST=True)):
                result=pair160.run_pair(engine,controls=controls,scope=scope,stage=stage,directory=Path(temp)/'success')
                self.assertEqual([(c[1],c[2]) for c in calls],[(32,'warmup65'),(16,'warmup65'),
                    (32,'zero65-1'),(16,'zero65-1'),(32,'zero65-2'),(16,'zero65-2'),(32,'full463'),(16,'full463')])
                self.assertEqual({c[0] for c in calls},{id(engine)})
                self.assertTrue(all(c[-1] is scope for c in calls));self.assertEqual(resets,list(range(1,9)))
                self.assertEqual(seals,[6]);self.assertTrue(all(c[4] is False for c in calls[:6]))
                self.assertTrue(all(c[4] is True for c in calls[6:]));self.assertEqual(actions,['candidate','original'])
                self.assertFalse(result['constructor_repeated']);self.assertEqual(len(result['traces']),8)
                engine.graph_runner.graph_cache_sealed=False;fail.append(True)
                with self.assertRaisesRegex(RuntimeError,'explicit trace failure'):
                    pair160.run_pair(engine,controls=controls,scope=scope,stage=stage,directory=Path(temp)/'failure',admission_windows=(24,))
                self.assertEqual(actions[-2:],['candidate','original'])

    def test_original_clocks_and_actual_first_callback_TTFT_separate(self):
        from metrics160 import full_metrics
        from pathlib import Path
        rows=[];events=[]
        classes=('tight','normal','loose')
        for i in range(463):
            rows.append(dict(request_id='HOST'+str(i),completion_token_ids=[0]*256,
                arrival_ts=1000+i*.001,slo_class=classes[i%3],slo_tpot_ms=40.,
                observed_tpot_ms=3.,paper_tpot_ms=3.*255/256))
            events.append(dict(request_index=i,request_id=rows[-1]['request_id'],elapsed_seconds=1.,
                token_ids=[0],finished=False,first_token_wall_seconds160=1001+i*.001))
        events += [dict(request_index=i,request_id=r['request_id'],elapsed_seconds=2.,
                        token_ids=[0]*255,finished=True) for i,r in enumerate(rows)]
        trace=dict(generate_results=rows,stream_events=events,host_generate_call_seconds=3.,elapsed_seconds=3.,
            native_elapsed_seconds=2.9,admission_window160=24,active_peak=24,actual_window_covered160=True)
        with tempfile.TemporaryDirectory(prefix='HOST160-clock-',dir=P) as temp:
            path=Path(temp)/'synthetic.json.gz';path.write_bytes(gzip.compress(json.dumps(trace).encode()))
            result=full_metrics(dict(path=str(path),sha256=sha(path)))
            self.assertTrue(math.isclose(result['TTFT160']['median_ms'],1000.))
            self.assertTrue(math.isclose(result['Native_paper']['median_ms'],3.*255/256))
            self.assertTrue(math.isclose(result['host_stream_paper']['median_ms'],1000/256))
            self.assertTrue(math.isclose(result['raw_tokens_per_second'],463*256/3))
            self.assertEqual(result['TTFT160']['extra_wall_samples'],463)
            trace['stream_events'][1]['request_index']=0
            path.write_bytes(gzip.compress(json.dumps(trace).encode()))
            with self.assertRaises((ValueError,RuntimeError)):
                full_metrics(dict(path=str(path),sha256=sha(path)))


if __name__ == '__main__':
    unittest.main()
