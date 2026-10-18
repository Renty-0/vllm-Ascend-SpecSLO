"""Actual capacity64 tree traces on original arrivals/output/SLO; HOST import."""
from dataclasses import replace
import gzip
import hashlib
import json
from pathlib import Path
import time

import poisson_coverage_probe as original
from live_poisson_probe import (LiveEntryObservation, PARTITIONS, _assert_live_decode,
                               _observe_begin, engine_value)

P = Path(__file__).resolve().parent
SCHEMA = 'V101.S35-actual-trace.v1'
PREFIX_SHA = {65: '72f76e28c26a62cfacbe62686de294fa12fdd7c0ec5bc64947542909fa1a4b5b',
             463: original.MANIFEST_SHA256}


def require(value, message):
    if not value:
        raise RuntimeError('V101: '+message)


def source_scope(count):
    require(count in PREFIX_SHA, 'only original first65/full463 scopes are supported')
    raw = original.MANIFEST.read_bytes()
    require(original._sha(raw) == original.MANIFEST_SHA256, 'public manifest changed')
    lines = raw.splitlines(keepends=True)
    require(len(lines) == 463 and original._sha(b''.join(lines[:count])) == PREFIX_SHA[count],
            'original request prefix changed')
    rows = [json.loads(line) for line in lines[:count]]
    deadlines = {'tight': 40., 'normal': 50., 'loose': 150.}
    require(len({r['request_id'] for r in rows}) == count and all(
        r['max_tokens'] == 256 and r['per_request_gamma'] == 4
        and r['slo_class'] in deadlines and r['slo_tpot_ms'] == deadlines[r['slo_class']]
        and original._finite(r['arrival_offset_sec']) and r['arrival_offset_sec'] >= 0
        and isinstance(r['prompt'], str) and r['prompt'] for r in rows), 'request controls changed')
    require([r['arrival_offset_sec'] for r in rows] == sorted(r['arrival_offset_sec'] for r in rows),
            'original Poisson arrival order changed')
    return dict(request_count=count, manifest_path=str(original.MANIFEST),
        manifest_sha256=original.MANIFEST_SHA256, prefix_sha256=PREFIX_SHA[count],
        output_tokens=256, arrival_seed=0, requests=original._projection(rows),
        maxseq=64, target=32, C=96, Q=128, KV=544,
        independent_AR_all_requests_qualified=False)


def request_rows(count):
    source_scope(count)
    return [json.loads(line) for line in original.MANIFEST.read_bytes().splitlines()[:count]]


def save_gzip(directory, name, value):
    """Serialize once, then atomically retain exact compressed bytes; no reserve."""
    raw = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    compressed = gzip.compress(raw, mtime=0)
    path = Path(directory)/name
    with path.open('xb') as stream:
        stream.write(compressed)
    return dict(path=str(path), bytes=len(compressed), uncompressed_bytes=len(raw),
                sha256=hashlib.sha256(compressed).hexdigest())


def generation_timing(host_seconds, results):
    """Use the measured original call boundary; retain Native's inner timer."""
    require(original._finite(host_seconds) and host_seconds > 0, 'actual generate call timer is invalid')
    native_seconds = results[0]['elapsed_seconds'] if results else None
    require(native_seconds is None or original._finite(native_seconds) and native_seconds >= 0,
            'original Native inner timer is invalid')
    return dict(elapsed_seconds=host_seconds, native_elapsed_seconds=native_seconds)


class CapacityEntryObservation(LiveEntryObservation):
    def install(self):
        engine = self.engine
        _assert_live_decode(engine._generate_spec_rhythm_tree_decode)
        names = ('target_tree_forward', '_prefill_and_sample_target_batch', '_generate_spec_rhythm_tree_decode')
        self.originals = {name: getattr(engine, name) for name in names}
        self.previous = {name: engine.__dict__.get(name, original._ABSENT) for name in names}
        def target(plans, *args, **kwargs):
            if not engine.is_draft:
                # This reads integer host geometry/numel only, never tensor
                # contents or a GPU->CPU value during the timed generation.
                self.observe(lambda: self.target_calls.append(original._target_record(plans, args, kwargs)))
            return self.originals[names[0]](plans, *args, **kwargs)
        def prefill(prompts, states, *args, **kwargs):
            def record():
                ids = kwargs.get('sequence_ids', args[0] if args else None)
                ids = list(range(len(prompts))) if ids is None else list(ids)
                self.prefill_calls.append(dict(sequence_ids=ids,
                    cached_tokens=[engine.cache_allocation.num_cached_tokens[i] for i in ids],
                    prompt_lengths=[len(prompt) for prompt in prompts],
                    _actual_prompt_tokens=[tuple(prompt) for prompt in prompts],
                    precomputed_target_tokens=kwargs.get('precomputed_target_tokens') is not None,
                    draft_prefill_completed=kwargs.get('draft_prefill_completed', False)))
            self.observe(record)
            return self.originals[names[1]](prompts, states, *args, **kwargs)
        def decode(*args, **kwargs):
            self.entry_calls += 1
            count = len(self.rows)
            require(self.entry_calls == 1 and not args and len(kwargs['request_params']) == count
                and len(kwargs['draft_states']) == len(kwargs['target_states']) == count,
                'original keyword decode scope/state count differs')
            self.origin = self.origin_provider(engine)
            params = [replace(param, arrival_ts=self.origin['leader_origin']+row['arrival_offset_sec'])
                      for param, row in zip(kwargs['request_params'], self.rows, strict=True)]
            self.arrivals = [param.arrival_ts for param in params]
            for states in (kwargs['draft_states'], kwargs['target_states']):
                for state, arrival in zip(states, self.arrivals, strict=True):
                    state.arrival_ts = arrival
            return self.originals[names[2]](*args, **dict(kwargs, request_params=params))
        self.wrappers = dict(zip(names, (target, prefill, decode)))
        for name, wrapper in self.wrappers.items():
            wrapper.__func__ = getattr(self.originals[name], '__func__', None)
            setattr(engine, name, wrapper)
        return self


def finalize_prefill_records(records):
    """Hash actual HOST argument snapshots after the generation timer ends."""
    for record in records:
        prompts = record.pop('_actual_prompt_tokens')
        require([len(prompt) for prompt in prompts] == record['prompt_lengths'],
                'actual prefill HOST snapshot length changed')
        require(len(prompts) == len(record['cached_tokens']) == len(record['sequence_ids'])
                and all(0 <= cached <= len(prompt) for cached, prompt in
                        zip(record['cached_tokens'], prompts, strict=True)),
                'actual prefill cache/prompt rows differ')
        record['prompt_sha256'] = [original._digest(prompt) for prompt in prompts]
        record['uncached_queries'] = sum(len(prompt)-cached for prompt, cached in
                                        zip(prompts, record['cached_tokens'], strict=True))
    return records


class HostPageHighwater:
    """Constant work per acquired page, <=32 host items per released row.

    Observe original allocator mutations after they execute. No changed
    allocation order, device tensors, per-cycle scan, or collective is added.
    """
    def __init__(self, engine):
        self.cache = engine.prefix_cache
        self.current = sum(block.ref_count > 0 for block in self.cache._blocks)
        self.peak = self.current
        self.acquired = self.released = 0
        self.originals, self.previous, self.wrappers = {}, {}, {}
        self.allocation_rows = []

    def install(self):
        cache = self.cache
        names = ('_acquire_block', '_release_table', 'allocate')
        self.originals = {name: getattr(cache, name) for name in names}
        self.previous = {name: cache.__dict__.get(name, original._ABSENT) for name in names}
        def acquire(*args, **kwargs):
            result = self.originals['_acquire_block'](*args, **kwargs)
            if cache._blocks[result[0]].ref_count == 1:
                self.current += 1
                self.acquired += 1
                self.peak = max(self.peak, self.current)
            return result
        def release(table):
            count = sum(block >= 0 and cache._blocks[block].ref_count == 1 for block in table)
            result = self.originals['_release_table'](table)
            self.current -= count
            self.released += count
            return result
        def allocate(*args, **kwargs):
            result = self.originals['allocate'](*args, **kwargs)
            self.allocation_rows.append(len(result.block_tables))
            return result
        self.wrappers = dict(zip(names, (acquire, release, allocate)))
        for name, wrapper in self.wrappers.items():
            setattr(cache, name, wrapper)
        return self

    def detach(self):
        for name, wrapper in self.wrappers.items():
            require(self.cache.__dict__.get(name) is wrapper, 'HOST allocator observer ownership changed')
            previous = self.previous[name]
            if previous is original._ABSENT:
                delattr(self.cache, name)
            else:
                setattr(self.cache, name, previous)
        actual = sum(block.ref_count > 0 for block in self.cache._blocks)
        require(actual == self.current == 0, 'allocator highwater accounting or final release differs')
        return dict(referenced_page_highwater=self.peak, final_referenced_pages=actual,
            acquired_pages=self.acquired, released_pages=self.released,
            allocation_storage_rows=self.allocation_rows, pool_blocks=self.cache.num_blocks,
            observation='original HOST acquire/release boundaries; no NPU tensor read')


def run_trace(engine, *, count, role, directory, qualified_s35_scope):
    from diagnostic_authorization import require_bounded_scope
    require_bounded_scope()
    from vllm_ascend.spec_decode.pearl import native_engine as native
    from startup_receipts_v75 import stage4, NATIVE_COUNTERS, FAILURE_COUNTERS
    from local_mixed_qualification import native_snapshot, native_delta
    from live_generation_probe import RuntimeObservation
    from live_runtime_factory import make_live_runtime_hooks
    from live_decode_patch import install
    from no_mixed_prefill_v143 import (MODE, prepare_production_contract,
                                      install_on_fresh_hooks)
    stage = lambda label, callback: stage4(engine, 'V101/'+role+'/'+label, callback)
    stage('actual-qualified-structure',lambda:qualified_s35_scope.require(engine))
    source = stage('source', lambda: source_scope(count))
    rows = request_rows(count)
    require(engine.config.max_num_seqs == 64 and engine.cache_allocation is None
            and engine.cache_block_tables is None, 'fresh capacity64 allocation is required')
    original._assert_original_decode(engine._generate_spec_rhythm_tree_decode)
    def prepare():
        # Native original tokenizer controls; every prompt is tokenized once
        # outside generation timing, no generated continuation is an input.
        prompts, _ = original.tokenize_prefix(engine.tokenizer, rows)
        return prompts, [native.NativeSamplingParams(**r) for r in original.sampling_specifications(rows)]
    prompts, params = stage('prepare', prepare)
    observations, begin_inputs, observation_errors, detach_begin, close_aborts = [], [], [], [], []
    prefill_contracts, prefill_controls = [], []
    def factory(actual):
        hooks = make_live_runtime_hooks(actual,allowed_partitions=PARTITIONS,qualified_s35_scope=qualified_s35_scope)
        prefill_controls.append(install_on_fresh_hooks(hooks, prefill_contracts[0]))
        observations.append(RuntimeObservation(hooks).install())
        detach_begin.append(_observe_begin(hooks, begin_inputs, observation_errors, close_aborts))
        return hooks
    cls = native.NativePearlEngine
    previous_class = cls.__dict__['_generate_spec_rhythm_tree_decode']
    installed_class = None
    observation = CapacityEntryObservation(engine, rows)
    pages = HostPageHighwater(engine)
    stream, restored_class = [], False
    before_generation = getattr(engine, '_dynamic_tree_generation', 0)
    sealed_before = bool(engine.graph_runner.graph_cache_sealed)
    try:
        def bind():
            nonlocal installed_class
            prefill_contracts.append(prepare_production_contract(P))
            install(native, factory)
            installed_class = cls.__dict__['_generate_spec_rhythm_tree_decode']
            observation.install()
            pages.install()
        stage('install-original-live', bind)
        before = stage('Native-before', lambda: native_snapshot(engine))
        policy_before = stage('policy-before', lambda: original._policy_snapshot(engine))
        def generate():
            start = time.perf_counter()
            results = engine.generate_batch(prompts, params, max_rounds=None,
                request_admission_callback=None, on_token_commit=lambda event: stream.append(dict(event)),
                include_graph_metrics=False)
            return results, time.perf_counter()-start
        results, host_seconds = stage('original-generate', generate)
        after = stage('Native-after', lambda: native_snapshot(engine))
        policy_after = stage('policy-after', lambda: original._policy_snapshot(engine))
    except BaseException as _v140_trace_error:
        try:
            from exception_evidence_v140 import retain_trace_failure
            retain_trace_failure(engine, _v140_trace_error, role=role, count=count, directory=directory,
                source=source, prompts=prompts, params=params, stream=stream, observation=observation,
                observations=observations, begin_inputs=begin_inputs, observation_errors=observation_errors,
                close_aborts=close_aborts, before=locals().get('before'), policy_before=locals().get('policy_before'),
                sealed_before=sealed_before, before_generation=before_generation,
                native_current=lambda: native_snapshot(engine), policy_current=lambda: original._policy_snapshot(engine))
        except BaseException:
            pass  # Do not mask the original failed stage or its original finally.
        raise
    finally:
        try:
            if observation.wrappers:
                observation.detach()
        finally:
            try:
                for detach in reversed(detach_begin):
                    detach()
                for value in observations:
                    value.detach()
            finally:
                if installed_class is not None:
                    require(cls.__dict__.get('_generate_spec_rhythm_tree_decode') is installed_class,
                            'live class delegate changed ownership')
                    cls._generate_spec_rhythm_tree_decode = previous_class
                    restored_class = True
    require(len(observations) == 1, 'one real hooks generation required')
    hooks = observations[0].hooks
    require(engine.cache_allocation is None and engine.cache_block_tables is None, 'trace cache survived release')
    page_stats = stage('host-page-highwater-detach', pages.detach)
    generation = getattr(engine, '_dynamic_tree_generation', None)
    require(generation == before_generation+1, 'original fresh source generation did not advance exactly once')
    counters = dict(engine.last_worker_decode_counters)
    peak = counters['spec_rhythm_peak_active_requests']
    require(0 < peak <= 64, 'actual active peak exceeds independent64 scope')
    delta = native_delta(before, after)
    require(all(delta[key] == 0 for key in FAILURE_COUNTERS), 'Native failure/fallback during actual trace')
    if role != 'warmup65':
        require(all(value == 0 for value in delta.values()), 'zero-discovery/measurement trace changed Native inventory')
    if results is not None:
        require(engine.rank == engine.topology.target_leader_rank and len(results) == count
            and all(len(row['completion_token_ids']) == 256 for row in results), 'original leader outputs are incomplete')
    require(not observation.errors and not observation_errors and observation.restored and restored_class,
            'original observations/bindings were not preserved')
    finalize_prefill_records(observation.prefill_calls)
    timing = generation_timing(host_seconds, results)
    elapsed = timing['elapsed_seconds']
    record = dict(schema=SCHEMA, rank=engine.rank, role=role, scope_count=count,
        source=source, manifest_sha256=source['manifest_sha256'], prefix_sha256=source['prefix_sha256'],
        requests=source['requests'], tokenized_inputs=[list(prompt) for prompt in prompts],
        sampling=original.sampling_specifications(rows), source_generation=generation,
        source_generation_before=before_generation, fresh_allocation=True,
        origin=observation.origin['leader_origin'], arrival_origin=observation.origin,
        absolute_arrivals=observation.arrivals, elapsed_seconds=elapsed,
        native_elapsed_seconds=timing['native_elapsed_seconds'],
        host_generate_call_seconds=host_seconds, timing_scope='original generate arrival entry through final output; startup/tokenization/warmup excluded',
        S35_structural_scope=__import__('shape_scope_v101').scope_report(qualified_s35_scope,engine),
        normal_target_calls=observation.target_calls, standalone_prefill_calls=observation.prefill_calls,
        production_prefill_mode=MODE, production_prefill_control=prefill_controls[0].evidence(),
        logical_events=observations[0].events, begin_inputs=begin_inputs, close_abort_identities=close_aborts,
        boundary_events=list(hooks.boundary_events), terminal_evidence=hooks.evidence(),
        stream_events=stream, generate_results=results,
        native=dict(before=before, after=after, delta=delta),
        forbidden_native_counters=list(NATIVE_COUNTERS), failure_native_counters=list(FAILURE_COUNTERS),
        policy_before=policy_before, policy_after=policy_after,
        decode_counters=counters, phase_seconds=dict(getattr(engine, 'last_worker_decode_phase_seconds', {})),
        allocator=page_stats, active_peak=peak, actual_capacity64_scope_completed=peak > 32,
        wrappers_restored=observation.restored, class_delegate_restored=restored_class,
        cache_released=True, sealed_before=sealed_before, sealed_after=bool(engine.graph_runner.graph_cache_sealed),
        independent_AR_all_requests_qualified=False, online_qualified=False, formal_Goodput=False)
    artifact = stage('retain-original-trace', lambda: save_gzip(directory,
        f'rank{engine.rank}-{role}-actual-trace.json.gz', record))
    return dict(role=role, source_generation=generation, scope_count=count, artifact=artifact,
        native=record['native'], forbidden_native_counters=list(NATIVE_COUNTERS),
        sealed_before=sealed_before, sealed_after=record['sealed_after'], cache_released=True,
        active_peak=peak, actual_capacity64_scope_completed=peak > 32,
        elapsed_seconds=elapsed, completed_output_tokens=0 if results is None else count*256)
