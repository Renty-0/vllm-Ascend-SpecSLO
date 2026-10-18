"""Explicit unscored live first48 diagnostic and independent eager reference.

No installation, model construction or device operation occurs on import or
source/audit calls. An allowlist describes diagnostic inputs, not qualification.
"""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from types import FunctionType

import poisson_coverage_probe as original
from live_generation_probe import (RuntimeObservation, original_clock_order,
    _chunk_contract, _derive_contract)


P = Path(__file__).resolve().parent
SCHEMA = 'V85.live-Poisson48-eager-reference.v1'
V84 = Path('/root/SpecSLO/budget-calibration-20261015/tree-live-poisson-policy-counter-v84')
V84_ROOT_AUDIT = Path('/root/SpecSLO/budget-calibration-20261015/budget-diagnostics/ROOT_V84_ACTUAL_POISSON48_AUDIT.json')
ROOT_AUDIT_SHA256 = 'b6d464260465ae47384d9a793941b1246f5e329ad03b820c653adb95ef6b17ac'
RAW_SHA256 = (
    '7b67d4a31c59b87ce748f6f74bcdac68dcd9fe0ee1722d56262a44c91270a218',
    '4a2c356d7c0499553bc46f6576b363879dbde4b91ef4e3d188940e03a34a6759',
    '0edd7310f05daf7acc344b0e5f5829b862eadb8bde7a2c792427ac8889f0cf75',
    '3419968fadcac704d0e6d4bc9084f66c791f5618489a9de38024880ed0bafecc')
PARTITIONS = frozenset((
    (1, (2,)), (1, (5,)), (2, (1, 5)), (2, (2, 5)), (2, (5, 2)), (2, (5, 3)), (2, (5, 5)),
    (3, (2, 5, 5)), (3, (3, 3, 3)), (3, (3, 5, 5)), (3, (4, 5, 4)), (3, (5, 5, 5)),
    (4, (5, 5, 5, 5)), (5, (5, 5, 5, 5, 5)),
    (6, (2, 5, 1, 5, 5, 5)), (6, (4, 4, 2, 4, 5, 4)), (6, (4, 4, 4, 4, 4, 4)), (6, (5, 5, 5, 5, 5, 5)),
    (7, (2, 5, 5, 5, 5, 5, 5)), (7, (3, 5, 5, 5, 5, 5, 5)), (7, (4, 4, 4, 5, 4, 5, 4)),
    (7, (5, 5, 1, 5, 5, 5, 5)), (7, (5, 5, 2, 5, 5, 5, 5)), (7, (5, 5, 5, 5, 5, 5, 5)),
    (8, (5, 5, 5, 5, 5, 5, 5, 1)), (8, (5, 5, 5, 5, 5, 5, 5, 5)),
    (9, (4, 4, 4, 4, 4, 4, 4, 4, 5)), (9, (5, 5, 5, 5, 5, 5, 2, 5, 5)), (9, (5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (10, (5, 2, 5, 5, 5, 5, 5, 5, 5, 5)), (10, (5, 4, 4, 4, 4, 4, 4, 4, 4, 5)),
    (10, (5, 5, 5, 4, 4, 4, 5, 4, 5, 4)), (10, (5, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (11, (5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5)), (11, (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (12, (5, 5, 2, 4, 4, 4, 4, 4, 4, 4, 4, 5)), (12, (5, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (12, (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (13, (1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)), (13, (3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (13, (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (14, (5, 5, 3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)), (14, (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (15, (2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)), (15, (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (16, (1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (16, (2, 5, 5, 4, 5, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4)),
    (16, (3, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (16, (4, 3, 5, 1, 5, 4, 4, 5, 4, 5, 4, 4, 4, 4, 4, 4)),
    (16, (4, 5, 4, 4, 5, 5, 4, 5, 5, 5, 4, 4, 5, 4, 5, 4)),
    (16, (4, 5, 5, 1, 4, 5, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4)),
    (16, (4, 5, 5, 3, 4, 5, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4)),
    (16, (4, 5, 5, 5, 4, 4, 5, 4, 4, 5, 5, 4, 5, 4, 5, 4)),
    (16, (5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (16, (5, 3, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (16, (5, 4, 4, 2, 4, 5, 5, 4, 4, 5, 4, 4, 4, 4, 5, 4)),
    (16, (5, 4, 4, 4, 5, 5, 4, 4, 5, 4, 4, 4, 4, 5, 4, 5)),
    (16, (5, 4, 4, 5, 5, 2, 4, 5, 4, 4, 4, 4, 5, 4, 5, 4)),
    (16, (5, 4, 4, 5, 5, 4, 4, 5, 4, 4, 4, 4, 5, 4, 5, 4)),
    (16, (5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (16, (5, 5, 2, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (16, (5, 5, 4, 5, 4, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5)),
    (16, (5, 5, 5, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (16, (5, 5, 5, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 5, 5)),
    (16, (5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5))))
_require, _digest = original._require, original._digest
_finite, _sha = original._finite, original._sha
LIVE_FILENAME = '<V81-bounded-live-mixed-prefill>'


def _build_live_source():
    from live_decode_patch import build_live_runtime
    return build_live_runtime()


def _live_source():
    import atomic_decode_patch as atomic
    import live_decode_patch as live
    paths = (atomic.REFERENCE, Path(atomic.__file__), Path(live.__file__), Path(__file__))
    dependencies = (live.build_live_runtime, live.build_blocked_runtime, live._target_only_call,
        atomic._statements, atomic._one, atomic._assigns, atomic._is_call,
        atomic._direct_function, atomic._insert_after, atomic._insert_before, _build_live_source)
    return _derive_contract('live-Poisson-source', tuple(path.read_bytes() for path in paths),
                            dependencies, _build_live_source)


def live_poisson_partitions():
    return PARTITIONS


def _scope(allowed):
    _require(type(allowed) is frozenset and allowed == PARTITIONS,
             'Live Poisson requires its exact explicit source-bound65 partition set')
    return [[n, list(b)] for n, b in sorted(allowed)]


def require_live_poisson_source():
    base = original.require_poisson_coverage_source()
    raw = V84_ROOT_AUDIT.read_bytes()
    _require(_sha(raw) == ROOT_AUDIT_SHA256, 'Pinned actual V84 root audit differs')
    receipt = json.loads(raw)
    _require(receipt['original_job_status'] == 'complete'
        and receipt['actual_coverage_records_accepted'] is True and receipt['errors'] == [],
        'Original coverage evidence was not actually accepted')
    reports, paths = [], {}
    for rank, wanted in enumerate(RAW_SHA256):
        path = V84/'local-output/original-poisson48-counter-v1'/f'rank{rank}-local-diagnostic.json'
        raw = path.read_bytes()
        _require(_sha(raw) == wanted and receipt['raw_reports'][str(path)] == wanted,
                 'Pinned actual V84 raw report differs: '+str(rank))
        report = json.loads(raw)
        _require(report['rank'] == rank and report['action'] == 'poisson-coverage-probe'
            and report['errors'] == [], 'Pinned actual V84 worker envelope differs')
        reports.append(report['poisson_coverage_probe'])
        paths[str(path)] = wanted
    target = reports[1]['target_calls']
    _require(reports[0]['target_calls'] == [] and all(reports[rank]['target_calls'] == target for rank in (2, 3)),
             'Pinned actual V84 target traces differ')
    observed = frozenset((row['N'], tuple(row['budgets'])) for row in target)
    _require(len(PARTITIONS) == 65 and observed == PARTITIONS
        and all((n, (5,)*n) in PARTITIONS for n in range(1, 17)),
        'Literal diagnostic65 scope differs from actual original coverage')
    return dict(schema=SCHEMA, original_input_and_decode=base,
        V84_raw_sha256=paths, V84_root_audit_path=str(V84_ROOT_AUDIT),
        V84_root_audit_sha256=ROOT_AUDIT_SHA256, allowed_partitions=_scope(PARTITIONS),
        generated_live_source_sha256=_sha(_live_source().encode()),
        original_clock_order=original_clock_order(),
        numerical_qualification_inferred=False, online_qualified=False, cost_or_Goodput=False)


def _assert_live_decode(bound):
    shell = getattr(bound, '__func__', None)
    _require(isinstance(shell, FunctionType)
        and shell.__code__ == original._expected_code(P/'tree_entry.py', 'fresh_decode')
        and Path(shell.__globals__.get('P', '')).resolve() == P, 'Original live fresh_decode shell changed')
    cells = dict(zip(shell.__code__.co_freevars, shell.__closure__ or ()))
    _require('decode_impl' in cells, 'Actual live delegate closure is missing')
    delegate = cells['decode_impl'].cell_contents
    decorated = False
    while isinstance(delegate, FunctionType) and hasattr(delegate, '__wrapped__'):
        closure = dict(zip(delegate.__code__.co_freevars, delegate.__closure__ or ()))
        _require(delegate.__code__.co_name == 'decorate_context'
            and Path(delegate.__code__.co_filename).as_posix().endswith('/torch/utils/_contextlib.py')
            and 'func' in closure and closure['func'].cell_contents is delegate.__wrapped__,
            'Actual live inference decorator changed')
        decorated = True
        delegate = delegate.__wrapped__
    expected = original._find_code(compile(_live_source(), LIVE_FILENAME, 'exec'), '_generate_spec_rhythm_tree_decode')
    _require(decorated and isinstance(delegate, FunctionType) and delegate.__code__ == expected,
             'Instance arrival wrapper did not capture the installed LIVE delegate')


class LiveEntryObservation:
    """Observe normal target/prefill calls and rebase only actual LIVE entry."""
    def __init__(self, engine, rows, *, origin_provider=original._entry_origin):
        self.engine, self.rows, self.origin_provider = engine, rows, origin_provider
        self.target_calls, self.prefill_calls, self.errors = [], [], []
        self.entry_calls, self.origin, self.arrivals = 0, None, []
        self.previous, self.originals, self.wrappers = {}, {}, {}
        self.restored = False

    def observe(self, callback):
        try:
            callback()
        except Exception as error:
            self.errors.append(type(error).__name__+': '+str(error))

    def install(self):
        engine = self.engine
        _assert_live_decode(engine._generate_spec_rhythm_tree_decode)
        names = ('target_tree_forward', '_prefill_and_sample_target_batch', '_generate_spec_rhythm_tree_decode')
        self.originals = {name:getattr(engine, name) for name in names}
        self.previous = {name:engine.__dict__.get(name, original._ABSENT) for name in names}
        def target(plans, *args, **kwargs):
            if not engine.is_draft:
                self.observe(lambda:self.target_calls.append(original._target_record(plans, args, kwargs)))
            return self.originals[names[0]](plans, *args, **kwargs)
        def prefill(prompts, states, *args, **kwargs):
            def record():
                ids = kwargs.get('sequence_ids', args[0] if args else None)
                ids = list(range(len(prompts))) if ids is None else list(ids)
                cached = [engine.cache_allocation.num_cached_tokens[i] for i in ids]
                self.prefill_calls.append(dict(sequence_ids=ids, cached_tokens=cached,
                    prompt_lengths=[len(prompt) for prompt in prompts],
                    prompt_sha256=[_digest(prompt) for prompt in prompts],
                    uncached_queries=sum(len(prompt)-value for prompt, value in zip(prompts, cached, strict=True)),
                    precomputed_target_tokens=kwargs.get('precomputed_target_tokens') is not None,
                    draft_prefill_completed=kwargs.get('draft_prefill_completed', False)))
            self.observe(record)
            return self.originals[names[1]](prompts, states, *args, **kwargs)
        def decode(*args, **kwargs):
            self.entry_calls += 1
            _require(self.entry_calls == 1 and not args and len(kwargs['request_params']) == 48
                and len(kwargs['draft_states']) == len(kwargs['target_states']) == 48,
                'One actual keyword LIVE decode entry with48 states is required')
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

    def detach(self):
        changed = []
        for name, wrapper in self.wrappers.items():
            if engine_value(self.engine, name) is not wrapper:
                changed.append(name)
                continue
            prior = self.previous[name]
            if prior is original._ABSENT:
                delattr(self.engine, name)
            else:
                setattr(self.engine, name, prior)
        self.restored = not changed
        _require(not changed, 'Live observer ownership changed: '+repr(changed))


def engine_value(engine, name):
    return engine.__dict__.get(name, original._ABSENT)


def _observe_begin(hooks, inputs, errors, close_aborts):
    existed, previous = 'begin_target_cycle' in hooks.__dict__, hooks.__dict__.get('begin_target_cycle')
    method = hooks.begin_target_cycle
    def forwarding(*args, **kwargs):
        try:
            row = hooks.transactions.current
            if row is not None:
                inputs.append(dict(cycle_id=kwargs['cycle_id'], owner=row.lease.sequence_id,
                    identity=[row.lease.identity.request_index, row.lease.identity.nonce, row.lease.identity.allocation_epoch],
                    cursor=row.committed_cursor, target_ids=list(kwargs['target_indices']),
                    budgets=[payload['plan'].candidate_budget for payload in kwargs['target_payloads']],
                    contexts=[payload['plan'].prefix_len for payload in kwargs['target_payloads']]))
        except Exception as error:
            errors.append(type(error).__name__+': '+str(error))
        return method(*args, **kwargs)
    hooks.begin_target_cycle = forwarding
    close_existed = 'close_scope_before_original_release' in hooks.__dict__
    close_previous = hooks.__dict__.get('close_scope_before_original_release')
    close_method = hooks.close_scope_before_original_release
    def closing(*args, **kwargs):
        try:
            row = hooks.transactions.current
            if row is not None:
                identity = row.lease.identity
                close_aborts.append([identity.request_index, identity.nonce, identity.allocation_epoch])
        except Exception as error:
            errors.append(type(error).__name__+': '+str(error))
        return close_method(*args, **kwargs)
    hooks.close_scope_before_original_release = closing
    def detach():
        _require(hooks.__dict__.get('begin_target_cycle') is forwarding, 'Live begin observer ownership changed')
        _require(hooks.__dict__.get('close_scope_before_original_release') is closing, 'Live close observer ownership changed')
        if existed:
            hooks.begin_target_cycle = previous
        else:
            delattr(hooks, 'begin_target_cycle')
        if close_existed:
            hooks.close_scope_before_original_release = close_previous
        else:
            delattr(hooks, 'close_scope_before_original_release')
    return detach


def _eager_reference(engine, prompts, params, stage):
    """Sequential fresh allocations; input is original prompts and own predictions."""
    import torch
    from mixed_target import target_failure_consensus
    from local_mixed_qualification import save
    _require(engine.cache_allocation is None and engine.cache_block_tables is None,
             'Candidate cache must be fully released before independent AR')
    output = None if engine.is_draft else []
    for request, (prompt, controls) in enumerate(zip(prompts, params, strict=True)):
        _require(controls.max_tokens == 256 and controls.ignore_eos is True
            and controls.temperature == controls.draft_temperature == 0., 'Actual AR controls differ')
        label = 'AR/'+str(controls.request_id)
        generated, calls = [], []
        try:
            stage(label+'/allocate', lambda:engine._allocate_cache([list(prompt)], enable_prefix_caching=False))
            for step in range(controls.max_tokens):
                @torch.inference_mode()
                def forward():
                    if engine.is_draft:
                        return None
                    tokens = list(prompt) if step == 0 else [generated[-1]]
                    positions = list(range(len(prompt))) if step == 0 else [len(prompt)+step-1]
                    owners = [0]*len(tokens)
                    engine._ensure_cache_capacity(owners, positions)
                    pos, metadata = engine.model.make_attention_metadata(owners, positions,
                        engine.cache_block_tables, engine._cache_slot_mapping(owners, positions),
                        use_fused_infer_attention=False)
                    hidden = engine.model(torch.tensor(tokens, dtype=torch.long, device=engine.device), pos, metadata)
                    logits = engine.model.compute_logits(hidden[-1:])[:, :engine.draft_vocab_size]
                    target_failure_consensus(engine,
                        ~(torch.isfinite(hidden).all() & torch.isfinite(logits).all()),
                        'Independent AR nonfinite: '+label+'/step'+str(step))
                    return int(logits.argmax(dim=-1).cpu().item())
                token = stage(label+'/step'+str(step)+'/forward', forward)
                stage(label+'/step'+str(step)+'/Native-finite', engine._vote_spec_rhythm_prefill_finiteness)
                if not engine.is_draft:
                    tokens = list(prompt) if step == 0 else [generated[-1]]
                    calls.append(dict(step=step, query_count=len(tokens),
                        start=0 if step == 0 else len(prompt)+step-1,
                        end=len(prompt) if step == 0 else len(prompt)+step,
                        input_sha256=_digest(tokens), hidden_logits_and_Native_vote_finite=True))
                    generated.append(token)
            if not engine.is_draft:
                output.append(dict(request_index=request, request_id=controls.request_id,
                    prompt_sha256=_digest(prompt), max_tokens=controls.max_tokens,
                    completion_token_ids=generated, calls=calls,
                    execution='direct Native eager PA; fresh allocation; own previous predictions'))
        finally:
            stage(label+'/release', engine._release_cache)
        stage(label+'/checkpoint', lambda:save(P/'live-progress', f'rank{engine.rank}-AR-request{request}.json',
            dict(schema=SCHEMA, rank=engine.rank, request_index=request,
                request_id=controls.request_id, completed_steps=controls.max_tokens,
                completed_requests=request+1, reference_row=None if engine.is_draft else output[-1],
                status='actual-reference-request-complete-unscored',
                online_qualified=False, cost_or_Goodput=False)))
    return output


def run_live_poisson_probe(engine, *, make_hooks, allowed_partitions):
    from diagnostic_authorization import require_live_poisson_scope, require_bounded_scope
    require_live_poisson_scope()
    require_bounded_scope()
    from vllm_ascend.spec_decode.pearl import native_engine as native
    from startup_receipts_v75 import stage4
    from local_mixed_qualification import native_snapshot, native_delta
    from live_runtime_hooks import LiveRuntimeHooks
    from live_decode_patch import install
    _require(isinstance(engine, native.NativePearlEngine) and engine.device.type == 'npu',
             'Actual live Poisson requires an existing Native NPU engine')
    scope = _scope(allowed_partitions)
    _require(callable(make_hooks), 'An explicit real live factory is required')
    stage = lambda name, callback:stage4(engine, 'live-Poisson-'+name, callback)
    source = stage('source', require_live_poisson_source)
    rows = original._manifest_rows(original.MANIFEST.read_bytes())
    def prepare():
        cfg = engine.config
        _require(cfg.enable_continuous_batching and cfg.enable_preemptive_scheduling
            and cfg.enable_spec_rhythm and cfg.spec_rhythm_online_prefill
            and not cfg.enable_prefix_caching and not cfg.spec_rhythm_kv_ready_arrivals
            and cfg.max_num_seqs == 32 and cfg.spec_rhythm_tree_width == 2 and cfg.spec_rhythm_tree_depth == 4
            and engine.cache_allocation is None and engine.cache_block_tables is None,
            'Original capacity32/tree/arrival/prefix-cache configuration differs')
        original._assert_original_decode(engine._generate_spec_rhythm_tree_decode)
        _require('_generate_spec_rhythm_tree_decode' not in engine.__dict__, 'Prior instance decode observer is forbidden')
        prompts, tokens = original.tokenize_prefix(engine.tokenizer, rows)
        params = [native.NativeSamplingParams(**row) for row in original.sampling_specifications(rows)]
        return prompts, tokens, params
    prompts, tokens, params = stage('prepare', prepare)
    observations, begin_inputs, observation_errors, detach_begin, close_aborts = [], [], [], [], []
    def factory(actual):
        hooks = make_hooks(actual, allowed_partitions=allowed_partitions)
        _require(isinstance(hooks, LiveRuntimeHooks), 'Factory must return actual LiveRuntimeHooks')
        observations.append(RuntimeObservation(hooks).install())
        detach_begin.append(_observe_begin(hooks, begin_inputs, observation_errors, close_aborts))
        return hooks
    cls = native.NativePearlEngine
    previous_class = cls.__dict__['_generate_spec_rhythm_tree_decode']
    installed_class, observation = None, LiveEntryObservation(engine, rows)
    stream, restored_class = [], False
    try:
        def install_live():
            nonlocal installed_class
            install(native, factory)
            installed_class = cls.__dict__['_generate_spec_rhythm_tree_decode']
            _assert_live_decode(engine._generate_spec_rhythm_tree_decode)
            observation.install()  # Captures the NEW class delegate, never the original one.
        stage('install-and-observe-live', install_live)
        before = stage('Native-before', lambda:native_snapshot(engine))
        policy_before = stage('policy-before', lambda:original._policy_snapshot(engine))
        results = stage('original-generate', lambda:engine.generate_batch(prompts, params,
            max_rounds=None, request_admission_callback=None,
            on_token_commit=lambda event:stream.append(dict(event)), include_graph_metrics=False))
        after = stage('Native-after', lambda:native_snapshot(engine))
        policy_after = stage('policy-after', lambda:original._policy_snapshot(engine))
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
                    _require(cls.__dict__.get('_generate_spec_rhythm_tree_decode') is installed_class,
                             'Live class delegate ownership changed')
                    cls._generate_spec_rhythm_tree_decode = previous_class
                    restored_class = True
    _require(len(observations) == 1, 'Exactly one original generation/live hook instance is required')
    hooks = observations[0].hooks
    stage('candidate-fully-released', lambda:_require(engine.cache_allocation is None
        and engine.cache_block_tables is None, 'Candidate allocation survived release'))
    stage('restored-original-delegate', lambda:original._assert_original_decode(engine._generate_spec_rhythm_tree_decode))
    from local_mixed_qualification import save
    stage('candidate-before-AR-checkpoint', lambda:save(P/'live-progress', f'rank{engine.rank}-candidate-before-AR.json',
        dict(schema=SCHEMA, status='actual-candidate-complete-reference-pending', rank=engine.rank,
            source=source, allowed_partitions=scope, tokenized_inputs=tokens,
            sampling=original.sampling_specifications(rows),
            decode_entry_count=observation.entry_calls, arrival_origin=observation.origin,
            absolute_arrivals=observation.arrivals, normal_target_calls=observation.target_calls,
            standalone_prefill_calls=observation.prefill_calls, logical_events=observations[0].events,
            begin_inputs=begin_inputs, close_abort_identities=close_aborts,
            boundary_events=list(hooks.boundary_events), terminal_evidence=hooks.evidence(),
            observation_errors=observation.errors+observation_errors,
            wrappers_restored=observation.restored, class_delegate_restored=restored_class,
            candidate_cache_released=engine.cache_allocation is None and engine.cache_block_tables is None,
            native_before=before, native_after=after, native_delta=native_delta(before, after),
            policy_before=policy_before, policy_after=policy_after,
            actual_decode_counters=dict(engine.last_worker_decode_counters),
            generate_results=results, stream_events=stream,
            numerical_qualification_inferred=False, all65_qualified=False, full463_qualified=False,
            online_qualified=False, cost_or_Goodput=False)))
    reference = _eager_reference(engine, prompts, params, stage)
    return dict(schema=SCHEMA, status='raw-complete', rank=engine.rank, is_draft=engine.is_draft,
        target_leader_rank=engine.topology.target_leader_rank,
        draft_ranks=list(engine.topology.draft_ranks), target_ranks=list(engine.topology.target_ranks),
        source=source, allowed_partitions=scope, tokenized_inputs=tokens,
        sampling=original.sampling_specifications(rows), decode_entry_count=observation.entry_calls,
        arrival_origin=observation.origin, absolute_arrivals=observation.arrivals,
        normal_target_calls=observation.target_calls, standalone_prefill_calls=observation.prefill_calls,
        observation_errors=observation.errors+observation_errors, wrappers_restored=observation.restored,
        class_delegate_restored=restored_class, actual_live_delegate_verified=True,
        logical_events=observations[0].events, begin_inputs=begin_inputs, close_abort_identities=close_aborts,
        boundary_events=list(hooks.boundary_events), terminal_evidence=hooks.evidence(),
        unreleased_dispositions_remaining=bool(hooks._unreleased_dispositions),
        native_before=before, native_after=after, native_delta=native_delta(before, after),
        policy_before=policy_before, policy_after=policy_after,
        actual_decode_counters=dict(engine.last_worker_decode_counters),
        generate_results=results, stream_events=stream, eager_AR_reference=reference,
        candidate_and_reference_cache_released=engine.cache_allocation is None and engine.cache_block_tables is None,
        observed_control_counts={kind:sum(event['kind'] == kind for event in observations[0].events)
            for kind in ('fallback-request', 'fallback-reset', 'abort', 'submit-error')},
        numerical_qualification_inferred=False, all65_qualified=False, full463_qualified=False,
        online_qualified=False, cost_or_Goodput=False)


def _clocks(values, count):
    _require(type(values) is dict and set(values) == {str(i) for i in range(count)}
        and all(_finite(value) and value >= 0 for value in values.values()), 'Actual live runtime clocks differ')


def _audit_logical(report):
    from flexible_scope import envelope
    tokens, events = report['tokenized_inputs'], report['logical_events']
    _require(type(events) is list, 'Actual live logical events are missing')
    current, pending, submitted, entry, last_abort = None, None, False, None, None
    fallback_requested = False
    reservations, commits, publications, aborted, seen, cycles = [], [], [], [], set(), set()
    for event in events:
        _require(type(event) is dict and type(event.get('kind')) is str, 'Malformed actual live event')
        kind = event['kind']
        if kind == 'reservation':
            identity, owner = event['identity'], event['owner']
            _require(current is None and last_abort is None and not fallback_requested
                and type(owner) is int and 0 <= owner < 48
                and type(identity) is list and len(identity) == 3 and identity[0] == owner
                and type(identity[1]) is str and bool(identity[1])
                and type(identity[2]) is int and identity[2] > 0 and tuple(identity) not in seen
                and event['cursor'] == 0 and event['prompt_length'] == tokens[owner]['token_count']
                and event['prompt_sha256'] == tokens[owner]['token_sha256']
                and event['initial_fixture_cursor'] is None and event['initial_fixture_scope'] is None,
                'Each real reservation must start at cursor0 without fixture adoption')
            seen.add(tuple(identity))
            current = dict(identity=identity, owner=owner, length=event['prompt_length'], cursor=0,
                           last_cycle=-1, final=None)
            reservations.append(dict(current))
        elif kind == 'plan':
            _require(current is not None and pending is None and current['final'] is None,
                     'A mixed plan has no current unused real reservation')
            n, budgets, cycle = len(event['target_ids']), tuple(event['budgets']), event['cycle_id']
            _require((n, budgets) in PARTITIONS and type(cycle) is int and cycle > current['last_cycle']
                and cycle not in cycles and len(set(event['target_ids'])) == n
                and all(type(i) is int and 0 <= i < 48 and i != current['owner'] for i in event['target_ids'])
                and len(event['proposal_ids']) == n and len(set(event['proposal_ids'])) == n
                and all(type(i) is int and i >= 0 for i in event['proposal_ids']),
                'Actually visited mixed geometry/IDs/proposals left the explicit65 scope')
            shape = envelope(n, budgets, current['length']-current['cursor'])
            _require((event['start'], event['end'], event['final']) ==
                (current['cursor'], current['cursor']+shape.prefill_queries, shape.final)
                and event['query_partition'] == list(shape.query_partition),
                'Actual mixed chunk cursor/partition differs from its real reservation')
            pending, submitted = dict(event, owner=current['owner'], identity=current['identity']), False
            cycles.add(cycle)
        elif kind == 'submitted':
            _require(pending is not None and not submitted and event['cycle_id'] == pending['cycle_id']
                and event['phase'] == 'submitted', 'Missing/duplicate actual mixed submission')
            submitted = True
        elif kind == 'commit':
            _require(pending is not None and submitted and all(event[key] == pending[key]
                for key in ('cycle_id', 'start', 'end', 'final')) and event['cursor'] == pending['end']
                and event['phase'] == ('committed-final-awaiting-publication' if pending['final'] else 'reserved')
                and (type(event['first_token']) is int and event['first_token'] >= 0
                     if pending['final'] else event['first_token'] is None),
                'Actual mixed submit/commit/cursor transition differs')
            _clocks(event['clocks_before_cycle_charge'], 48)
            commit = dict(pending, **{key:value for key, value in event.items() if key not in pending})
            commits.append(commit)
            current.update(cursor=event['cursor'], last_cycle=event['cycle_id'], final=commit if event['final'] else None)
            pending, submitted = None, False
        elif kind == 'publication-entry':
            _require(current is not None and current['final'] is not None and entry is None
                and event['owner'] == current['owner'] and current['owner'] not in event['active_before']
                and len(event['active_before']) < 32 and len(set(event['active_before'])) == len(event['active_before'])
                and all(type(i) is int and 0 <= i < 48 for i in event['active_before']),
                'Original mixed publication lost its pending owner/capacity')
            before, after = current['final']['clocks_before_cycle_charge'], event['clocks_after_cycle_charge']
            _clocks(after, 48)
            _require(after[str(current['owner'])] == before[str(current['owner'])]
                and all(after[key] >= value for key, value in before.items())
                and all(after[str(i)] > before[str(i)] for i in event['active_before']),
                'Mixed publication preceded old-active clock charging or charged its own prompt')
            entry = event
        elif kind == 'publication':
            _require(current is not None and entry is not None and event['owner'] == current['owner']
                and event['first_token'] == current['final']['first_token']
                and event['decode_elapsed_ms'] == event['decode_start_elapsed_ms'] ==
                    entry['clocks_after_cycle_charge'][str(current['owner'])]
                and event['reservation_remaining'] is False, 'Mixed first-token publication differs or repeats')
            publications.append(dict(event, identity=current['identity'], cycle_id=current['last_cycle']))
            current, entry = None, None
        elif kind == 'fallback-request':
            _require(current is not None and pending is None and not fallback_requested,
                     'Fallback request has no owned idle reservation or repeats before reset')
            fallback_requested = True
        elif kind == 'abort':
            _require(current is not None and pending is None and entry is None
                and event['owner'] == current['owner'] and type(event['physical_invalidation_required']) is bool
                and event['physical_invalidation_required'] == (current['cursor'] > 0),
                'Actual abort does not match its partial physical-write history')
            last_abort = dict(current)
            aborted.append(last_abort)
            current = None
        elif kind == 'fallback-reset':
            _require(current is None and last_abort is not None, 'Fallback reset lacks actual allocator invalidation/abort')
            last_abort = None
            fallback_requested = False
        else:
            raise original.CoverageRejected('Unexpected live event/failure: '+kind)
    close_aborts = report['close_abort_identities']
    _require(type(close_aborts) is list and not fallback_requested
        and (close_aborts == [] if last_abort is None else close_aborts == [last_abort['identity']]),
        'Fallback request/abort/reset is incomplete, or terminal abort lacks its real close boundary')
    _require(current is None and pending is None and entry is None and commits
        and any(commit['start'] == 0 for commit in commits)
        and any(len(commit['target_ids']) > 1 for commit in commits),
        'Actual from0 mixed work and N>1 mixed commit are required; idle/N1 is insufficient')
    inputs = report['begin_inputs']
    _require(type(inputs) is list and all(type(row) is dict for row in inputs)
        and len({row['cycle_id'] for row in inputs}) == len(inputs), 'Actual common live cycle inputs differ')
    by_cycle = {row['cycle_id']:row for row in inputs}
    for commit in commits:
        begin = by_cycle.get(commit['cycle_id'])
        _require(begin is not None and begin['owner'] == commit['owner'] and begin['identity'] == commit['identity']
            and begin['cursor'] == commit['start'] and begin['target_ids'] == commit['target_ids']
            and begin['budgets'] == commit['budgets'] and len(begin['contexts']) == len(begin['target_ids'])
            and all(type(c) is int and 0 <= c < 4096 for c in begin['contexts']),
            'Actual mixed commit differs from its common logical input/identity')
    return dict(reservations=reservations, commits=commits, publications=publications,
                aborted=aborted, begin_by_cycle=by_cycle)


def _audit_boundaries(report, logical):
    events = report['boundary_events']
    _require(type(events) is list and bool(events), 'Actual live boundary evidence is absent')
    generic, names = [], []
    known = {'original-V45-chunk-source-contract', 'reservation-lease-packets',
        'live-prefix-write-scope', 'live-query-write-footprint-bound',
        'original-tree-commit-evidence', 'actual-prompt-draft-materialization-ledger'}
    for event in events:
        _require(type(event) is dict and type(event.get('stage')) is str, 'Malformed live boundary event')
        if 'sequence' not in event:
            _require(event['stage'] in known, 'Unexpected non-stage live evidence')
            continue
        seq, name, ranks = event['sequence'], event['stage'], event['ranks']
        _require(type(seq) is int and seq == len(generic) and type(ranks) is list and len(ranks) == 4
            and sorted(row.get('rank') for row in ranks) == list(range(4))
            and all(type(row.get('rank')) is int and row.get('sequence') == seq
                and row.get('stage') == name and row.get('error') is None for row in ranks)
            and event['boundary_kind'] == 'host-status-agreement', 'Actual host boundary rank/error/order differs')
        generic.append(event)
        names.append(name)
    commits, publications, reservations = logical['commits'], logical['publications'], logical['reservations']
    expected = {
        'live-original-source-and-config':1, 'live-original-V45-chunk-binding':1,
        'live-original-allocation-tracker-install':1,
        **{name:len(reservations) for name in ('live-reservation-observed-owner', 'reservation-original-allocator',
            'reservation-logical-consensus', 'transaction-reservation-host-commit')},
        **{name:len(report['begin_inputs']) for name in ('live-mixed-cycle-host-inputs', 'mixed-cycle-shape-consensus')},
        **{name:len(commits) for name in ('live-prefix-write-declaration', 'live-mixed-layout-and-metadata-preflight',
            'mixed-target-submit-at-original-join', 'original-draft-chunk-preflight',
            'original-draft-prompt-local-forward', 'original-native-chunk-finite-and-broadcast',
            'live-prefix-structural-write-proof-at-original-join', 'transaction-prepared-host-commit',
            'original-finish-cycle-physical-fence', 'original-finish-cycle-logical-consensus',
            'transaction-cursor-host-commit', 'actual-prompt-draft-ledger-after-tree-commit')},
        **{name:len(publications) for name in ('original-publication-preflight', 'original-publication-token-consensus',
            'transaction-publication-host-claim', 'original-first-token-publication', 'transaction-publication-host-complete')},
        'original-reservation-physical-invalidation':len(logical['aborted'])}
    _require(Counter(names) == Counter({key:value for key, value in expected.items() if value}),
             'Actual original chunk/commit/publication/invalidation stage counts differ')
    phases = ('live-prefix-write-declaration', 'live-mixed-layout-and-metadata-preflight',
        'mixed-target-submit-at-original-join', 'original-draft-chunk-preflight', 'original-draft-prompt-local-forward',
        'original-native-chunk-finite-and-broadcast', 'live-prefix-structural-write-proof-at-original-join',
        'transaction-prepared-host-commit', 'original-finish-cycle-physical-fence',
        'original-finish-cycle-logical-consensus', 'transaction-cursor-host-commit',
        'actual-prompt-draft-ledger-after-tree-commit')
    positions = {name:[i for i, value in enumerate(names) if value == name] for name in expected}
    for index, commit in enumerate(commits):
        indices = [positions[name][index] for name in phases]
        _require(indices == sorted(set(indices)), 'Original mixed phase order changed')
        if commit['final']:
            publication = next(i for i, value in enumerate(publications) if value['cycle_id'] == commit['cycle_id'])
            indices = [indices[-1]]+[positions[name][publication] for name in (
                'original-publication-preflight', 'original-publication-token-consensus',
                'transaction-publication-host-claim', 'original-first-token-publication', 'transaction-publication-host-complete')]
            _require(indices == sorted(set(indices)), 'Final publication precedes its original commit/clock boundary')
    selected = lambda name:[row for row in events if row['stage'] == name]
    _require(selected('original-V45-chunk-source-contract') == [_chunk_contract()], 'Original Native chunk source contract differs')
    packets = selected('reservation-lease-packets')
    _require(len(packets) == len(reservations), 'Each actual reservation needs its allocator packets')
    for packet, reservation in zip(packets, reservations, strict=True):
        owner, identity = reservation['owner'], reservation['identity']
        _require(packet['nonce'] == identity[1] and sorted(row['rank'] for row in packet['ranks']) == list(range(4)),
                 'Actual lease nonce/rank packet differs')
        for row in packet['ranks']:
            pages, others = row['reserved_pages'], row['live_pages']
            _require(row['request_index'] == owner and row['allocation_epoch'] == identity[2]
                and row['capacity'] == 32 and owner not in row['active'] and owner in row['pending']
                and 0 < len(row['active']) < 32 and row['eligible_greedy'] is True
                and row['prompt_length'] == row['committed_length'] == reservation['length']
                and row['prompt_sha256'] == report['tokenized_inputs'][owner]['token_sha256']
                and len(pages) == (reservation['length']+127)//128 and len(set(pages)) == len(pages)
                and not set(pages)&set(others) and all(type(p) is int and 0 <= p < 544 for p in (*pages, *others)),
                'Actual full-prompt reservation/allocator ownership differs')
    footprints, fences = selected('live-query-write-footprint-bound'), selected('original-tree-commit-evidence')
    _require(len(footprints) == len(fences) == len(commits), 'Actual footprint/original commit fence is absent')
    for footprint, evidence, commit in zip(footprints, fences, commits, strict=True):
        prefixes = footprint['protected_prefix_lengths']
        prefix = lambda owner:prefixes.get(owner, prefixes.get(str(owner)))
        begin = logical['begin_by_cycle'][commit['cycle_id']]
        _require(footprint['cycle_id'] == commit['cycle_id']
            and footprint['query_write_count'] == (commit['end']-commit['start'] if report['is_draft'] else 197)
            and footprint['runtime_KV_bytes_compared'] is False and prefix(commit['owner']) == commit['start']
            and len(prefixes) == len(commit['target_ids'])+1
            and all(prefix(owner) == context for owner, context in zip(commit['target_ids'], begin['contexts'], strict=True)),
            'Actual live write footprint/protected prefixes differ')
        ranks = evidence['ranks']
        _require(type(ranks) is list and len(ranks) == 4 and all(row == ranks[0] for row in ranks)
            and ranks[0]['cycle_id'] == commit['cycle_id'] and ranks[0]['structural_failed'] == 0
            and ranks[0]['numerical_failed'] == 0 and ranks[0]['finished_home'] in (0, 1)
            and ranks[0]['next_home'] == 1-ranks[0]['finished_home']
            and ranks[0]['local_compaction_stream_fenced'] is True, 'Actual original tree fence/consensus differs')
    ledgers = selected('actual-prompt-draft-materialization-ledger')
    _require(len(ledgers) == (len(commits) if report['is_draft'] else 0), 'Actual draft ledger role/count differs')
    for ledger, commit in zip(ledgers, commits):
        _require(ledger['owner'] == commit['owner'] and ledger['cycle_id'] == commit['cycle_id']
            and ledger['after'] == dict(safe_before=commit['end'], tokens={})
            and ledger['sampled_completion_root_claimed'] is False, 'Actual draft ledger overclaims evaluated tokens')
    return generic


def _audit_reference(reference, tokens, results):
    _require(type(reference) is list and len(reference) == 48, 'Actual48 independent AR rows are required')
    for i, (row, tokenized, candidate) in enumerate(zip(reference, tokens, results, strict=True)):
        output, calls = row['completion_token_ids'], row['calls']
        _require(row['request_index'] == i and row['request_id'] == candidate['request_id']
            and row['prompt_sha256'] == tokenized['token_sha256'] and row['max_tokens'] == 256
            and row['execution'] == 'direct Native eager PA; fresh allocation; own previous predictions'
            and type(output) is list and len(output) == 256
            and all(type(value) is int and value >= 0 for value in output)
            and output == candidate['completion_token_ids'] and type(calls) is list and len(calls) == 256,
            'Independent eager AR differs from actual candidate outputs/request controls')
        length = tokenized['token_count']
        for step, call in enumerate(calls):
            _require(call['step'] == step and call['query_count'] == (length if step == 0 else 1)
                and call['start'] == (0 if step == 0 else length+step-1)
                and call['end'] == (length if step == 0 else length+step)
                and call['input_sha256'] == (tokenized['token_sha256'] if step == 0 else _digest([output[step-1]]))
                and call['hidden_logits_and_Native_vote_finite'] is True,
                'Independent AR did not use its own previous prediction or actual finite boundary')


def audit_live_poisson_reports(reports):
    """Pure HOST validation; only actual visited geometry/output is described."""
    from local_mixed_qualification import native_delta
    from startup_receipts_v75 import FAILURE_COUNTERS
    source = require_live_poisson_source()
    rows = original._manifest_rows(original.MANIFEST.read_bytes())
    _require(type(reports) is list and len(reports) == 4
        and all(type(row) is dict and type(row.get('rank')) is int for row in reports)
        and sorted(row['rank'] for row in reports) == list(range(4)), 'Four distinct actual raw live reports are required')
    reports = sorted(reports, key=lambda row:row['rank'])
    first = reports[0]
    leader, targets, drafts = first['target_leader_rank'], first['target_ranks'], first['draft_ranks']
    _require(type(leader) is int and type(targets) is list and type(drafts) is list
        and len(targets) == 3 and len(drafts) == 1 and leader in targets
        and sorted(targets+drafts) == list(range(4)), 'Actual original topology differs')
    common = ('source', 'allowed_partitions', 'tokenized_inputs', 'sampling', 'absolute_arrivals',
        'logical_events', 'begin_inputs', 'standalone_prefill_calls', 'observed_control_counts', 'close_abort_identities')
    logical = None
    for report in reports:
        _require(report['schema'] == SCHEMA and report['status'] == 'raw-complete'
            and report['source'] == source and report['allowed_partitions'] == _scope(PARTITIONS)
            and report['target_leader_rank'] == leader and report['target_ranks'] == targets and report['draft_ranks'] == drafts
            and type(report['is_draft']) is bool and report['is_draft'] == (report['rank'] in drafts)
            and all(report[key] is False for key in ('numerical_qualification_inferred', 'all65_qualified',
                'full463_qualified', 'online_qualified', 'cost_or_Goodput', 'unreleased_dispositions_remaining'))
            and all(report[key] is True for key in ('wrappers_restored', 'class_delegate_restored',
                'actual_live_delegate_verified', 'candidate_and_reference_cache_released'))
            and report['observation_errors'] == [] and report['decode_entry_count'] == 1
            and all(report[key] == first[key] for key in common), 'Live source/scope/all4 trace/cleanup differs')
        origin = report['arrival_origin']
        _require(type(origin) is dict and origin['entry_barriers'] == origin['origin_broadcasts'] == 1
            and all(_finite(origin[key]) for key in ('leader_origin', 'local_before_broadcast', 'local_after_broadcast'))
            and origin['local_before_broadcast'] <= origin['local_after_broadcast']
            and origin['leader_origin'] == first['arrival_origin']['leader_origin']
            and report['absolute_arrivals'] == [origin['leader_origin']+row['arrival_offset_sec'] for row in rows],
            'Actual LIVE decode-entry arrival origin/offsets differ')
        if report['rank'] == leader:
            _require(origin['local_before_broadcast'] <= origin['leader_origin'] <= origin['local_after_broadcast'],
                     'Leader origin was not sampled at actual LIVE entry')
        delta = native_delta(report['native_before'], report['native_after'])
        _require(report['native_delta'] == delta and all(delta[key] == 0 for key in FAILURE_COUNTERS),
                 'Actual Native finite/validation/graph failure or fallback occurred')
        terminal = report['terminal_evidence']
        _require(terminal['reservation_remaining'] is False and terminal['committed_cursor'] is None
            and terminal['prepared_cursor'] is None and terminal['qualification_not_inferred_from_runtime_state'] is True,
            'Actual live transaction was not closed')
        current_logical = _audit_logical(report)
        if logical is None:
            logical = current_logical
        _audit_boundaries(report, current_logical)
        expected_control = {kind:sum(event['kind'] == kind for event in report['logical_events'])
            for kind in ('fallback-request', 'fallback-reset', 'abort', 'submit-error')}
        _require(report['observed_control_counts'] == expected_control and expected_control['submit-error'] == 0,
                 'Actual fallback/abort/submission accounting differs')
        counters = report['actual_decode_counters']
        _require(type(counters) is dict and all(type(counters.get(key)) is int and counters[key] >= 0
            and counters[key] == first['actual_decode_counters'][key] for key in original.LOGICAL_COUNTERS)
            and counters['spec_rhythm_tree_rounds'] > 0 and 1 <= counters['spec_rhythm_peak_active_requests'] <= 32
            and counters['spec_rhythm_live_admitted_requests'] == counters['spec_rhythm_aborted_requests'] == 0
            and counters.get('spec_rhythm_target_only_fallback_rounds', 0) == 0,
            'Original loop logical counters differ or include unobserved target-only fallback')
        if report['is_draft']:
            _require(report['normal_target_calls'] == [] and report['eager_AR_reference'] is None,
                     'Draft cannot claim target-only results')
        else:
            _require(type(report['normal_target_calls']) is list, 'Original normal-target trace is missing')
            if report['normal_target_calls']:
                original._audit_target_calls(report['normal_target_calls'])
        if report['rank'] != leader:
            _require(report['generate_results'] is None and report['stream_events'] == [], 'Only original leader owns client output')
    trace = reports[targets[0]]['normal_target_calls']
    _require(all(reports[rank]['normal_target_calls'] == trace for rank in targets), 'Normal target TP traces differ')
    tokens = first['tokenized_inputs']
    _require(type(tokens) is list and len(tokens) == 48 and first['sampling'] == original.sampling_specifications(rows)
        and all(row['request_index'] == i and type(row['token_count']) is int and row['token_count'] >= 2
            and type(row['token_sha256']) is str and len(row['token_sha256']) == 64 for i, row in enumerate(tokens)),
        'Actual original48 tokenized input/sampling projection differs')
    commits, publications = logical['commits'], logical['publications']
    q = sum(row['real_query_count'] for row in trace)+sum(sum(row['budgets'])+len(row['budgets']) for row in commits)
    b = sum(sum(row['budgets']) for row in trace)+sum(sum(row['budgets']) for row in commits)
    for report in reports:
        before, after = report['policy_before'], report['policy_after']
        _require(type(before) is dict and type(after) is dict and set(before) == set(after) == set(original.POLICY_COUNTERS)
            and all(type(value) is int and value >= 0 for snapshot in (before, after) for value in snapshot.values())
            and after[original.POLICY_COUNTERS[0]]-before[original.POLICY_COUNTERS[0]] == len(trace)+len(commits)
            and after[original.POLICY_COUNTERS[1]]-before[original.POLICY_COUNTERS[1]] == q
            and report['actual_decode_counters']['spec_rhythm_verified_tokens'] == b,
            'Original policy accounting must include every mixed call exactly once')
        if not report['is_draft']:
            counters = report['actual_decode_counters']
            _require(counters.get('spec_rhythm_target_query_tokens') == sum(row['physical_M'] for row in trace)
                +sum(sum(row['budgets'])+len(row['budgets']) for row in commits)
                and counters.get('spec_rhythm_target_graph_padding_queries') == sum(
                    row['physical_M']-row['real_query_count'] for row in trace),
                'Native normal/mixed query accounting changed its original meaning')
    standalone = []
    for call in first['standalone_prefill_calls']:
        ids = call['sequence_ids']
        _require(type(ids) is list and ids and all(type(i) is int and 0 <= i < 48 for i in ids)
            and call['cached_tokens'] == [0]*len(ids)
            and call['prompt_lengths'] == [tokens[i]['token_count'] for i in ids]
            and call['prompt_sha256'] == [tokens[i]['token_sha256'] for i in ids]
            and call['uncached_queries'] == sum(tokens[i]['token_count'] for i in ids)
            and call['precomputed_target_tokens'] is False and call['draft_prefill_completed'] is False,
            'Fallback/original standalone prefill must recompute its full genuine prompt')
        standalone.extend(ids)
    published = [row['owner'] for row in publications]
    _require(sorted(standalone+published) == list(range(48))
        and all(row['owner'] in standalone for row in logical['aborted'])
        and first['actual_decode_counters']['spec_rhythm_prefill_requests'] == 48
        and first['actual_decode_counters']['spec_rhythm_prefill_batches'] ==
            len(first['standalone_prefill_calls'])+len(publications), 'Every request needs exactly one actual first-token path')
    owner = reports[leader]
    results = owner['generate_results']
    _require(type(results) is list and len(results) == 48, 'Actual48 original client result rows are required')
    delivered, finished, previous = [[] for _ in rows], [0]*48, -1.
    _require(type(owner['stream_events']) is list and bool(owner['stream_events']), 'Actual client stream is missing')
    for event in owner['stream_events']:
        i, output = event['request_index'], event['token_ids']
        _require(type(i) is int and 0 <= i < 48 and event['request_id'] == rows[i]['request_id']
            and type(output) is list and output and all(type(t) is int and t >= 0 for t in output)
            and type(event['finished']) is bool and _finite(event['elapsed_seconds'])
            and event['elapsed_seconds'] >= previous and finished[i] == 0,
            'Actual client publication is malformed/repeated/out of order')
        previous = event['elapsed_seconds']
        delivered[i].extend(output)
        finished[i] += int(event['finished'])
    for i, (result, row) in enumerate(zip(results, rows, strict=True)):
        _require(type(result['completion_token_ids']) is list and len(result['completion_token_ids']) == 256
            and delivered[i] == result['completion_token_ids'] and finished[i] == 1
            and result['request_id'] == row['request_id'] and result['finish_reason'] == 'length'
            and result['arrival_ts'] == first['absolute_arrivals'][i] and result['max_tokens'] == 256
            and result['ignore_eos'] is True and result['temperature'] == result['draft_temperature'] == 0.
            and result['slo_class'] == row['slo_class'] and result['slo_tpot_ms'] == row['slo_tpot_ms']
            and result['tree_mode'] is True, 'Actual48 output/arrival/policy/stream contract differs')
    _require(all(delivered[row['owner']][0] == row['first_token'] for row in publications), 'Mixed publication is not its real first client token')
    reference = reports[targets[0]]['eager_AR_reference']
    _require(all(reports[rank]['eager_AR_reference'] == reference for rank in targets), 'Independent AR target TP outputs/traces differ')
    _audit_reference(reference, tokens, results)
    histogram = Counter((len(row['budgets']), tuple(row['budgets'])) for row in commits)
    return dict(schema=SCHEMA, scope='actual visited live first48 with exact independent eager AR',
        request_count=48, client_output_tokens=12288, independent_eager_AR_exact=True,
        mixed_committed_chunks=len(commits), mixed_publications=len(publications),
        actual_visited_partitions=[dict(N=n, budgets=list(b), chunks=count) for (n, b), count in sorted(histogram.items())],
        visited_N16=any(n == 16 for n, _ in histogram), observed_control_counts=first['observed_control_counts'],
        numerical_qualification_inferred=False, all65_qualified=False, full463_qualified=False,
        online_qualified=False, cost_or_Goodput=False)
