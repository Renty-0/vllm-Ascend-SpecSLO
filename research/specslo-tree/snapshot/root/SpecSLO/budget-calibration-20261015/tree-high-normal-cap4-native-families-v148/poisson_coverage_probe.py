"""Unscored first48 Poisson coverage of the ORIGINAL V63 tree loop.

Import/source checks/audit are host-only. Execution is an explicit separate
diagnostic action on an existing Native engine. Observers never inspect KV or
device values, add a per-cycle collective, install a mixed loop, or score time.
"""
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time
from types import CodeType, FunctionType


P = Path(__file__).resolve().parent
MANIFEST = Path('/root/data/specslo-baselines/nano-pearl-ascend-v1/runtime/manifest.jsonl')
MANIFEST_SHA256 = 'dc58dc5af1c5b28c6944f8f3faadeaae2dae34298c07b3ac84745d367c67d662'
PREFIX_SHA256 = '0b9e23b2543382a50c8ecd96389a28ceea479f73980f619bc9f497db6b04fd5b'
RUNTIME_SHA256 = '89303eadf7753d9a0482a24d6fa0613c482b5a50a8c979a3bc1aec5ef3e4095c'
SCHEMA = 'V83.original-Poisson-prefix-coverage.v1'
REQUEST_COUNT = 48
OUTPUT_TOKENS = 256
POLICY_COUNTERS = ('dynamic_tree_target_calls', 'dynamic_tree_target_actual_queries')
LOGICAL_COUNTERS = ('spec_rhythm_tree_rounds', 'spec_rhythm_tree_nodes',
    'spec_rhythm_tree_accepted_nodes', 'spec_rhythm_verified_tokens',
    'spec_rhythm_tree_full_accepts', 'spec_rhythm_tree_rejections',
    'spec_rhythm_prefill_batches', 'spec_rhythm_prefill_requests',
    'spec_rhythm_peak_active_requests', 'spec_rhythm_live_admitted_requests',
    'spec_rhythm_aborted_requests')
_CODE_CACHE = {}


class CoverageRejected(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise CoverageRejected(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(tokens):
    return _sha(json.dumps(list(tokens), separators=(',', ':')).encode())


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _manifest_rows(raw):
    _require(_sha(raw) == MANIFEST_SHA256, 'Pinned full463 manifest bytes differ')
    lines = raw.splitlines(keepends=True)
    _require(len(lines) == 463 and _sha(b''.join(lines[:REQUEST_COUNT])) == PREFIX_SHA256,
             'Pinned original48 prefix bytes differ')
    rows = [json.loads(line) for line in lines[:REQUEST_COUNT]]
    deadlines = {'tight': 40., 'normal': 50., 'loose': 150.}
    _require(len({row['request_id'] for row in rows}) == REQUEST_COUNT
        and all(type(row['prompt']) is str and row['prompt']
            and _finite(row['arrival_offset_sec']) and row['arrival_offset_sec'] >= 0
            and row['max_tokens'] == OUTPUT_TOKENS and row['per_request_gamma'] == 4
            and row['slo_class'] in deadlines
            and row['slo_tpot_ms'] == deadlines[row['slo_class']] for row in rows)
        and [row['arrival_offset_sec'] for row in rows] == sorted(row['arrival_offset_sec'] for row in rows),
        'Original prefix IDs/offsets/class/SLO/output contract differs')
    return rows


def _projection(rows):
    return [dict(request_index=i, request_id=row['request_id'],
        arrival_offset_sec=row['arrival_offset_sec'], prompt_sha256=_sha(row['prompt'].encode()),
        slo_class=row['slo_class'], slo_tpot_ms=row['slo_tpot_ms'],
        max_tokens=row['max_tokens'], manifest_gamma=row['per_request_gamma'], candidate_gamma=5)
        for i, row in enumerate(rows)]


def _find_code(code, name):
    found = []
    def visit(node):
        for value in node.co_consts:
            if isinstance(value, CodeType):
                if value.co_name == name:
                    found.append(value)
                visit(value)
    visit(code)
    _require(len(found) == 1, 'Original source function is not unique: '+name)
    return found[0]


def _expected_code(path, name):
    raw = path.read_bytes()  # Verify exact bytes before every cache lookup.
    key = (str(path), _sha(raw), name)
    if key not in _CODE_CACHE:
        _CODE_CACHE[key] = _find_code(compile(raw, str(path), 'exec'), name)
    return _CODE_CACHE[key]


def require_poisson_coverage_source():
    """Validate source/input only; this grants no device/online capability."""
    rows = _manifest_rows(MANIFEST.read_bytes())
    sources = {}
    for name in ('runtime_tree_decode.py', 'v63_runtime_reference.py'):
        sources[name] = _sha((P/name).read_bytes())
        _require(sources[name] == RUNTIME_SHA256, 'Original V63 decode bytes differ: '+name)
    _require(json.loads((P/'RUNTIME_SOURCE_PIN.json').read_text()) == {'sha256': RUNTIME_SHA256},
             'Original runtime source pin differs')
    provenance = json.loads((P/'PARENT_PROVENANCE_V75.json').read_text())
    native = provenance['native_contract_sources']
    _require(type(native) is dict and len(native) == 4, 'Native source closure is missing')
    for path, wanted in native.items():
        _require(_sha(Path(path).read_bytes()) == wanted, 'Pinned Native source bytes differ: '+path)
    for name in ('tree_entry.py', 'target_layout.py', 'BOUNDED_BUCKET_WHITELIST.json'):
        sources[name] = _sha((P/name).read_bytes())
    _expected_code(P/'runtime_tree_decode.py', '_generate_spec_rhythm_tree_decode')
    _expected_code(P/'tree_entry.py', 'fresh_decode')
    return dict(schema=SCHEMA, manifest_path=str(MANIFEST), manifest_sha256=MANIFEST_SHA256,
        prefix_sha256=PREFIX_SHA256, prefix_count=REQUEST_COUNT,
        source_sha256=sources, native_source_sha256=native, requests=_projection(rows),
        numerical_qualification_inferred=False, new_mixed_qualified=False,
        online_qualified=False, cost_or_Goodput=False)


def sampling_specifications(rows):
    """Keep frozen per-class policy; explicit B5 is original tree candidate cap."""
    return [dict(temperature=0., draft_temperature=0., top_p=1., top_k=0,
        draft_top_p=1., draft_top_k=0, max_tokens=OUTPUT_TOKENS, ignore_eos=True,
        slo_class=row['slo_class'], slo_tpot_ms=row['slo_tpot_ms'],
        request_id=row['request_id'], spec_rhythm_max_gamma=5, arrival_ts=None) for row in rows]


def tokenize_prefix(tokenizer, rows):
    # Exact existing benchmark formatting AND encode defaults; no new tokenizer.
    prompts = [list(tokenizer.encode(tokenizer.apply_chat_template(
        [{'role': 'user', 'content': row['prompt']}], tokenize=False, add_generation_prompt=True)))
        for row in rows]
    _require(all(row and all(type(token) is int and token >= 0 for token in row) for row in prompts),
             'Existing tokenizer produced an empty/invalid prompt')
    return prompts, [dict(request_index=i, token_count=len(row), token_sha256=_digest(row))
                     for i, row in enumerate(prompts)]


def _assert_original_decode(bound):
    shell = getattr(bound, '__func__', None)
    _require(isinstance(shell, FunctionType)
        and shell.__code__ == _expected_code(P/'tree_entry.py', 'fresh_decode')
        and Path(shell.__globals__.get('P', '')).resolve() == P,
        'Coverage requires the unchanged original fresh_decode shell')
    closure = dict(zip(shell.__code__.co_freevars, shell.__closure__ or ()))
    _require('decode_impl' in closure, 'Original decode delegate is absent')
    delegate = closure['decode_impl'].cell_contents
    # Native source executes the original @torch.inference_mode decorator.
    # Accept only its actual contextlib wrapper, never an arbitrary __wrapped__.
    while isinstance(delegate, FunctionType) and hasattr(delegate, '__wrapped__'):
        cells = dict(zip(delegate.__code__.co_freevars, delegate.__closure__ or ()))
        _require(delegate.__code__.co_name == 'decorate_context'
            and Path(delegate.__code__.co_filename).as_posix().endswith('/torch/utils/_contextlib.py')
            and 'func' in cells and cells['func'].cell_contents is delegate.__wrapped__,
            'Original decode decorator was replaced')
        delegate = delegate.__wrapped__
    _require(isinstance(delegate, FunctionType)
        and delegate.__code__ == _expected_code(P/'runtime_tree_decode.py', '_generate_spec_rhythm_tree_decode'),
        'Prior live/mixed/replaced decode delegate is forbidden for coverage')


def _entry_origin(engine):
    import torch.distributed as dist
    group = engine.groups.verification_coordination_group
    _require(group is not None and dist.get_world_size(group) == 4
        and str(dist.get_backend(group)) == 'gloo', 'Original four-rank Gloo entry group is required')
    dist.barrier(group=group)
    before = time.time()
    origin = time.time() if engine.rank == engine.topology.target_leader_rank else None
    packet = [origin]
    dist.broadcast_object_list(packet, src=engine.topology.target_leader_rank, group=group)
    after = time.time()
    _require(_finite(packet[0]), 'Leader arrival origin is not finite')
    return dict(leader_origin=packet[0], local_before_broadcast=before,
                local_after_broadcast=after, entry_barriers=1, origin_broadcasts=1)


def _target_record(plans, args, kwargs):
    n = kwargs.get('real_tree_count') or len(plans)
    _require(type(n) is int and 1 <= n <= len(plans), 'Actual real target count is invalid')
    budgets = [plan.candidate_budget for plan in plans[:n]]
    q = [plan.parent_indices.numel()+1 for plan in plans]
    ids = kwargs.get('sequence_ids', args[2] if len(args) > 2 else None)
    _require(ids is not None and len(ids) == len(plans), 'Actual target owner IDs are absent')
    _require(all(type(x) is int for x in budgets+q+list(ids))
        and q[:n] == [b+1 for b in budgets], 'Actual target host geometry is inconsistent')
    return dict(N=n, budgets=budgets, contexts=[plan.prefix_len for plan in plans[:n]],
        real_query_count=sum(q[:n]), physical_S=len(plans), physical_M=sum(q),
        physical_query_partition=q, sequence_ids=list(ids),
        family='causal' if max(budgets) <= 4 else 'full9')


class OriginalObservation:
    """Instance observers preserve delegate arguments/results and __func__."""
    def __init__(self, engine, rows, *, origin_provider=_entry_origin):
        self.engine, self.rows, self.origin_provider = engine, rows, origin_provider
        self.target_calls, self.prefill_calls, self.errors = [], [], []
        self.entry_calls, self.origin, self.arrivals = 0, None, []
        self.originals, self.wrappers, self.previous = {}, {}, {}
        self.restored = False

    def _observe(self, callback):
        # Instrumentation must not skip the original operation on one role.
        try:
            callback()
        except Exception as error:
            self.errors.append(type(error).__name__+': '+str(error))

    def install(self):
        engine = self.engine
        _assert_original_decode(engine._generate_spec_rhythm_tree_decode)
        names = ('target_tree_forward', '_prefill_and_sample_target_batch',
                 '_generate_spec_rhythm_tree_decode')
        self.originals = {name: getattr(engine, name) for name in names}
        self.previous = {name: engine.__dict__.get(name, _ABSENT) for name in names}
        def target(plans, *args, **kwargs):
            if not engine.is_draft:
                self._observe(lambda:self.target_calls.append(_target_record(plans, args, kwargs)))
            return self.originals['target_tree_forward'](plans, *args, **kwargs)
        def prefill(prompts, states, *args, **kwargs):
            def record():
                ids = kwargs.get('sequence_ids', args[0] if args else None)
                ids = list(range(len(prompts))) if ids is None else list(ids)
                cached = [engine.cache_allocation.num_cached_tokens[i] for i in ids]
                self.prefill_calls.append(dict(sequence_ids=ids,
                    prompt_lengths=[len(prompt) for prompt in prompts], cached_tokens=cached,
                    uncached_queries=sum(len(prompt)-c for prompt, c in zip(prompts, cached, strict=True)),
                    prompt_sha256=[_digest(prompt) for prompt in prompts],
                    coordinate_before_broadcast=kwargs.get('coordinate_before_broadcast', False),
                    precomputed_target_tokens=kwargs.get('precomputed_target_tokens') is not None,
                    draft_prefill_completed=kwargs.get('draft_prefill_completed', False)))
            self._observe(record)
            return self.originals['_prefill_and_sample_target_batch'](prompts, states, *args, **kwargs)
        def decode(*args, **kwargs):
            self.entry_calls += 1
            _require(self.entry_calls == 1 and not args, 'Exactly one original keyword decode entry is required')
            params = kwargs['request_params']
            _require(len(params) == len(self.rows) == REQUEST_COUNT
                and len(kwargs['draft_states']) == len(params) == len(kwargs['target_states']),
                'Original decode input request/state count differs')
            self.origin = self.origin_provider(engine)
            origin = self.origin['leader_origin']
            changed = [replace(param, arrival_ts=origin+row['arrival_offset_sec'])
                       for param, row in zip(params, self.rows, strict=True)]
            self.arrivals = [param.arrival_ts for param in changed]
            for states in (kwargs['draft_states'], kwargs['target_states']):
                for state, arrival in zip(states, self.arrivals, strict=True):
                    state.arrival_ts = arrival
            forwarded = dict(kwargs, request_params=changed)
            return self.originals['_generate_spec_rhythm_tree_decode'](*args, **forwarded)
        self.wrappers = dict(zip(names, (target, prefill, decode)))
        for name, wrapper in self.wrappers.items():
            # ORIGINAL loop uses this identity for its Native fast-path choice.
            wrapper.__func__ = getattr(self.originals[name], '__func__', None)
            setattr(engine, name, wrapper)
        return self

    def detach(self):
        ownership_errors = []
        for name, wrapper in self.wrappers.items():
            if engine_value(self.engine, name) is not wrapper:
                ownership_errors.append(name)
                continue
            previous = self.previous[name]
            if previous is _ABSENT:
                delattr(self.engine, name)
            else:
                setattr(self.engine, name, previous)
        self.restored = not ownership_errors
        _require(not ownership_errors, 'Observer wrapper ownership changed: '+repr(ownership_errors))


_ABSENT = object()


def engine_value(engine, name):
    return engine.__dict__.get(name, _ABSENT)


def _policy_snapshot(engine):
    from budget_policy import TreeBudgetPolicy
    function = getattr(engine.graph_metrics, '__func__', None)
    _require(isinstance(function, FunctionType)
        and function.__code__ == _expected_code(P/'tree_entry.py', 'measured_metrics')
        and Path(function.__globals__.get('P', '')).resolve() == P,
        'Actual original measured_metrics binding changed')
    cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
    _require('policy' in cells, 'Actual metrics policy closure is missing')
    policy = cells['policy'].cell_contents
    _require(type(policy) is TreeBudgetPolicy and type(policy.counts) is Counter
        and all(type(key) is str and type(value) is int and value >= 0
                for key, value in policy.counts.items()),
        'Actual TreeBudgetPolicy Counter type/value changed')
    account = getattr(getattr(engine, '_specslo_account_real_target_plans', None), '__func__', None)
    _require(isinstance(account, FunctionType)
        and account.__code__ == _expected_code(P/'tree_entry.py', 'account_real_target_plans'),
        'Actual original target accounting binding changed')
    accounting_cells = dict(zip(account.__code__.co_freevars, account.__closure__ or ()))
    _require('policy' in accounting_cells and accounting_cells['policy'].cell_contents is policy,
             'Metrics and actual target accounting do not share their original policy')
    # Counter.__missing__ genuinely returns zero without inserting a key. Native
    # measured_metrics publishes only materialized items, notably none before
    # this draft worker's first actual target call. No Native metric is defaulted.
    snapshot = {key:policy.counts[key.removeprefix('dynamic_tree_')] for key in POLICY_COUNTERS}
    metrics = engine.graph_metrics()
    _require(type(metrics) is dict and all(key not in metrics or
        type(metrics[key]) is int and metrics[key] == value for key, value in snapshot.items()),
        'Published Native policy metrics disagree with the actual Counter')
    return snapshot


def run_poisson_coverage_probe(engine):
    """One explicitly permitted original invocation; no model construction."""
    from diagnostic_authorization import require_poisson_coverage_scope
    require_poisson_coverage_scope()
    from vllm_ascend.spec_decode.pearl import native_engine as native
    _require(isinstance(engine, native.NativePearlEngine) and engine.device.type == 'npu',
             'CPU stubs cannot execute or qualify actual Poisson coverage')
    from startup_receipts_v75 import stage4
    from local_mixed_qualification import native_snapshot, native_delta
    stage = lambda name, callback:stage4(engine, 'Poisson-coverage-'+name, callback)
    source = stage('source', require_poisson_coverage_source)
    rows = _manifest_rows(MANIFEST.read_bytes())
    def prepare():
        cfg = engine.config
        _require(cfg.enable_continuous_batching and cfg.enable_preemptive_scheduling
            and cfg.enable_spec_rhythm and cfg.spec_rhythm_online_prefill
            and not cfg.enable_prefix_caching and not cfg.spec_rhythm_kv_ready_arrivals
            and cfg.max_num_seqs == 32 and cfg.spec_rhythm_tree_width == 2
            and cfg.spec_rhythm_tree_depth == 4
            and engine.cache_allocation is None and engine.cache_block_tables is None,
            'Original capacity32/tree/online-prefill/fresh-cache configuration is required')
        prompts, tokens = tokenize_prefix(engine.tokenizer, rows)
        params = [native.NativeSamplingParams(**row) for row in sampling_specifications(rows)]
        return prompts, tokens, params
    prompts, tokens, params = stage('prepare', prepare)
    observation = OriginalObservation(engine, rows)
    stream = []
    try:
        stage('observe-install', observation.install)
        before = stage('native-before', lambda:native_snapshot(engine))
        policy_before = stage('policy-before', lambda:_policy_snapshot(engine))
        results = stage('original-generate', lambda:engine.generate_batch(prompts, params,
            max_rounds=None, on_token_commit=lambda event:stream.append(dict(event)),
            request_admission_callback=None, include_graph_metrics=False))
        after = stage('native-after', lambda:native_snapshot(engine))
        policy_after = stage('policy-after', lambda:_policy_snapshot(engine))
    finally:
        if observation.wrappers:
            observation.detach()
    return dict(schema=SCHEMA, rank=engine.rank, is_draft=engine.is_draft,
        target_leader_rank=engine.topology.target_leader_rank,
        draft_ranks=list(engine.topology.draft_ranks), target_ranks=list(engine.topology.target_ranks),
        source=source, tokenized_inputs=tokens, sampling=sampling_specifications(rows),
        decode_entry_count=observation.entry_calls, arrival_origin=observation.origin,
        absolute_arrivals=observation.arrivals, target_calls=observation.target_calls,
        prefill_calls=observation.prefill_calls, observation_errors=observation.errors,
        wrappers_restored=observation.restored, original_delegate_verified=True,
        native_before=before, native_after=after, native_delta=native_delta(before, after),
        policy_before=policy_before, policy_after=policy_after,
        actual_decode_counters=dict(engine.last_worker_decode_counters),
        generate_results=results, stream_events=stream,
        candidate_cache_released=engine.cache_allocation is None and engine.cache_block_tables is None,
        numerical_qualification_inferred=False, new_mixed_qualified=False,
        online_qualified=False, cost_or_Goodput=False, status='raw-complete')


def _audit_target_calls(calls):
    from target_layout import envelope
    _require(type(calls) is list and calls, 'Target-only actual shape trace is absent')
    for row in calls:
        n, budgets = row['N'], row['budgets']
        _require(type(n) is int and 1 <= n <= 32 and type(budgets) is list and len(budgets) == n
            and all(type(b) is int and 1 <= b <= 5 for b in budgets)
            and sum(budgets) <= 96 and sum(budgets)+n <= 128,
            'Original real N/B/query budget is invalid')
        family = 'causal' if max(budgets) <= 4 else 'full9'
        m, s, padding = envelope(n, sum(budgets)+n, family)
        _require(row['family'] == family and row['real_query_count'] == sum(budgets)+n
            and row['physical_M'] == m and row['physical_S'] == s
            and row['physical_query_partition'] == [b+1 for b in budgets]+padding
            and type(row['sequence_ids']) is list and len(row['sequence_ids']) == s
            and all(type(i) is int and i >= 0 for i in row['sequence_ids'])
            and len(set(row['sequence_ids'])) == s
            and type(row['contexts']) is list and len(row['contexts']) == n
            and all(type(c) is int and c >= 1 for c in row['contexts']),
            'Observed original physical/context/owner geometry differs')


def audit_poisson_coverage_reports(reports):
    """Pure HOST audit of actual four-rank raw records; no NPU/AR/GP claim."""
    source = require_poisson_coverage_source()
    rows = _manifest_rows(MANIFEST.read_bytes())
    from local_mixed_qualification import native_delta
    from startup_receipts_v75 import FAILURE_COUNTERS
    _require(type(reports) is list and len(reports) == 4
        and all(type(row) is dict and type(row.get('rank')) is int for row in reports)
        and sorted(row['rank'] for row in reports) == list(range(4)), 'Four distinct raw ranks are required')
    ordered = sorted(reports, key=lambda row:row['rank'])
    first = ordered[0]
    leader, targets, drafts = first['target_leader_rank'], first['target_ranks'], first['draft_ranks']
    _require(type(leader) is int and type(targets) is list and type(drafts) is list
        and len(drafts) == 1 and len(targets) == 3 and leader in targets
        and sorted(targets+drafts) == list(range(4)), 'Original 1draft/3target topology differs')
    logical_keys = ('source', 'tokenized_inputs', 'sampling', 'absolute_arrivals', 'prefill_calls')
    for report in ordered:
        _require(report['schema'] == SCHEMA and report['status'] == 'raw-complete'
            and report['source'] == source and report['target_leader_rank'] == leader
            and report['target_ranks'] == targets and report['draft_ranks'] == drafts
            and type(report['is_draft']) is bool and report['is_draft'] == (report['rank'] in drafts)
            and all(report[key] is False for key in ('numerical_qualification_inferred',
                'new_mixed_qualified', 'online_qualified', 'cost_or_Goodput'))
            and all(report[key] is True for key in ('wrappers_restored',
                'original_delegate_verified', 'candidate_cache_released'))
            and report['observation_errors'] == [] and report['decode_entry_count'] == 1,
            'Source/scope/original invocation/cleanup evidence differs on rank '+str(report['rank']))
        _require(all(report[key] == first[key] for key in logical_keys), 'All4 input/prefill trace mismatch')
        origin = report['arrival_origin']
        _require(type(origin) is dict and origin['entry_barriers'] == 1 and origin['origin_broadcasts'] == 1
            and all(_finite(origin[key]) for key in ('leader_origin', 'local_before_broadcast', 'local_after_broadcast'))
            and origin['local_before_broadcast'] <= origin['local_after_broadcast']
            and origin['leader_origin'] == first['arrival_origin']['leader_origin'], 'Fresh leader origin contract differs')
        _require(report['absolute_arrivals'] == [origin['leader_origin']+row['arrival_offset_sec'] for row in rows],
                 'Absolute arrivals do not preserve the exact original offsets')
        if report['rank'] == leader:
            _require(origin['local_before_broadcast'] <= origin['leader_origin'] <= origin['local_after_broadcast'],
                     'Leader origin was not sampled at the actual decode entry')
        delta = native_delta(report['native_before'], report['native_after'])
        _require(report['native_delta'] == delta and all(delta[key] == 0 for key in FAILURE_COUNTERS),
                 'Native validation/fallback/failure occurred; raw discovery remains reported')
        counters = report['actual_decode_counters']
        _require(type(counters) is dict
            and all(type(counters.get(key)) is int and counters[key] >= 0 for key in LOGICAL_COUNTERS)
            and counters['spec_rhythm_tree_rounds'] > 0
            and counters.get('spec_rhythm_live_admitted_requests') == 0
            and counters.get('spec_rhythm_aborted_requests') == 0
            and 1 <= counters['spec_rhythm_peak_active_requests'] <= 32
            and counters.get('spec_rhythm_target_only_fallback_rounds', 0) == 0
            and all(counters[key] == first['actual_decode_counters'][key] for key in LOGICAL_COUNTERS),
            'Original loop counters indicate missing work/abort/changed admission')
        if report['is_draft']:
            _require(report['target_calls'] == [], 'Draft cannot claim target-only observations')
        else:
            _audit_target_calls(report['target_calls'])
        if report['rank'] != leader:
            _require(report['generate_results'] is None and report['stream_events'] == [],
                     'Only the original target leader owns client output')
    target_trace = ordered[targets[0]]['target_calls']
    _require(all(ordered[rank]['target_calls'] == target_trace for rank in targets), 'Target TP host traces differ')
    expected_queries = sum(row['real_query_count'] for row in target_trace)
    expected_verified = sum(sum(row['budgets']) for row in target_trace)
    for report in ordered:
        before, after = report['policy_before'], report['policy_after']
        _require(type(before) is dict and type(after) is dict
            and set(before) == set(after) == set(POLICY_COUNTERS)
            and all(type(value) is int and value >= 0 for snapshot in (before, after) for value in snapshot.values())
            and after[POLICY_COUNTERS[0]]-before[POLICY_COUNTERS[0]] == len(target_trace)
            and after[POLICY_COUNTERS[1]]-before[POLICY_COUNTERS[1]] == expected_queries
            and report['actual_decode_counters']['spec_rhythm_verified_tokens'] == expected_verified,
            'Original policy/verification accounting differs from observed actual tree work')
        if not report['is_draft']:
            counters = report['actual_decode_counters']
            _require(counters.get('spec_rhythm_target_query_tokens') == sum(row['physical_M'] for row in target_trace)
                and counters.get('spec_rhythm_target_graph_padding_queries') == sum(
                    row['physical_M']-row['real_query_count'] for row in target_trace),
                'Original Native physical-query/padding counters differ from the target trace')
    tokens = first['tokenized_inputs']
    _require(type(tokens) is list and len(tokens) == REQUEST_COUNT
        and all(row['request_index'] == i and type(row['token_count']) is int and row['token_count'] > 0
            and type(row['token_sha256']) is str and len(row['token_sha256']) == 64
            for i, row in enumerate(tokens)) and first['sampling'] == sampling_specifications(rows),
        'Exact tokenized input/sampling projection is missing')
    seen = []
    _require(type(first['prefill_calls']) is list and first['prefill_calls'], 'Original all4 prompt work was not visited')
    for call in first['prefill_calls']:
        ids = call['sequence_ids']
        _require(type(ids) is list and ids and all(type(i) is int and 0 <= i < REQUEST_COUNT for i in ids)
            and call['cached_tokens'] == [0]*len(ids)
            and call['prompt_lengths'] == [tokens[i]['token_count'] for i in ids]
            and call['prompt_sha256'] == [tokens[i]['token_sha256'] for i in ids]
            and call['uncached_queries'] == sum(tokens[i]['token_count'] for i in ids)
            and type(call['coordinate_before_broadcast']) is bool
            and call['precomputed_target_tokens'] is False and call['draft_prefill_completed'] is False,
            'Observed original full prompt work differs or uses mixed completion')
        seen.extend(ids)
    _require(sorted(seen) == list(range(REQUEST_COUNT)), 'Every original prompt must be prefetched exactly once')
    _require(first['actual_decode_counters']['spec_rhythm_prefill_requests'] == REQUEST_COUNT
        and first['actual_decode_counters']['spec_rhythm_prefill_batches'] == len(first['prefill_calls']),
        'Original Native prompt counters differ from the all4 prompt trace')
    owner = ordered[leader]
    results = owner['generate_results']
    _require(type(results) is list and len(results) == REQUEST_COUNT, '48 original client result rows required')
    delivered = [[] for _ in rows]
    finished = [0]*REQUEST_COUNT
    previous_elapsed = -1.
    _require(type(owner['stream_events']) is list and owner['stream_events'], 'Original client stream is absent')
    for event in owner['stream_events']:
        i, output = event['request_index'], event['token_ids']
        _require(type(i) is int and 0 <= i < REQUEST_COUNT and event['request_id'] == rows[i]['request_id']
            and type(output) is list and output and all(type(t) is int and t >= 0 for t in output)
            and type(event['finished']) is bool and _finite(event['elapsed_seconds'])
            and event['elapsed_seconds'] >= previous_elapsed and finished[i] == 0,
            'Malformed/duplicate/out-of-order original stream publication')
        previous_elapsed = event['elapsed_seconds']
        delivered[i].extend(output)
        finished[i] += int(event['finished'])
    for i, (result, original) in enumerate(zip(results, rows, strict=True)):
        output = result['completion_token_ids']
        _require(type(output) is list and len(output) == OUTPUT_TOKENS
            and all(type(token) is int and token >= 0 for token in output)
            and delivered[i] == output and finished[i] == 1
            and result['request_id'] == original['request_id'] and result['finish_reason'] == 'length'
            and result['arrival_ts'] == first['absolute_arrivals'][i]
            and result['max_tokens'] == OUTPUT_TOKENS and result['ignore_eos'] is True
            and result['temperature'] == 0. and result['draft_temperature'] == 0.
            and result['slo_class'] == original['slo_class'] and result['slo_tpot_ms'] == original['slo_tpot_ms']
            and result['tree_mode'] is True, 'Original output/clock/policy/stream contract differs')
    histogram = Counter((row['N'], tuple(row['budgets']), row['physical_M'], row['physical_S']) for row in target_trace)
    return dict(schema=SCHEMA, scope='unscored original-tree first48 coverage; not mixed qualification',
        request_count=REQUEST_COUNT, client_output_tokens=REQUEST_COUNT*OUTPUT_TOKENS,
        original_target_calls=len(target_trace), original_prefill_calls=len(first['prefill_calls']),
        observed_geometry=[dict(N=n, budgets=list(b), physical_M=m, physical_S=s, calls=count)
            for (n, b, m, s), count in sorted(histogram.items())],
        numerical_qualification_inferred=False, new_mixed_qualified=False,
        online_qualified=False, cost_or_Goodput=False)
