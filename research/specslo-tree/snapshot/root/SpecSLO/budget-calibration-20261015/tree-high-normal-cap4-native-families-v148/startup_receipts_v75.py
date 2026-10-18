"""Audit actual Native mappings against planned capacity64 draft families."""
import hashlib
import json
from pathlib import Path


P = Path(__file__).resolve().parent
MAPPINGS = ('entries', 'draft_entries', 'target_entries')
NATIVE_COUNTERS = (
    'aclgraph_captures', 'aclgraph_capture_attempts', 'aclgraph_runtime_validation_replays',
    'aclgraph_failed_captures', 'aclgraph_capacity_fallbacks', 'aclgraph_shape_fallbacks',
    *(f'aclgraph_{role}_{field}' for role in ('generic', 'draft', 'target')
      for field in ('eager_fallback_calls', 'disabled_entry_calls', 'unclassified_calls',
                    'runtime_validation_calls', 'runtime_validation_failures')),
)
FAILURE_COUNTERS = tuple(name for name in NATIVE_COUNTERS if any(
    word in name for word in ('failed_captures', 'fallback', 'disabled_entry',
                             'unclassified', 'validation_failures')))


def expected_prefill_keys(query_counts):
    """Exact original one-request causal hidden keys, never generic entries."""
    values = tuple(query_counts)
    if (len(values) != 201 or len(set(values)) != 201
            or any(type(n) is not int or not 1 <= n <= 1024 for n in values)):
        raise RuntimeError('Original exact-prefill inventory needs 201 distinct legal extents')
    attention = 'fia:exact-length-task-update|buffers:'+repr((((1, 32), (1, 32), (2048, 2048)),))
    return {(f'target-hidden:0|steps:1|{attention}', (n,)) for n in values}


def original_draft_family_keys():
    """Original Native qualifier at max64/width2/depth4: seven buckets, two masks, two heads."""
    from finite_inventory_plan import VOCAB
    keys = set()
    for bucket in (2, 4, 8, 16, 32, 64, 128):
        signatures = (
            'fia:exact-length-task-update|buffers:'
            +repr((((bucket, 32), (bucket, 32), (2048, 2048)),)),
            'tree-fia-full:exact-length-task-update|buffers:'
            +repr((((bucket, 1, 4*bucket+1, 4096), (bucket, 32)),)),
        )
        for attention in signatures:
            for output, vocab in (('logits', VOCAB), ('hidden', 0)):
                keys.add((f'target-{output}:{vocab}|steps:1|{attention}', (5*bucket,)))
    return keys


def _native_snapshot(engine):
    runner = engine.graph_runner
    mappings = [getattr(runner, name) for name in MAPPINGS]
    if any(not isinstance(mapping, dict) for mapping in mappings) or len({id(m) for m in mappings}) != 3:
        raise RuntimeError('Native generic/draft/target mappings must be distinct actual dictionaries')
    objects = [id(entry) for mapping in mappings for entry in mapping.values()]
    if len(set(objects)) != len(objects):
        raise RuntimeError('Native inventory aliases one entry under multiple keys or mappings')
    if runner.disabled_entry_keys:
        raise RuntimeError('Actual Native startup contains disabled graph keys')
    for name, mapping in zip(MAPPINGS, mappings):
        if any(getattr(entry, 'runtime_validated', None) is not True for entry in mapping.values()):
            raise RuntimeError('Actual Native startup has an unqualified entry in '+name)
    metrics = engine.graph_metrics()
    counters = {}
    for name in NATIVE_COUNTERS:
        value = metrics.get(name)
        if type(value) is not int or value < 0:
            raise RuntimeError('Actual nonnegative integer Native counter is missing: '+name)
        counters[name] = value
    if any(counters[name] for name in FAILURE_COUNTERS):
        raise RuntimeError('Actual Native startup reports capture/validation/fallback failure')
    return {'keys': {name: set(mapping) for name, mapping in zip(MAPPINGS, mappings)},
            'counts': {name: len(mapping) for name, mapping in zip(MAPPINGS, mappings)},
            'counters': counters}


def _explicit_draft_keys():
    from finite_inventory_plan import draft_keys
    families = draft_keys()
    return {'entries': set(), 'draft_entries': set(families['chain']),
            'target_entries': set(families['hidden_PA']) | set(families['frontier_PA'])}


def validate_native_inventory(engine, prefill_queries=(), *, original_draft_prepared=False):
    """Pure host audit of real mappings/flags/counters; no device qualification."""
    snapshot = _native_snapshot(engine)
    if engine.is_draft:
        expected = _explicit_draft_keys()
        if original_draft_prepared:
            expected['target_entries'] |= original_draft_family_keys()
    else:
        from finite_inventory_plan import targets
        queries = tuple(prefill_queries)
        if getattr(engine, '_dynamic_exact_prefill_qualified', None) != {(n, 1) for n in queries}:
            raise RuntimeError('Actual exact-prefill qualification set differs from all 201 original extents')
        decode = {row['expected_key'] for row in targets()}
        expected = {'entries': set(), 'draft_entries': set(),
                    'target_entries': expected_prefill_keys(queries)}
        # The target receipt stores repr(key); do not deserialize or trust its count.
        actual_decode = {key for key in snapshot['keys']['target_entries'] if repr(key) in decode}
        if set(map(repr, actual_decode)) != decode or len(decode) != 44:
            raise RuntimeError('Actual Native mapping lacks the exact 44 original decode keys')
        expected['target_entries'] |= actual_decode
    if snapshot['keys'] != expected:
        raise RuntimeError('Actual original Native key sets differ: '+repr(snapshot['counts']))
    return snapshot


def _run_original_draft_family(engine):
    if not engine.is_draft:
        return
    from vllm_ascend.spec_decode.pearl import native_engine as native
    provenance = json.loads((P/'PARENT_PROVENANCE_V75.json').read_text())
    path, wanted = next((path, sha) for path, sha in provenance['native_contract_sources'].items()
                        if path.endswith('/native_engine.py'))
    actual = Path(native.__file__).resolve()
    method = native.NativePearlEngine._qualify_tree_draft_graph_buckets
    if (actual != Path(path).resolve() or hashlib.sha256(actual.read_bytes()).hexdigest() != wanted
            or getattr(engine._qualify_tree_draft_graph_buckets, '__func__', None) is not method
            or Path(method.__code__.co_filename).resolve() != actual):
        raise RuntimeError('Original Native draft qualifier source/binding changed')
    engine._allocate_cache([], enable_prefix_caching=False, reserve_sequence_capacity=True)
    scratch = tuple(engine._tree_graph_scratch_sequence_ids)
    if len(scratch) != 64 or set(scratch) & set(range(64)):
        raise RuntimeError('Original draft qualifier lacks its 64 private scratch owners')
    for sequence in (*scratch, *range(64)):
        engine._activate_cache_sequence(sequence, [0], enable_prefix_caching=False)
    # No WORLD/CPU collective is entered by this original draft-only method.
    # Its unchanged implementation includes the actual final NPU synchronize.
    method(engine)


def prepare_original_draft_family(engine):
    """All four ranks call; only draft allocates/calls the original qualifier."""
    def before():
        runner = engine.graph_runner
        if (engine.cache_allocation is not None or engine.cache_block_tables is not None
                or runner.graph_cache_sealed or runner.max_graph_entries != 512
                or engine.config.max_num_seqs != 64
                or engine.config.spec_rhythm_tree_width != 2
                or engine.config.spec_rhythm_tree_depth != 4):
            raise RuntimeError('Original draft family requires released max64/width2/depth4 startup')
        if hasattr(engine, '_v80_original_draft_family'):
            raise RuntimeError('Original draft-family preparation cannot be repeated')
        if engine.is_draft:
            return validate_native_inventory(engine)
        return validate_native_inventory(engine, _prefill_queries())
    initial = stage4(engine, 'original-draft-family-before', before)
    try:
        # A reachable draft failure returns to all-rank stage4 before any next
        # stage or CPU rendezvous; target ranks never run the draft method.
        stage4(engine, 'original-draft-family-native-qualification', lambda: _run_original_draft_family(engine))
        def after():
            final = validate_native_inventory(engine, _prefill_queries(), original_draft_prepared=True)
            added = {name: final['keys'][name]-initial['keys'][name] for name in MAPPINGS}
            expected = {name: set() for name in MAPPINGS}
            if engine.is_draft:
                expected['target_entries'] = original_draft_family_keys()
            if added != expected or any(initial['keys'][name]-final['keys'][name] for name in MAPPINGS):
                raise RuntimeError('Original draft qualifier changed an unexpected actual Native key')
            delta = {name: final['counters'][name]-initial['counters'][name] for name in NATIVE_COUNTERS}
            count = 28 if engine.is_draft else 0
            if (any(value < 0 for value in delta.values())
                    or delta['aclgraph_captures'] != count
                    or delta['aclgraph_capture_attempts'] != count
                    or delta['aclgraph_runtime_validation_replays'] != count
                    or any(delta[name] for name in FAILURE_COUNTERS)
                    or (not engine.is_draft and any(delta.values()))):
                raise RuntimeError('Original draft-family actual Native discovery/validation counters differ')
            return {'rank': engine.rank, 'added_native_keys': {name: sorted(map(repr, keys))
                        for name, keys in added.items()},
                    'before_native_entry_counts': initial['counts'],
                    'actual_native_entry_counts': final['counts'],
                    'native_before': initial['counters'], 'native_after': final['counters'],
                    'native_delta': delta, 'actual_added_qualified_entries': count,
                    'original_native_method_executed': bool(engine.is_draft),
                    'native_qualification_flags_modified': False}
        evidence = stage4(engine, 'original-draft-family-after', after)
    finally:
        def release():
            if engine.is_draft:
                engine._release_cache()
            if engine.cache_allocation is not None or engine.cache_block_tables is not None:
                raise RuntimeError('Original draft-family actual allocation remained after release')
        stage4(engine, 'original-draft-family-cache-release', release)
    engine._v80_original_draft_family = evidence
    return evidence


def _prefill_queries():
    rows = json.loads((P/'prompt-shape-examples.json').read_text())['examples']
    return tuple(row['query_count'] for row in rows)


def stage4(engine, label, callback):
    import torch.distributed as dist
    group = engine.groups.verification_coordination_group
    value, error = None, None
    try:
        value = callback()
        # This qualification path is untimed. Finish any target TP projection
        # before peers enter the following CPU/Gloo agreement.
        import torch
        if engine.device.type != 'npu':
            raise RuntimeError('CPU contracts cannot attest an actual Native device fence')
        torch.npu.synchronize()
    except BaseException as exc:
        error = type(exc).__name__+': '+str(exc)
    rows = [None]*4
    dist.all_gather_object(rows, {'rank': engine.rank, 'label': label, 'error': error}, group=group)
    if (sorted(row['rank'] for row in rows) != list(range(4))
            or any(row['label'] != label or row['error'] is not None for row in rows)):
        raise RuntimeError('V75 allrank startup boundary failed '+label+': '+repr(rows))
    return value


def audit_actual_startup(engine):
    def inspect():
        runner = engine.graph_runner
        if (runner.max_graph_entries != 512 or engine.config.num_kvcache_blocks != 544
                or len(engine.model.layers) != (28 if engine.is_draft else 64)
                or runner.graph_cache_sealed):
            raise RuntimeError('Actual original startup geometry/layers/seal differs')
        original_family = getattr(engine, '_v80_original_draft_family', None)
        if (not isinstance(original_family, dict)
                or original_family.get('rank') != engine.rank
                or original_family.get('actual_added_qualified_entries') != (28 if engine.is_draft else 0)
                or original_family.get('original_native_method_executed') is not bool(engine.is_draft)
                or original_family.get('native_qualification_flags_modified') is not False):
            raise RuntimeError('Actual original draft-family preparation evidence is missing')
        inventory = validate_native_inventory(engine, _prefill_queries(), original_draft_prepared=True)
        if original_family.get('actual_native_entry_counts') != inventory['counts']:
            raise RuntimeError('Native startup inventory changed after original draft-family preparation')
        directory = P/'startup-progress'
        mc2_path = directory/f'MC2-preflight-rank{engine.rank}.json'
        mc2 = json.loads(mc2_path.read_text())
        if (mc2.get('all4rank_contract_passed') is not True or mc2.get('failed_checks') != []
                or mc2.get('native_qualification_flags_modified') is not False
                or len(mc2.get('planned_full_chain_rows', [])) != 22):
            raise RuntimeError('Actual original MC2 preflight incomplete')
        if not engine.is_draft:
            if (mc2.get('manifest_route_count') != 128 or mc2.get('disabled_routes') != []
                    or mc2.get('routes_with_missing_or_unqualified_rows') != []):
                raise RuntimeError('Actual original 128-route MC2 contract rejected')
            prefill = json.loads((directory/f'prefill-rank{engine.rank}.json').read_text())
            target = json.loads((directory/f'finite-target-rank{engine.rank}.json').read_text())
            if (prefill.get('status') != 'complete' or prefill.get('completed') != 201
                    or prefill.get('total') != 201 or prefill.get('transition_checks_passed') != 8
                    or target.get('status') != 'complete' or target.get('fixtures') != 44
                    or len(set(target.get('native_keys', []))) != 44):
                raise RuntimeError('Actual original 201+8/44 startup receipts incomplete')
            for text in target['native_keys']:
                matches = [entry for key, entry in runner.target_entries.items() if repr(key) == text]
                if len(matches) != 1 or not matches[0].runtime_validated:
                    raise RuntimeError('Original target receipt lacks successful actual Native entry')
            if set(target['native_keys']) != {repr(key) for key in inventory['keys']['target_entries']
                    if key not in expected_prefill_keys(_prefill_queries())}:
                raise RuntimeError('Original target receipt does not bind the exact disjoint 44 decode keys')
            details = {'prefill': prefill, 'target': target}
        else:
            draft = json.loads((directory/f'finite-draft-rank{engine.rank}.json').read_text())
            keys = [text for values in draft.get('native_keys', {}).values() for text in values]
            if (draft.get('status') != 'complete' or draft.get('explicit_keys') != 86
                    or len(keys) != 86 or len(set(keys)) != 86
                    or set(keys) != {repr(key) for values in _explicit_draft_keys().values() for key in values}):
                raise RuntimeError('Actual capacity64 draft 86 explicit-key receipt incomplete')
            actual = [(key, entry) for mapping in (runner.draft_entries, runner.target_entries)
                      for key, entry in mapping.items()]
            for text in keys:
                matches = [entry for key, entry in actual if repr(key) == text]
                if len(matches) != 1 or not matches[0].runtime_validated:
                    raise RuntimeError('Original draft receipt lacks successful actual Native entry')
            details = {'draft': draft}
        return {'rank': engine.rank, 'original_startup_receipts': details,
                'actual_native_entry_counts': inventory['counts'],
                'actual_native_keys': {name: sorted(map(repr, keys)) for name, keys in inventory['keys'].items()},
                'actual_native_counters': inventory['counters'],
                'original_draft_family': original_family,
                'actual_MC2_preflight': mc2, 'new_mixed197_qualified': False,
                'online_atomic_qualified': False}
    report = stage4(engine, 'actual-original-startup-receipts', inspect)
    from inventory_memory import checkpoint
    count = sum(report['actual_native_entry_counts'].values())
    report['actual_original_memory_finite_vote'] = checkpoint(engine, 'V80_original_startup', count, count)
    return report
