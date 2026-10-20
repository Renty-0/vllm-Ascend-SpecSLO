"""Independent immutable-token history actors for the BLOCKED local V78 S35 scope.

No import constructs a model.  Public execution actors require the actual local
bounded permit, the original four-rank coordinator, and an NPU engine.  Source
initialization is explicitly a fixture operation, never an online tree commit.
Target progressive calls are supplied by mixed_target_reference_v75; neither
candidate KV nor a dense intermediate prompt prefill can initialize a reference.
"""
from dataclasses import dataclass
import copy
import hashlib
import json

from diagnostic_authorization import require_bounded_scope
from local_qualification_scope import (
    SCOPES, SOURCE_INPUT_SHA256, fixture_for, PHYSICAL_M, PHYSICAL_S,
)


class HistoryRejected(RuntimeError):
    pass


_EPOCH_ISSUER = object()
_ISSUED_INITIAL_PREFIXES = {}
ADOPTION_KIND = 'independent-initial-fixture-prefix-not-online-tree-commit'


def _sha(tokens):
    return hashlib.sha256(json.dumps(list(tokens), separators=(',', ':')).encode()).hexdigest()


@dataclass(frozen=True)
class FrozenHistorySource:
    scope: str
    source_input_sha256: str
    tree_ids: tuple[str, ...]
    tree_tokens: tuple[tuple[int, ...], ...]
    prompt_tokens: tuple[int, ...]
    initial_cursor: int
    chunk_ends: tuple[int, ...]
    token_sha256: tuple[str, ...]


def immutable_source(fixture):
    """Check semantic identity against the pinned real provider, not its flags."""
    scope = fixture.get('scope')
    if scope not in SCOPES:
        raise HistoryRejected('Independent history has no declared local qualification scope')
    original = fixture_for(scope)
    for key in ('trees', 'anchors', 'prompt', 'initial_prompt_tokens', 'chunks', 'provenance'):
        if fixture.get(key) != original[key]:
            raise HistoryRejected('Immutable V45 source selection/tokens changed: '+key)
    trees = tuple(tuple(c['token_ids']) for c in original['trees'])
    prompt = tuple(original['prompt']['token_ids'])
    return FrozenHistorySource(scope.name, SOURCE_INPUT_SHA256,
        tuple(c['request_id'] for c in original['trees']), trees, prompt,
        scope.initial_prompt_cursor, tuple(scope.chunk_ends),
        tuple(_sha(t) for t in trees)+( _sha(prompt),))


@dataclass(frozen=True)
class InitialRankEvidence:
    rank: int
    role: str
    layer_count: int
    engine_epoch: int
    allocation_epoch: int
    cursor: int
    prompt_sha256: str
    source_input_sha256: str
    prefix_sha256: tuple[str, ...]
    all_prefix_finite: bool
    local_stream_fenced: bool
    pages: tuple[int, ...]


@dataclass(frozen=True)
class InitialPrefixEvidence:
    scope: str
    source_input_sha256: str
    prompt_sha256: str
    request_index: int
    nonce: str
    allocation_epoch: int
    initial_cursor: int
    prompt_length: int
    engine_epoch: int
    rank_records: tuple[InitialRankEvidence, ...]
    adoption_kind: str = ADOPTION_KIND


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_initial_prefix_evidence(evidence, identity, *, scope=None):
    """Validate a CPU certificate; this function does not attest model execution.

    The manager must additionally require the active local permit and match its
    RESERVED lease/physical pages.  Only initialize_private_prompt_cursor emits
    a certificate from this actor's live, allocation-bound epoch.
    """
    if not isinstance(evidence, InitialPrefixEvidence):
        raise HistoryRejected('Initial prefix requires typed immutable evidence')
    declared = next((s for s in SCOPES if s.name == evidence.scope), None)
    if (declared is None or (scope is not None and declared != scope)
            or any(type(v) is not int for v in (evidence.request_index,
                evidence.allocation_epoch, evidence.initial_cursor, evidence.prompt_length, evidence.engine_epoch))
            or evidence.source_input_sha256 != SOURCE_INPUT_SHA256
            or not _digest(evidence.prompt_sha256)
            or evidence.initial_cursor != declared.initial_prompt_cursor
            or evidence.prompt_length != 2301 or evidence.engine_epoch <= 0
            or evidence.adoption_kind != ADOPTION_KIND
            or evidence.request_index != identity.request_index
            or evidence.request_index != declared.target_requests
            or evidence.nonce != identity.nonce or not evidence.nonce
            or evidence.allocation_epoch != identity.allocation_epoch
            or type(evidence.allocation_epoch) is not int or evidence.allocation_epoch <= 0
            or not isinstance(evidence.rank_records, tuple) or len(evidence.rank_records) != 4):
        raise HistoryRejected('Initial prefix source/cursor/lease identity differs')
    records = evidence.rank_records
    if (any(not isinstance(r, InitialRankEvidence) for r in records)
            or sorted(r.rank for r in records) != list(range(4))
            or sum(r.role == 'draft' for r in records) != 1
            or sum(r.role == 'target' for r in records) != 3):
        raise HistoryRejected('Initial prefix lacks exactly one draft and three target ranks')
    for r in records:
        expected_layers = 28 if r.role == 'draft' else 64
        if (any(type(v) is not int for v in (r.rank, r.layer_count, r.engine_epoch,
                r.allocation_epoch, r.cursor))
                or r.layer_count != expected_layers or r.engine_epoch != evidence.engine_epoch
                or r.allocation_epoch != evidence.allocation_epoch
                or r.cursor != evidence.initial_cursor
                or r.prompt_sha256 != evidence.prompt_sha256
                or r.source_input_sha256 != evidence.source_input_sha256
                or r.all_prefix_finite is not True or r.local_stream_fenced is not True
                or not isinstance(r.prefix_sha256, tuple)
                or len(r.prefix_sha256) != 2*expected_layers
                or any(not _digest(v) for v in r.prefix_sha256)
                or not isinstance(r.pages, tuple) or len(r.pages) != 18
                or len(set(r.pages)) != len(r.pages)
                or any(type(p) is not int or not 0 <= p < 544 for p in r.pages)):
            raise HistoryRejected('A rank lacks real epoch/layers/finite/fence/full prompt pages')
    # BF16/TP shards and physical pages are role-local: do not compare their
    # hashes or physical IDs across ranks.
    return evidence


def _device_fence(engine):
    import torch
    if engine.device.type != 'npu':
        raise HistoryRejected('CPU contracts cannot certify actual Native stream completion')
    torch.npu.synchronize()


def _gather_all4(engine, value):
    import torch.distributed as dist
    group = engine.groups.verification_coordination_group
    if group is None or str(dist.get_backend(group)) != 'gloo' or dist.get_world_size(group) != 4:
        raise HistoryRejected('Independent actors require the original four-rank Gloo group')
    out = [None]*4
    dist.all_gather_object(out, value, group=group)
    return tuple(out)


def _arrays(engine):
    out = []
    for layer in engine.model.layers:
        for name in ('key_cache', 'value_cache'):
            raw = getattr(layer.self_attn, name)
            view = raw.flatten(0, 1)
            if view.data_ptr() != raw.data_ptr():
                raise HistoryRejected('Independent KV indexing must preserve actual cache storage')
            out.append(view)
    if len(out) != (56 if engine.is_draft else 128):
        raise HistoryRejected('Actual draft28/target64 layer cache count differs')
    return tuple(out)


def _slots(engine, owner, positions):
    import torch
    block_size = int(engine.config.kvcache_block_size)
    table = engine.cache_allocation.block_tables[owner]
    positions = tuple(positions)
    if (block_size != 128 or len(set(positions)) != len(positions)
            or any(type(p) is not int or not 0 <= p < 4096 for p in positions)):
        raise HistoryRejected('Independent logical KV position is invalid')
    values = [table[p//block_size]*block_size+p%block_size for p in positions]
    if (any(table[p//block_size] < 0 for p in positions)
            or any(not 0 <= v < 544*block_size for v in values)
            or len(set(values)) != len(values)):
        raise HistoryRejected('Independent logical KV references an unallocated or aliased slot')
    return torch.tensor(values, dtype=torch.long, device=engine.device)


def query_KV(engine, metadata):
    """Immediate independent clones of every real physical query row/layer."""
    import torch
    slots = metadata.slot_mapping.to(dtype=torch.long)
    if bool((slots < 0).any().item()):
        raise HistoryRejected('Full FIA queries cannot include PA read-only padding slots')
    return tuple(v.index_select(0, slots).clone() for v in _arrays(engine))


def prefix_evidence(engine, histories):
    """Local diagnostic bytes and finite checks; caller stages any rejection."""
    import torch
    hashes = {}
    failed = False
    arrays = _arrays(engine)
    for owner, tokens in sorted(histories.items()):
        index = _slots(engine, owner, range(len(tokens)))
        values = []
        for cache in arrays:
            value = cache.index_select(0, index)
            failed |= not bool(torch.isfinite(value).all().item())
            values.append(hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest())
        hashes[owner] = tuple(values)
    if failed:
        raise HistoryRejected('Independent used-prefix KV contains nonfinite values')
    return hashes


def _require_engine(engine):
    require_bounded_scope()
    if (engine.device.type != 'npu' or engine.config.enable_prefix_caching
            or int(engine.config.num_kvcache_blocks) != 544
            or int(engine.config.kvcache_block_size) != 128
            or int(engine.config.max_model_len) != 4096
            or len(engine.model.layers) != (28 if engine.is_draft else 64)):
        raise HistoryRejected('Independent actual engine/config/layer scope differs')


def _epoch(engine, epoch):
    _require_engine(engine)
    if (epoch.get('_issuer') is not _EPOCH_ISSUER
            or epoch.get('allocation_object') is not engine.cache_allocation
            or epoch.get('engine_epoch') != getattr(engine, '_v75_independent_engine_epoch', None)
            or epoch.get('failed') is not False):
        raise HistoryRejected('Independent epoch was replaced, invalidated, or not built by this actor')
    return epoch['immutable_source']


def _write_initial_prefix(engine, owner, tokens, *, stage_all4):
    # Exactly the immutable initial source, never an intermediate phase cursor
    # history which must have been generated by the first genuine mixed call.
    for start in range(0, len(tokens), 512):
        values = list(tokens[start:start+512])
        def forward(values=values, start=start):
            import torch
            hidden = engine._run_packed_hidden(values, [owner]*len(values),
                list(range(start, start+len(values))), use_aclgraph=False,
                use_fused_infer_attention=True)
            if not bool(torch.isfinite(hidden).all().item()):
                raise HistoryRejected('Independent initial prefix returned nonfinite hidden')
            # No hidden is retained across another graph or used as evidence of
            # a head which was not actually projected in this initial operation.
            return None
        stage_all4('independent-initial-prefix-'+str(owner)+'-'+str(start),
            forward)


def epoch_owner_partition(n, pool):
    """Pure owner contract, never allocator or numerical qualification."""
    if type(n) is not int or not 1<=n<=32:
        raise HistoryRejected('Pure owner partition requires an integer N in 1..32')
    PHYSICAL_S=19 if n<=16 else 35
    values = tuple(pool)
    d = PHYSICAL_S-n-1
    if (d != PHYSICAL_S-1-n or len(values) < d+PHYSICAL_S
            or any(type(v) is not int or v < 0 for v in values)
            or len(set(values)) != len(values)):
        raise HistoryRejected('Original private owner pool cannot represent the disjoint S35 epoch')
    pads = values[:d]
    refs = values[d:d+PHYSICAL_S]
    ids = tuple(range(n+1))+pads
    if (len(ids) != PHYSICAL_S or len(refs) != PHYSICAL_S
            or len(set(ids+refs)) != 2*PHYSICAL_S):
        raise HistoryRejected('Actual and independent 35-span owners alias')
    return dict(actual_padding_ids=pads, reference_padding_ids=refs[n+1:],
        candidate_span_ids=list(ids), reference_span_ids=list(refs),
        reference_tree_ids=list(refs[:n]), reference_prompt_id=refs[n],
        new_NPU_scope_qualified=False)


def allocate_independent_epoch(engine, fixture, *, stage_all4):
    """Allocate real and oracle owners; independently write ONLY initial history.

    Full prompt tokens reserve its 18 pages, but only initial_cursor tokens are
    evaluated.  Later rounds cannot extend a live lease or substitute dense
    re-prefill.  Reference activations use the same original allocator API.
    """
    source = stage_all4('independent-source-contract', lambda: immutable_source(fixture))
    stage_all4('independent-engine-contract', lambda: _require_engine(engine))
    n = len(source.tree_tokens)
    PHYSICAL_S=fixture['scope'].physical_S
    phases = fixture['scope'].phases()
    d = PHYSICAL_S-n-1
    prefixes = [t[:-1] for t in source.tree_tokens]+[source.prompt_tokens[:source.initial_cursor]]
    allocation_tokens = [list(t) for t in prefixes[:-1]]+[list(source.prompt_tokens)]
    def allocate():
        engine._release_cache()
        engine._allocate_cache(allocation_tokens, enable_prefix_caching=False,
            reserve_sequence_capacity=True)
        pool = list(engine._tree_graph_scratch_sequence_ids)
        owners = epoch_owner_partition(n,pool)
        pads = owners['actual_padding_ids']
        refs = owners['reference_span_ids']
        ids = owners['candidate_span_ids']
        histories = prefixes+[()]*d
        for owner, tokens in zip(refs, histories, strict=True):
            # Full prompt reserves pages; it is still evaluated ONLY to the
            # immutable initial cursor below. Empty padding has no fake token.
            reserve = source.prompt_tokens if owner == refs[n] else tokens
            if reserve:
                engine._activate_cache_sequence(owner, list(reserve), enable_prefix_caching=False)
        pages = set()
        allocations = {}
        for ordinal, owner in enumerate(ids+refs):
            span = ordinal % PHYSICAL_S
            maximum = (len(prefixes[span])+5*len(phases)+4 if span < n
                else len(source.prompt_tokens) if span == n
                else max(p['scratch_queries'][span-n-1] for p in phases))
            engine._ensure_cache_capacity([owner], [maximum-1])
            table = tuple(p for p in engine.cache_allocation.block_tables[owner] if p >= 0)
            if (not table or len(set(table)) != len(table) or pages.intersection(table)
                    or any(not 0 <= p < 544 for p in table)
                    or engine.cache_allocation.num_cached_tokens[owner] != 0):
                raise HistoryRejected('Private original allocator pages alias or use prefix-cache history')
            pages.update(table)
            allocations[owner] = table
        engine._dynamic_tree_known_KV = {owner: dict(safe_before=len(tokens), tokens={})
            for owner, tokens in zip(ids+refs, histories*2, strict=True)}
        engine._v75_independent_engine_epoch = getattr(engine, '_v75_independent_engine_epoch', 0)+1
        return dict(_issuer=_EPOCH_ISSUER, immutable_source=source, scope=fixture['scope'],
            source_input_sha256=source.source_input_sha256, allocation_object=engine.cache_allocation,
            engine_epoch=engine._v75_independent_engine_epoch, failed=False,
            candidate_tree_ids=list(range(n)), prompt_id=n, actual_padding_ids=pads,
            reference_tree_ids=refs[:n], reference_prompt_id=refs[n],
            reference_padding_ids=tuple(refs[n+1:]),
            candidate_span_ids=ids, span_ids=ids, reference_span_ids=refs, reference_ids=refs,
            source_prefixes=tuple(histories), span_prefixes=tuple(histories),
            reference_prefixes=tuple(histories), candidate_prefixes=tuple(histories),
            initial_prompt_prefix=source.initial_cursor, independent_prompt_cursor=source.initial_cursor,
            next_prompt_phase=0, allocations=allocations, physical_pages=len(pages),
            activation_epochs={owner: 1 for owner in ids+refs},
            draft_reference_ledger=copy.deepcopy(engine._dynamic_tree_known_KV),
            initial_fixture_adoption_is_online_commit=False, history_events=[])
    epoch = stage_all4('independent-actual-allocator-epoch', allocate)
    try:
        for owner, tokens in zip(epoch['candidate_span_ids']+epoch['reference_span_ids'],
                epoch['source_prefixes']*2, strict=True):
            _write_initial_prefix(engine, owner, tokens, stage_all4=stage_all4)
        stage_all4('independent-initial-device-fence', lambda: _device_fence(engine))
        histories = dict(zip(epoch['candidate_span_ids']+epoch['reference_span_ids'],
            epoch['source_prefixes']*2, strict=True))
        epoch['initial_prefix_hashes'] = stage_all4('independent-all-used-prefix-finite-hashes',
            lambda: prefix_evidence(engine, histories))
        epoch['initial_prefix_forward_completed'] = True
    except BaseException:
        epoch['failed'] = True
        raise
    return epoch


def initialize_private_prompt_cursor(manager, identity, epoch, *, stage_all4):
    """Use the manager's local-only adoption API, never assign its cursor.

    stage_all4 must be the actual NativeBoundaries.stage bound method.  Its
    engine and observed activation epoch bind the physical lease.  An initial
    fixture prefix has not executed original online tree consensus/commit.
    """
    boundaries = getattr(stage_all4, '__self__', None)
    engine = getattr(boundaries, 'engine', None)
    if engine is None:
        raise HistoryRejected('Initial adoption requires the actual bound four-rank coordinator')
    def observe():
        source = _epoch(engine, epoch)
        if (epoch.get('initial_prefix_forward_completed') is not True
                or epoch['next_prompt_phase'] != 0
                or epoch['independent_prompt_cursor'] != source.initial_cursor
                or identity.request_index != epoch['prompt_id']
                or getattr(boundaries, 'allocation_object', None) is not engine.cache_allocation
                or boundaries.activation_epochs.get(identity.request_index) != identity.allocation_epoch):
            raise HistoryRejected('Initial adoption lacks actual original allocator activation evidence')
        _device_fence(engine)
        hashes = prefix_evidence(engine, {epoch['prompt_id']: source.prompt_tokens[:source.initial_cursor]})
        if hashes[epoch['prompt_id']] != epoch['initial_prefix_hashes'][epoch['prompt_id']]:
            raise HistoryRejected('Initialized prompt prefix changed before local adoption')
        return InitialRankEvidence(engine.rank, 'draft' if engine.is_draft else 'target',
            len(engine.model.layers), epoch['engine_epoch'], identity.allocation_epoch,
            source.initial_cursor, _sha(source.prompt_tokens), source.source_input_sha256,
            hashes[epoch['prompt_id']], True, True, epoch['allocations'][epoch['prompt_id']])
    if epoch['immutable_source'].initial_cursor==0:
        def original_zero_cursor():
            source=_epoch(engine,epoch);row=manager.current
            if (row is None or row.lease.identity!=identity or row.committed_cursor!=0
                or row.last_committed_cycle_id!=-1 or row.initial_fixture_cursor is not None
                or row.initial_fixture_scope is not None or epoch['next_prompt_phase']!=0):
                raise HistoryRejected('Cursor0 representative must retain its actual fresh reservation')
            return type('OriginalZeroCursor',(),dict(initial_cursor=0,adoption_kind='original-reservation-cursor0-no-adoption'))()
        initial=stage_all4('independent-original-reservation-cursor0',original_zero_cursor)
        epoch['_manager']=manager;epoch['prompt_lease_identity']=identity
        return initial
    local = stage_all4('independent-initial-prompt-adoption-observe', observe)
    records = _gather_all4(engine, local)
    evidence = InitialPrefixEvidence(epoch['scope'].name, SOURCE_INPUT_SHA256,
        _sha(epoch['immutable_source'].prompt_tokens), identity.request_index, identity.nonce,
        identity.allocation_epoch, epoch['initial_prompt_prefix'], 2301,
        epoch['engine_epoch'], records)
    stage_all4('independent-initial-prompt-adoption-certificate',
        lambda: validate_initial_prefix_evidence(evidence, identity, scope=epoch['scope']))
    # A typed dataclass can be constructed by a caller; it is not an execution
    # capability. Register this exact object only after real device/all4 work.
    _ISSUED_INITIAL_PREFIXES[id(evidence)] = dict(evidence=evidence, engine=engine,
        epoch=epoch, identity=identity, consumed=False)
    stage_all4('independent-initial-prompt-manager-adoption',
        lambda: manager.adopt_local_initial_prefix(identity, evidence=evidence))
    epoch['_manager'] = manager
    epoch['prompt_lease_identity'] = identity
    epoch['history_events'].append(dict(kind=ADOPTION_KIND, cursor=evidence.initial_cursor,
        original_online_tree_commit_claimed=False))
    return evidence


def consume_initial_prefix_evidence(evidence, identity, *, expected_pages):
    """Single-use local capability check called by the manager before adoption.

    No collective is hidden here: initialize_private_prompt_cursor wraps the
    manager operation in the original all4 stage. Forged success dictionaries,
    ordinary dataclasses, stale allocations, and reused certificates all fail.
    """
    require_bounded_scope()
    issued = _ISSUED_INITIAL_PREFIXES.get(id(evidence))
    if (issued is None or issued['evidence'] is not evidence
            or issued['consumed'] or issued['identity'] != identity):
        raise HistoryRejected('Initial prefix evidence was not issued by real reconstruction or was consumed')
    engine, epoch = issued['engine'], issued['epoch']
    _epoch(engine, epoch)
    validate_initial_prefix_evidence(evidence, identity, scope=epoch['scope'])
    boundaries = getattr(engine, '_v75_atomic_boundaries', None)
    own = next(r for r in evidence.rank_records if r.rank == engine.rank)
    pages = tuple(p for p in engine.cache_allocation.block_tables[identity.request_index] if p >= 0)
    if (boundaries is None or boundaries.allocation_object is not engine.cache_allocation
            or boundaries.activation_epochs.get(identity.request_index) != identity.allocation_epoch
            or tuple(expected_pages) != pages or own.pages != pages
            or own.engine_epoch != epoch['engine_epoch']):
        raise HistoryRejected('Initial prefix evidence no longer matches its live original allocation lease')
    # No new model operation is issued here.  Recheck actual prefix bytes rather
    # than treating a previously true field as proof that the live KV survived.
    source = epoch['immutable_source']
    current = prefix_evidence(engine, {identity.request_index: source.prompt_tokens[:evidence.initial_cursor]})
    if current[identity.request_index] != own.prefix_sha256:
        raise HistoryRejected('Initialized prefix bytes changed before the lease adopted them')
    issued['consumed'] = True
    return evidence


def _advance_prompt_reference(engine, epoch, phase_index, *, recompute_same_m, stage_all4):
    """Execute the independent FULL197 actor and consume its real retained history.

    The callback receives immutable tokens + epoch descriptors, not candidate KV.
    It must independently write both real-reference and scratch-reference prompt
    history before returning its same-M/S full numerical record. This method
    records cursor progress of the *oracle only*, never the transaction manager.
    """
    def preflight():
        source = _epoch(engine, epoch)
        if (type(phase_index) is not int or phase_index != epoch['next_prompt_phase']
                or not 0 <= phase_index < len(epoch['scope'].phases())):
            raise HistoryRejected('Prompt reference phase was repeated, skipped or reordered')
        phase = epoch['scope'].phases()[phase_index]
        if epoch['independent_prompt_cursor'] != phase['prompt_start']:
            raise HistoryRejected('Independent progressive cursor lacks its previous real history')
        return source, phase
    source, phase = stage_all4('independent-prompt-reference-phase-contract', preflight)
    histories = dict(zip(epoch['reference_span_ids'], epoch['reference_prefixes'], strict=True))
    before = stage_all4('independent-prompt-reference-prefix-before', lambda: prefix_evidence(engine, histories))
    epoch['_prompt_reference_inflight'] = True
    record = recompute_same_m(engine, epoch, phase_index)
    def draft_prompt():
        if not engine.is_draft:
            return None
        import torch
        owner = epoch['reference_prompt_id']
        positions = list(range(phase['prompt_start'], phase['prompt_end']))
        values = list(source.prompt_tokens[phase['prompt_start']:phase['prompt_end']])
        # The actual original draft chunk uses this exact independent eager FIA
        # method (no LM head and no target KV transfer). Keep its true shape.
        hidden = engine._run_packed_hidden(values, [owner]*len(values), positions,
            use_aclgraph=False, use_fused_infer_attention=True).clone()
        _, meta = engine._prepare_attention_metadata([owner]*len(values), positions, True)
        caches = query_KV(engine, meta)
        if (not bool(torch.isfinite(hidden).all().item())
                or any(not bool(torch.isfinite(v).all().item()) for v in caches)):
            raise HistoryRejected('Independent draft prompt chunk returned nonfinite hidden/KV')
        return dict(hidden=hidden, query_KV=caches, owner=owner,
            start=phase['prompt_start'], end=phase['prompt_end'],
            actual_draft_layers=28, candidate_KV_copied=False)
    draft_record = stage_all4('independent-draft-prompt-reference-chunk', draft_prompt)
    def validate():
        after = prefix_evidence(engine, histories)
        if before != after:
            raise HistoryRejected('Independent mixed call changed its retained source prefix')
        if engine.is_draft:
            if record is not None or draft_record is None:
                raise HistoryRejected('Draft rank cannot manufacture a target first-token reference')
            return after
        required = ('passed', 'source_prefix_finite', 'all_physical_hidden_original_matcher',
            'all64_query_KV_original_matcher', 'exact_target_heads', 'mirror_recomputed')
        if (not isinstance(record, dict) or any(record.get(k) is not True for k in required)
                or record.get('copied_candidate_KV') is not False
                or record.get('phase_index') != phase_index
                or record.get('prompt_start') != phase['prompt_start']
                or record.get('prompt_end') != phase['prompt_end']
                or record.get('reference_span_ids') != epoch['reference_span_ids']
                or record.get('reference_prompt_id') != epoch['reference_prompt_id']
                or record.get('physical_M') != PHYSICAL_M or record.get('segments') != epoch['scope'].physical_S
                or record.get('mask_capacity') != 128
                or record.get('source_input_sha256') != SOURCE_INPUT_SHA256
                or record.get('input_prompt_tokens') != list(source.prompt_tokens[phase['prompt_start']:phase['prompt_end']])):
            raise HistoryRejected('Independent same-M progressive reference lacks actual source/model evidence')
        first = record.get('first_token')
        if (phase['final'] and (type(first) is not int or first < 0)) or (not phase['final'] and first is not None):
            raise HistoryRejected('Partial reference has no publication head; final has one true prompt token')
        values = record.get('reference_query_KV')
        if (not isinstance(values, tuple) or len(values) != 128
                or any(int(v.shape[0]) != 197 for v in values)):
            raise HistoryRejected('Independent reference must retain all64 K/V full197 query rows')
        return after
    try:
        stage_all4('independent-prompt-reference-model-evidence', validate)
        heads = _gather_all4(engine, dict(rank=engine.rank, is_draft=engine.is_draft,
            first_token=None if engine.is_draft else record['first_token']))
        def agree_head():
            target = [r['first_token'] for r in heads if r['is_draft'] is False]
            if (sorted(r['rank'] for r in heads) != list(range(4)) or len(target) != 3
                    or any(t != target[0] for t in target)
                    or (not phase['final'] and target[0] is not None)):
                raise HistoryRejected('Independent target first-token reference disagrees across ranks')
            return target[0]
        first_token = stage_all4('independent-prompt-target-head-consensus', agree_head)
        values = list(epoch['reference_prefixes'])
        values[len(source.tree_tokens)] = source.prompt_tokens[:phase['prompt_end']]
        epoch['reference_prefixes'] = tuple(values)
        epoch['independent_prompt_cursor'] = phase['prompt_end']
        epoch['next_prompt_phase'] += 1
        epoch['history_events'].append(dict(kind='independently-teacher-forced-prompt-chunk',
            start=phase['prompt_start'], end=phase['prompt_end'],
            copied_candidate_KV=False, dense_intermediate_prefill=False,
            final=phase['final'], first_token=first_token,
            original_WORLD_broadcast_claimed=False))
    except BaseException:
        epoch['failed'] = True
        raise
    return record if not engine.is_draft else draft_record


def advance_prompt_reference(engine, epoch, phase_index, *, recompute_same_m, stage_all4):
    """All-rank entry; any error after the first reference work poisons its epoch."""
    try:
        return _advance_prompt_reference(engine, epoch, phase_index,
            recompute_same_m=recompute_same_m, stage_all4=stage_all4)
    except BaseException:
        if epoch.get('_prompt_reference_inflight') is True:
            epoch['failed'] = True
        raise
    finally:
        epoch.pop('_prompt_reference_inflight', None)


def commit_reference_tree(engine, epoch, plans, emitted, accepted_paths,
        reference_query_KV, *, stage_all4):
    """Compact ONLY independent query clones using the real observed verdict.

    This is not a substitute for the candidate's original Native consensus and
    commit. It reconstructs the corresponding oracle history and protects all
    pre-existing reference prefixes. Draft KV is deferred and must be repaired
    from its own ledger, never from these target tensors.
    """
    def preflight():
        source = _epoch(engine, epoch)
        n = len(source.tree_tokens)
        if (not len(plans) == len(emitted) == len(accepted_paths) == n
                or (engine.is_draft and reference_query_KV is not None)
                or (not engine.is_draft and (not isinstance(reference_query_KV, tuple)
                    or len(reference_query_KV) != 128
                    or any(int(v.shape[0]) != 197 for v in reference_query_KV)))):
            raise HistoryRejected('Target compaction requires its complete independent64-layer query clones')
        for span, (p, out, path) in enumerate(zip(plans, emitted, accepted_paths, strict=True)):
            if (p.prefix_len != len(epoch['reference_prefixes'][span])
                    or p.candidate_budget != epoch['scope'].budgets[span] or len(out) != len(path)+1
                    or len(out) not in range(1,min(5,p.candidate_budget+1)+1)
                    or any(type(t) is not int or t < 0 for t in out)
                    or any(type(i) is not int or not 0 <= i < p.candidate_budget for i in path)):
                raise HistoryRejected('Observed B5 verdict is not aligned with the current independent tree source')
            parent = -1
            parents = p.parent_indices.tolist()
            for node in path:
                if parents[node] != parent:
                    raise HistoryRejected('Observed accepted nodes are not a real ancestor path')
                parent = node
        return source
    source = stage_all4('independent-tree-commit-contract', preflight)
    histories = dict(zip(epoch['reference_span_ids'], epoch['reference_prefixes'], strict=True))
    before = stage_all4('independent-tree-prefix-before-compaction', lambda: prefix_evidence(engine, histories))
    def compact():
        if engine.is_draft:
            return dict(independent_query_clones_only=True, copied_candidate_KV=False,
                original_candidate_commit_claimed=False, draft_KV_materialization_deferred=True)
        import torch
        caches = _arrays(engine)
        for span, (owner, plan, path) in enumerate(zip(epoch['reference_tree_ids'], plans, accepted_paths, strict=True)):
            # The first emitted token has already been evaluated as the current
            # root; accepted candidate nodes contribute their own source rows.
            relative = [0]+[1+int(node) for node in path]
            destination = _slots(engine, owner, range(plan.prefix_len, plan.prefix_len+len(relative)))
            index = torch.tensor([sum(p.candidate_budget+1 for p in plans[:span])+i for i in relative], dtype=torch.long, device=engine.device)
            for cache, independent in zip(caches, reference_query_KV, strict=True):
                cache.index_copy_(0, destination, independent.index_select(0, index))
        _device_fence(engine)
        if prefix_evidence(engine, histories) != before:
            raise HistoryRejected('Independent compaction changed a committed source prefix')
        return dict(independent_query_clones_only=True, copied_candidate_KV=False,
            original_candidate_commit_claimed=False)
    try:
        result = stage_all4('independent-target-query-to-canonical-reference', compact)
    except BaseException:
        epoch['failed'] = True
        raise
    if engine.is_draft:
        # The old valid draft prefix/ledger is retained. Newly committed source
        # tokens become valid draft KV only through the next independent repair.
        return result
    values = list(epoch['reference_prefixes'])
    for span, out in enumerate(emitted):
        # Reference prefix excludes the new unevaluated root, exactly as the
        # original logical pipeline. The prior current root is immutable source.
        old_root = source.tree_tokens[span][-1] if not epoch.get('reference_tree_roots') else epoch['reference_tree_roots'][span]
        values[span] = values[span]+(int(old_root),)+tuple(out[:-1])
    epoch['reference_prefixes'] = tuple(values)
    epoch['reference_tree_roots'] = tuple(out[-1] for out in emitted)
    return result


def observe_original_commit(engine, epoch, record, *, stage_all4):
    """Accept original observed logical history, never its candidate cache.

    The original_cycle actor remains responsible for real host verification,
    finite/structural consensus, compaction, state.apply and finish_cycle. These
    records are not numerical qualification and cannot substitute for that call.
    """
    def observe():
        source = _epoch(engine, epoch)
        n = len(source.tree_tokens)
        old = epoch.get('logical_contexts', source.tree_tokens)
        values = record.get('actual_logical_tokens')
        emitted = record.get('committed_tokens')
        if (record.get('original_finish_cycle_executed') is not True
                or record.get('extracted_structural_and_numerical_lanes_unchanged') is not True
                or type(record.get('cycle_id')) is not int
                or record['cycle_id'] <= epoch.get('last_observed_cycle', -1)
                or not isinstance(values, list) or not isinstance(emitted, list)
                or len(values) != n or len(emitted) != n
                or set(record.get('actual_prefix_epochs', {})) != set(range(n))):
            raise HistoryRejected('Independent history has no successful original observed cycle')
        for previous, current, out in zip(old, values, emitted, strict=True):
            if (not isinstance(out, list) or not 1 <= len(out) <= 5
                    or any(type(t) is not int or t < 0 for t in out)
                    or current != list(previous)+out):
                raise HistoryRejected('Observed logical history is not an append of the actual original verdict')
        return tuple(tuple(v) for v in values)
    current = stage_all4('independent-observed-original-history', observe)
    epoch['logical_contexts'] = current
    prefixes = list(epoch['candidate_prefixes'])
    prefixes[:len(current)] = [row[:-1] for row in current]
    epoch['candidate_prefixes'] = tuple(prefixes)
    epoch['last_observed_cycle'] = record['cycle_id']
    return dict(source_tokens_unchanged=True, actual_observed_append=True,
        candidate_KV_copied=False, online_commit_reexecuted=False)


def observe_atomic_prompt_commit(engine, epoch, reservation, *, stage_all4):
    """Observe the real manager AFTER its original-cycle-backed commit.

    This changes only the source descriptor used by the next oracle. It cannot
    advance the manager, synthesize a tree commit, or publish a first token.
    """
    def observe():
        from atomic_prefill import Phase, Reservation
        source = _epoch(engine, epoch)
        index = epoch.get('last_candidate_prompt_phase', -1)+1
        if not 0 <= index < len(epoch['scope'].phases()):
            raise HistoryRejected('Actual atomic prompt commit was repeated')
        phase = epoch['scope'].phases()[index]
        manager = epoch.get('_manager')
        if (type(reservation) is not Reservation or manager is None
                or manager.current is not reservation
                or reservation.lease.identity != epoch.get('prompt_lease_identity')
                or reservation.initial_fixture_scope != (source.scope if source.initial_cursor else None)
                or reservation.initial_fixture_cursor != (source.initial_cursor if source.initial_cursor else None)
                or reservation.prompt != source.prompt_tokens
                or reservation.committed_cursor != phase['prompt_end']
                or reservation.last_committed_cycle_id != epoch.get('last_observed_cycle')
                or reservation.last_committed_cycle_id < 0
                or reservation.chunk is not None or reservation.prepared_cursor is not None
                or reservation.phase != (Phase.FINAL if phase['final'] else Phase.RESERVED)
                or len(epoch['candidate_prefixes'][len(source.tree_tokens)]) != phase['prompt_start']):
            raise HistoryRejected('Prompt history has no actual manager commit after the original observed tree cycle')
        return source, index, phase
    source, index, phase = stage_all4('independent-observe-original-atomic-prompt-commit', observe)
    prefixes = list(epoch['candidate_prefixes'])
    prefixes[len(source.tree_tokens)] = source.prompt_tokens[:phase['prompt_end']]
    epoch['candidate_prefixes'] = tuple(prefixes)
    epoch['last_candidate_prompt_phase'] = index
    return dict(original_manager_commit_observed=True, committed_cursor=phase['prompt_end'],
        manager_cursor_assigned=False, first_token_published_by_reference=False)


def _normal_entries(epoch):
    source = epoch['immutable_source']
    contexts = epoch.get('logical_contexts', source.tree_tokens)
    entries = []
    for owner, values in zip(epoch['reference_tree_ids'], contexts, strict=True):
        p = len(values)-1
        known = epoch['draft_reference_ledger'][owner]
        start = p
        for position in range(known['safe_before'], p):
            if known['tokens'].get(position) != values[position]:
                start = position
                break
        entries.append((owner, p, start, tuple(values[start:p+1])))
    return tuple(entries)


def normal_draft_count_contract(epoch):
    """Exact declared actor domain, checked on all roles before draft-only work."""
    scope=epoch['scope'];source=epoch['immutable_source']
    counts=frozenset(s.target_requests for s in SCOPES)
    n=scope.target_requests
    contexts=epoch.get('logical_contexts',source.tree_tokens)
    if (scope not in SCOPES or counts!=frozenset(range(15,33)) or type(n) is not int
        or n not in counts or len(source.tree_tokens)!=n or len(contexts)!=n
        or len(epoch['candidate_tree_ids'])!=n or len(epoch['reference_tree_ids'])!=n
        or len(set(epoch['candidate_tree_ids']+list(epoch['reference_tree_ids'])))!=2*n):
        raise HistoryRejected('Independent normal draft count/source/owner domain differs from declared scopes')
    return n


def recompute_normal_draft(engine, native, epoch, *, stage_all4):
    """Independent eager PA/top2 reconstruction of a normal four-token pool.

    Real and dummy physical rows follow the original normal FastV5/root-seed
    ABI.  The independent packed first call is built from immutable committed
    tokens and this actor's own ledger. No candidate input, hidden or KV is used.
    The returned fullphysical hidden/encoded/query KV must still be compared by
    the bounded qualifier; this actor does not mark any Native entry qualified.
    """
    def validate():
        _epoch(engine, epoch)
        n=normal_draft_count_contract(epoch)
        entries=_normal_entries(epoch) if engine.is_draft else ()
        if engine.is_draft and len(entries)!=n:
            raise HistoryRejected('Independent normal entry count differs from the declared source')
        return entries
    entries = stage_all4('independent-normal-draft-source', validate)
    def recompute():
        if not engine.is_draft:
            return None
        import torch
        from types import SimpleNamespace
        from chain_metadata import build as chain_metadata
        n = len(entries)
        head = native._next_linear_draft_graph_bucket(n, engine.config.max_num_seqs)
        has_repair = any(len(row[3]) > 1 for row in entries)
        real = sum(len(row[3]) for row in entries) if has_repair else n
        physical = native._next_linear_draft_graph_bucket(real, 128) if has_repair else head
        histories = {owner: epoch.get('logical_contexts', epoch['immutable_source'].tree_tokens)[i][:start]
            for i, (owner, _, start, _) in enumerate(entries)}
        # The PAD reads prefix0 of the first real owner's block table. That source
        # prefix is included in the protected real history, never borrowed from the
        # candidate owner with the same logical tokens.
        before = prefix_evidence(engine, histories)
        ids, positions, tokens, root_indices = [], [], [], []
        for owner, p, start, values in entries:
            ids.extend([owner]*len(values))
            positions.extend(range(start, p+1))
            tokens.extend(values)
            root_indices.append(len(tokens)-1)
        engine._ensure_cache_capacity(ids, positions)
        engine._ensure_cache_capacity([owner for owner, _, _, _ in entries for _ in range(4)],
            [p+j for _, p, _, _ in entries for j in range(4)])
        x = torch.tensor(tokens, dtype=torch.long, device=engine.device)
        pos, meta = engine._prepare_attention_metadata(ids, positions, False)
        x, pos, meta = native.NativeACLGraphRunner._pad_inputs(x, pos, meta, physical)
        if (meta.use_fused_infer_attention or meta.context_lens.tolist() != [p+1 for p in positions]+[1]*(physical-real)
                or meta.slot_mapping.tolist()[real:] != [-1]*(physical-real)):
            raise HistoryRejected('Independent normal PA lost its exact physical dummy/causal contract')
        first = engine.model(x, pos, meta).clone()  # Before shared graph-pool users.
        if not bool(torch.isfinite(first).all().item()):
            raise HistoryRejected('Independent first normal PA returned nonfinite hidden')
        selected = first
        if has_repair:
            selected = first.new_zeros((head, first.shape[-1]))
            selected[:n].copy_(first.index_select(0, torch.tensor(root_indices, dtype=torch.long, device=engine.device)))
        def top2(hidden):
            logits = engine.model.compute_logits(hidden)[:, :engine.draft_vocab_size]
            probability = torch.softmax(logits.float(), dim=-1)
            confidence, token = probability.topk(2, dim=-1)
            return token.long()*1_000_001+torch.round(confidence.clamp(0,1)*1_000_000).long(), token[:, 0]
        seed, inputs = top2(selected)
        encoded = [seed]
        tail_hidden = []
        tail_input_tokens = []
        plans = [SimpleNamespace(prefix_len=p+1) for _, p, _, _ in entries]
        owners = [row[0] for row in entries]
        _, tail_positions, tail_metadata = chain_metadata(engine, plans, owners, [0]*n, 3, head)
        for p, attn in zip(tail_positions, tail_metadata, strict=True):
            tail_input_tokens.append(inputs.clone())
            hidden = engine.model(inputs, p, attn).clone()
            tail_hidden.append(hidden)
            value, inputs = top2(hidden)
            encoded.append(value)
        physical_encoded = torch.stack(encoded, dim=1)
        real_encoded = physical_encoded[:n]
        pairs = real_encoded.cpu().tolist()
        rows = [[pair[0]//1_000_001 for pair in row]+[pair[1]//1_000_001 for pair in row] for row in pairs]
        scores = [[pair[0]%1_000_001/1_000_000 for pair in row]+[pair[1]%1_000_001/1_000_000 for pair in row] for row in pairs]
        ledger = {owner: dict(safe_before=p,
            tokens={p+j: token for j, token in enumerate((values[-1], *row[:3]))})
            for (owner, p, _, values), row in zip(entries, rows, strict=True)}
        if prefix_evidence(engine, histories) != before:
            raise HistoryRejected('Independent normal draft changed a valid known prefix')
        index = torch.cat([_slots(engine, owner, range(start, p+4)) for owner, p, start, _ in entries])
        query = tuple(a.index_select(0, index).clone() for a in _arrays(engine))
        if any(not bool(torch.isfinite(v).all().item()) for v in (*tail_hidden, physical_encoded, *query)):
            raise HistoryRejected('Independent normal draft tail/encoded/queryKV is nonfinite')
        epoch['draft_reference_ledger'].update(copy.deepcopy(ledger))
        prefixes = list(epoch['reference_prefixes'])
        contexts = epoch.get('logical_contexts', epoch['immutable_source'].tree_tokens)
        prefixes[:n] = [tuple(v[:-1]) for v in contexts]
        epoch['reference_prefixes'] = tuple(prefixes)
        return dict(rows=rows, scores=scores, ledger=ledger, first_hidden=first,
            first_metadata=meta, first_inputs=x.clone(), first_positions=pos.clone(),
            first_real_count=real, first_physical=physical, root_mode=has_repair,
            root_indices=tuple(root_indices), head_rows=head,
            tail_hidden=tuple(tail_hidden), tail_input_tokens=tuple(tail_input_tokens),
            physical_encoded=physical_encoded, encoded=real_encoded,
            query_KV=query, query_starts=tuple(row[2] for row in entries),
            query_lengths=tuple(row[1]+4-row[2] for row in entries),
            independent_reference_owners=tuple(owners), candidate_KV_copied=False,
            native_qualification_assigned=False, NPU_qualification_receipt=False,
            normal_request_count=n, declared_scope=epoch['scope'].name,
            reference_common_stage='independent-normal-draft-reference-model-and-ledger')

    return stage_all4('independent-normal-draft-reference-model-and-ledger',recompute)


def compare_normal_draft(engine, epoch, actual, reference, *, actual_first_hidden,
        actual_tail_hidden, actual_physical_encoded, actual_query_KV,
        actual_ledger, stage_all4):
    """Original matcher and exact public ABI; callers supply immediate snapshots.

    The candidate first/tail tensors must be cloned at their actual return,
    never retrieved from resident graph outputs after another graph replay.
    Missing fullphysical evidence fails; real-only equality cannot close a PAD
    gate. This remains numerical comparison, not changed-input qualification.
    """
    def compare():
        _epoch(engine, epoch)
        if not engine.is_draft:
            if actual is not None or reference is not None:
                raise HistoryRejected('Target role cannot supply a draft reference')
            return None
        import torch
        matcher = engine.graph_runner._target_outputs_match
        n = normal_draft_count_contract(epoch)
        if (reference['normal_request_count']!=n or reference['declared_scope']!=epoch['scope'].name
            or reference['reference_common_stage']!='independent-normal-draft-reference-model-and-ledger'):
            raise HistoryRejected('Independent normal reference lacks its exact declared count/common stage')
        if (actual['draft_token_ids'] != reference['rows']
                or actual['root_token_ids'] != [values[-1] for values in epoch.get('logical_contexts',
                    epoch['immutable_source'].tree_tokens)]
                or len(actual['node_confidences']) != n
                or len(actual_tail_hidden) != 3
                or not isinstance(actual_query_KV, tuple) or len(actual_query_KV) != 56
                or actual_first_hidden.shape != reference['first_hidden'].shape
                or actual_physical_encoded.shape != reference['physical_encoded'].shape
                or not torch.equal(actual_physical_encoded, reference['physical_encoded'])
                or not matcher((actual_first_hidden,), (reference['first_hidden'],))
                or not matcher(tuple(actual_tail_hidden), reference['tail_hidden'])
                or not matcher(actual_query_KV, reference['query_KV'])):
            raise HistoryRejected('Independent fullphysical first/tail/encoded/all28queryKV or public tokens differ')
        for scores, expected in zip(actual['node_confidences'], reference['scores'], strict=True):
            if not torch.equal(scores.cpu(), torch.tensor(expected, dtype=scores.dtype)):
                raise HistoryRejected('Independent draft node confidence differs exactly')
        for candidate_owner, ref_owner in zip(epoch['candidate_tree_ids'], epoch['reference_tree_ids'], strict=True):
            if actual_ledger.get(candidate_owner) != reference['ledger'].get(ref_owner):
                raise HistoryRejected('Independent normal draft ledger differs at a real owner')
        return dict(normal_request_count=n,declared_scope=epoch['scope'].name,
            reference_common_stage='independent-normal-draft-reference-model-and-ledger',
            all_physical_first_hidden_original_matcher=True,
            all_physical_tail_hidden_original_matcher=True, exact_physical_top2_confidence=True,
            exact_public_rows_roots_node_confidences=True, all28_query_KV_original_matcher=True,
            independent_per_owner_ledger=True, candidate_KV_used_to_initialize_reference=False,
            native_changed_input_qualification_claimed=False, NPU_qualification_receipt=False)
    return stage_all4('independent-normal-draft-original-numeric-and-exact-ABI', compare)
