"""Issue E4 only after fresh same-engine75 lifecycle and original model gates."""
from dataclasses import dataclass, field
from scope159 import Capability

_ISSUER = object()


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT423 current proof: '+message)


@dataclass(frozen=True)
class Record:
    issuer: object = field(repr=False)
    owner: object = field(repr=False)
    model: object = field(repr=False)
    bounded: object = field(repr=False)
    audit: object = field(repr=False)
    dispatcher: object = field(repr=False)
    entries: tuple = field(repr=False)
    capability: Capability


def evidence(engine):
    base = getattr(engine, '_v173_qualified_component', None)
    canonical = getattr(engine, '_canonical257_actual_proof', None)
    bucket = getattr(engine, '_uniform_KV_bucket423_proof', None)
    bound = getattr(engine, '_root423_bounded_actual', None)
    audit = getattr(engine, '_root423_bounded_audit', None)
    need(base is not None and base['audit']['component_numerical_qualified'] is True
         and base['causal80']['audit']['math_independent_qualified'] is True,
         'fresh original173 independent math')
    need(canonical is not None and canonical['original_component'] is engine._v209_qualified_component
         and canonical['cases'] == 227 and len(canonical['math']['cases']) == 227,
         'same-engine original227 with original component object')
    need(bucket is not None and bucket['cases'] == 681
         and bucket['same_model_instance'] is True
         and bucket['original227_FULL_math_preserved'] is True,
         'fresh original681 additional target math')
    need(bound is not None and bound['two_complete21_zero_traces'] is True
         and len(bound['traces']) == 75 and bound['Native_flags_assigned'] is False,
         'actual current75-case lifecycle object')
    for pass_index in range(3):
        values = [v for v in bound['traces'] if v['pass_index'] == pass_index]
        need(len(values) == 25, 'all25 cases each pass')
        if pass_index:
            from scope419 import NATIVE_COUNTERS
            need(all(set(v['Native_delta']) == set(NATIVE_COUNTERS)
                     and all(type(x) is int and x == 0 for x in v['Native_delta'].values())
                     for v in values), 'two full21 zero repeated passes')
    need(audit is not None and audit['passed'] is True
         and audit['full_required_natural_coverage'] is True,
         'all4 raw histories independently audited')
    binding = getattr(engine, '_parent4_target391_binding', None)
    need(binding is not None and binding['model_id'] == id(engine.model)
         and binding['original173'] is base,
         'same actual target binding and current model')
    if not engine.is_draft:
        need(binding['target_state'] is engine._branch_tail_state
             and binding['actual_metadata_matches'] > 0,
             'current target metadata observed in real execution')
    need(engine.cache_allocation is None and engine.cache_block_tables is None
         and not engine.graph_runner.graph_cache_sealed,
         'fixture released and serving graphs not sealed')
    return bound, audit


def verify_record(engine, record):
    need(type(record) is Record and record.issuer is _ISSUER
         and record.owner is engine and record.model is engine.model,
         'exact issuer/current owner')
    need(record.bounded is getattr(engine, '_root423_bounded_actual', None)
         and record.audit is getattr(engine, '_root423_bounded_audit', None),
         'no historical or copied proof permission')
    from vllm_ascend.spec_decode.pearl import native_engine as native
    from factory423 import current
    need(record.dispatcher is current(native), 'same installed original fallback and E4 dispatcher')
    need(record.capability == Capability(frozenset({(24, 4)}),
         frozenset((4, r) for r in range(1, 25)), mode='production'),
         'exact tested shared and repair domain')
    return True


def issue(engine, native):
    bound, audited = evidence(engine)
    from factory423 import current
    dispatcher = current(native)
    runner = engine.graph_runner
    entries = []
    if engine.is_draft:
        key = (f'draft-greedy:{engine.draft_vocab_size}|steps:4|paged|confidence', 28)
        entry = runner.draft_entries.get(key)
        need(entry is not None and entry.runtime_validated is True
             and entry.validated_real_row_count >= 28
             and key not in runner.disabled_entry_keys,
             'naturally closed shared4 actual28-row Native entry')
        entries.append(('draft', key, entry))
        target_keys = [('target-hidden:0|steps:1|paged', (m,))
                       for m in (8, 12, 16, 20, 24, 28)]
        target_keys.append((f'target-greedy:{engine.draft_vocab_size}|steps:1|paged', (4,)))
        for key in target_keys:
            entry = runner.target_entries.get(key)
            need(entry is not None and entry.runtime_validated is True
                 and key not in runner.disabled_entry_keys,
                 'naturally closed original union/frontier Native entry '+str(key))
            entries.append(('target', key, entry))
    record = Record(_ISSUER, engine, engine.model, bound, audited, dispatcher,
                    tuple(entries), Capability(frozenset({(24, 4)}),
                    frozenset((4, r) for r in range(1, 25)), mode='production'))
    verify_record(engine, record)
    from live423 import activate
    return activate(engine, record)


def snapshot(engine):
    record = getattr(engine, '_root423_E4_permission', None)
    if record is None:
        return dict(issued=False, full463_qualified=False)
    verify_record(engine, record)
    from live423 import permission
    permission(engine)
    return dict(issued=True, same_engine_protocol_cases=75,
        repeated_zero_cases=50, natural_mixed_and_leaf_coverage=True,
        same_engine_original227_and681=True, naturally_validated_entries=len(record.entries),
        shared4_pairs=[[24, 4]], repair_range=[0, 24],
        original_fallback_preserved=True, Native_flags_assigned=False,
        full463_qualified=False)
