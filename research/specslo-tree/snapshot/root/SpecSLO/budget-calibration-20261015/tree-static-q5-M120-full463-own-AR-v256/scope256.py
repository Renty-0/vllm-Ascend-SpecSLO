"""Explicit extension of original209 evidence, never inherited device proof."""
from dataclasses import dataclass, field

_ISSUER = object()


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT256 scope: '+message)


def current(engine):
    permission = getattr(engine, '_canonical256_permission', None)
    need(permission is not None and permission['issuer'] is _ISSUER
         and permission['engine'] is engine
         and permission['proof'] is getattr(engine, '_canonical256_actual_proof', None),
         'same actual independent canonical evidence and model instance')
    from vllm_ascend.spec_decode.pearl import native_engine as native
    need(native.NativePearlEngine._pad_spec_rhythm_target_tree_graph is engine._canonical256_pad,
         'same guarded canonical physical pad')
    need(native.NativePearlEngine.target_tree_forward is engine._canonical256_forward,
         'same actual canonical forward and geometry observer')
    need(engine._branch_tail_state is permission['state'], 'same actual branch-copy state')
    need(engine._branch_tail_state.prepare is permission['prepare']
         and engine.graph_runner.run_target_greedy is permission['greedy'], 'same actual static model projection callbacks')
    if not engine.is_draft:
        runner = engine.graph_runner
        for key, entry in permission['entries'].items():
            need(runner.target_entries.get(key) is entry and entry.runtime_validated is True
                 and key not in runner.disabled_entry_keys, 'same naturally validated canonical graph entries')
    return permission


@dataclass(frozen=True)
class Scope:
    engine: object = field(repr=False)
    original: object = field(repr=False)
    issuer: object = field(repr=False)

    def require(self, engine):
        need(self.issuer is _ISSUER and self.engine is engine, 'same issued extended scope')
        current(engine)
        need(self.original.require(engine) == frozenset(), 'all original209/173 gates retained')
        return frozenset()


def issue(engine, *, component_proof):
    import scope209
    original = scope209.issue(engine, component_proof=component_proof)
    proof = getattr(engine, '_canonical256_actual_proof', None)
    need(proof is not None and proof['original_component'] is component_proof
         and proof['cases'] == 227 and proof['original_sampler_unchanged'] is True,
         'fresh same-instance independent node proof')
    need(len(proof['math']['cases']) == 227 and engine.cache_allocation is None
         and engine.cache_block_tables is None and not engine.graph_runner.graph_cache_sealed,
         'all 227 actual cases complete and independent bank released')
    if not engine.is_draft:
        need(proof['math']['all_independent_prefixes_unchanged'] is True,
             'all committed independent prefixes protected')
        from oracle256 import cases
        from audit_static_calls256 import audit_calls
        audit_calls(cases(),proof['actual_static_graph_calls'])
    need(not hasattr(engine, '_canonical256_permission'), 'issue new scope once')
    entries = {}
    if not engine.is_draft:
        signatures = {(24,10),(24,11)}
        for segments, capacity in signatures:
            shape = (((segments, 1, capacity, 4096), (segments, 32)),)
            key = (f'target-greedy:{engine.draft_vocab_size}|steps:1|tree-fia-full:exact-length-task-update|buffers:{shape}', (120,))
            entry = engine.graph_runner.target_entries.get(key)
            need(entry is not None, 'every predeclared actual canonical greedy key captured')
            entries[key] = entry
    if not engine.is_draft:
        need(bool(entries), 'actual canonical graphs captured and naturally validated')
    engine._canonical256_permission = dict(issuer=_ISSUER, engine=engine, proof=proof,
                                         state=engine._branch_tail_state, entries=entries,
                                         prepare=engine._branch_tail_state.prepare,
                                         greedy=engine.graph_runner.run_target_greedy)
    engine._canonical256_phase = 'qualified'
    result = Scope(engine, original, _ISSUER)
    result.require(engine)
    return result


def report(scope, engine):
    import scope209
    scope.require(engine)
    previous = scope209.report(scope.original, engine)
    return dict(previous, schema='ROOT256.actual-canonical-extended-standalone-scope.v1',
                zero_new_keys=False, original209_scope=previous,
                canonical_same_instance_cases=227, canonical_M=120,
                effective_max_target_requests=24, effective_max_verification_candidates=96,
                original_configured_target_requests=32, original_configured_verification_candidates=96,
                linear_FULL_mask_capacity=10, branch_copy_FULL_mask_capacity=11,
                fixed_KV_extent=2560, physical_primary_plus_side_groups=24,
                canonical_natural_entries=len(current(engine)['entries']),
                no_Native_qualification_flag_assignment=True,
                whole_workload_AR_qualified=False, formal_Goodput=False, adopted=False)
