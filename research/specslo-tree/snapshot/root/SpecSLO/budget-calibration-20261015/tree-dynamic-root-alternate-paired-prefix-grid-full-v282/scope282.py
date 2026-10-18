"""Explicit extension of original209 evidence, never inherited device proof."""
from dataclasses import dataclass, field

_ISSUER = object()


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT282 scope: '+message)


def current(engine):
    from static_state282 import require_copy_bindings
    require_copy_bindings(engine)
    permission = getattr(engine, '_canonical282_permission', None)
    import budget282
    if permission is not None:
        need(permission['budget_shape'] is budget282.shape and permission['target_plan'] is budget282.build_plan, 'same LIVE group-budget delegates')
    need(permission is not None and permission['issuer'] is _ISSUER
         and permission['engine'] is engine
         and permission['proof'] is getattr(engine, '_canonical282_actual_proof', None),
         'same actual independent canonical evidence and model instance')
    from vllm_ascend.spec_decode.pearl import native_engine as native
    need(native.NativePearlEngine._pad_spec_rhythm_target_tree_graph is engine._canonical282_pad,
         'same guarded canonical physical pad')
    need(native.NativePearlEngine.target_tree_forward is engine._canonical282_forward,
         'same actual canonical forward and geometry observer')
    need(engine._branch_tail_state is permission['state'], 'same actual branch-copy state')
    need(engine._branch_tail_state.prepare is permission['prepare']
         and engine.graph_runner.run_target_greedy is permission['greedy'], 'same actual static model projection callbacks')
    if engine.is_draft:
        need(engine._ROOT281_qualified_private_copy is permission['draft_copy'], 'same numerically checked persistent private copy object')
    if not engine.is_draft:
        runner = engine.graph_runner
        need(runner._run_target is permission['dispatcher'], 'same private capture-grid dispatcher')
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
    proof = getattr(engine, '_canonical282_actual_proof', None)
    need(proof is not None and proof['original_component'] is component_proof
         and proof['cases'] == 408 and proof['original_sampler_unchanged'] is True,
         'fresh same-instance independent node proof')
    need(len(proof['math']['cases']) == 408 and engine.cache_allocation is None
         and engine.cache_block_tables is None and not engine.graph_runner.graph_cache_sealed,
         'all 408 actual cases complete and independent bank released')
    if not engine.is_draft:
        need(proof['math']['all_independent_prefixes_unchanged'] is True,
             'all committed independent prefixes protected')
        from oracle282 import cases
        from audit_static_calls282 import audit_calls
        audit_calls(cases(),proof['actual_static_graph_calls'])
    draft=getattr(engine,'_ROOT282_draft_numeric_proof',None)
    need(draft is not None,'fresh actual draft numerical and postbranch proof')
    if engine.is_draft:
        need(len(draft['primary_alternate45']['cases'])==45 and len(draft['postbranch20']['cases'])==20,
             'all actual fresh draft45 and postbranch20 controls')
        need(all(r['alternate_allhidden_bitwise']and r['alternate_all56KV_bitwise']and r['exact_alternate_top2']for r in draft['primary_alternate45']['cases']),
             'same-instance genuine alternate numerical gates')
        need(all(r['original_next_primary_exacttop2_and_all56KV']and r['next_genuine_alternate_bitwise_hidden_and56KV']and r['canonical_prefix_KV_ledger_tables_protected']for r in draft['postbranch20']['cases']),
             'same-instance original deferred repair and following alternate')
        need(getattr(engine,'_ROOT281_qualified_private_copy',None)is not None,'actual qualified persistent private graph')
    need(not hasattr(engine, '_canonical282_permission'), 'issue new scope once')
    entries = {}
    if not engine.is_draft:
        signatures=[(10,0)]+[(11,b)for b in(128,256,512,1024,1792)]
        for capacity,grid in signatures:
            shape=(((24,1,capacity,4096),(24,32)),)
            text=f'target-greedy:{engine.draft_vocab_size}|steps:1|tree-fia-full:exact-length-task-update|buffers:{shape}'
            if grid:text+='|ROOT280:copy-grid:'+str(grid)
            key=(text,(120,))
            entry = engine.graph_runner.target_entries.get(key)
            need(entry is not None, 'every predeclared actual canonical greedy key captured')
            entries[key] = entry
    if not engine.is_draft:
        need(bool(entries), 'actual canonical graphs captured and naturally validated')
    engine._canonical282_permission = dict(issuer=_ISSUER, engine=engine, proof=proof,
                                         state=engine._branch_tail_state, entries=entries,
                                         prepare=engine._branch_tail_state.prepare,
                                         greedy=engine.graph_runner.run_target_greedy,
                                         budget_shape=__import__('budget282').shape, target_plan=__import__('budget282').build_plan,
                                         dispatcher=getattr(engine.graph_runner,'_run_target',None),
                                         draft_copy=getattr(engine,'_ROOT281_qualified_private_copy',None))
    engine._canonical282_phase = 'qualified'
    result = Scope(engine, original, _ISSUER)
    result.require(engine)
    return result


def report(scope, engine):
    import scope209
    scope.require(engine)
    previous = scope209.report(scope.original, engine)
    return dict(previous, schema='ROOT282.actual-canonical-extended-standalone-scope.v1',
                zero_new_keys=False, original209_scope=previous,
                canonical_same_instance_cases=408, canonical_M=120,
                actual_LIVE_dynamic_B1to8_root_alternate_policy=True, actual_target_primary_side_group_budget=24,
                actual_original218_repair_solver_and_generation_state_preserved=True,
                checked_raw_byte_KV_copy_installed=True,
                original_global_environment_TARGET0_unchanged=True, actual_LIVE_target_SLO_first=True,
                effective_max_target_requests=24, effective_max_verification_candidates=96,
                original_configured_target_requests=32, original_configured_verification_candidates=96,
                linear_FULL_mask_capacity=10, branch_copy_FULL_mask_capacity=11,
                fixed_KV_extent=2560, physical_primary_plus_side_groups=24,
                canonical_natural_entries=len(current(engine)['entries']),
                genuine_draft_cases45=True,postbranch_repair_controls20=True,actual_online_paired_gain=True,
                no_Native_qualification_flag_assignment=True,
                whole_workload_AR_qualified=False, formal_Goodput=False, adopted=False)
