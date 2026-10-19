"""Keep all original209/257 obligations, add current693 proofs and four real entries."""
from dataclasses import dataclass,field
import scope257 as original

_ISSUER=object()


def need(value,message):
    if value is not True:raise RuntimeError('ROOT383 scope: '+message)


def current(engine):
    inherited=original.current(engine)
    permission=getattr(engine,'_hybrid383_permission',None)
    need(permission is not None and permission['issuer'] is _ISSUER
        and permission['engine'] is engine and permission['proof'] is engine._hybrid383_actual_proof,
        'same current693 proof and engine')
    from vllm_ascend.spec_decode.pearl import native_engine as native
    need(native.NativePearlEngine.draft_tree_forward is engine._best257_draft383,'original optimized draft retained')
    state=engine._branch_tail_state
    need(state.adaptive377_enabled is True,'same adaptive geometry enabled')
    if not engine.is_draft:
        from vllm_ascend.spec_decode.pearl import native_model
        need(state.uniform_KV_bucket377_enabled is True
            and engine.model.forward is permission['model_forward']
            and engine.model.lm_head._project_local is permission['head_forward']
            and native_model.matmul_allreduce_add_rmsnorm_or_fallback is permission['mc2'],
            'same qualified persistent hybrid callbacks')
        for key,entry in permission['entries'].items():
            need(engine.graph_runner.target_entries.get(key)is entry and entry.runtime_validated is True
                and key not in engine.graph_runner.disabled_entry_keys,'same naturally validated four graph entries')
    return inherited


@dataclass(frozen=True)
class Scope:
    engine:object=field(repr=False)
    original:object=field(repr=False)
    issuer:object=field(repr=False)

    def require(self,engine):
        need(self.engine is engine and self.issuer is _ISSUER,'same issued full scope')
        need(self.original.require(engine)==frozenset(),'all original209/257 proof and Native obligations')
        current(engine)
        return frozenset()


def issue(engine,*,component_proof):
    proof=getattr(engine,'_hybrid383_actual_proof',None)
    need(proof is not None and proof['cases']==693 and proof['same_model_instance'] is True
        and proof['original227_FULL_math_preserved'] is True and len(proof['banks'])==3
        and engine.cache_allocation is None and engine.cache_block_tables is None
        and not engine.graph_runner.graph_cache_sealed,'fresh693 and released banks before original warm/seal')
    need(not hasattr(engine,'_hybrid383_permission'),'issue exactly once')
    entries={};model_forward=head_forward=mc2=None
    if not engine.is_draft:
        from vllm_ascend.spec_decode.pearl import native_model
        model_forward=engine.model.forward;head_forward=engine.model.lm_head._project_local
        mc2=native_model.matmul_allreduce_add_rmsnorm_or_fallback
        for m,s in [(80,16),(120,24)]:
            for capacity in (10,11):
                shape=(((s,1,capacity,4096),(s,32)),)
                key=(f'target-greedy:{engine.draft_vocab_size}|steps:1|tree-fia-full:exact-length-task-update|buffers:{shape}',(m,))
                entry=engine.graph_runner.target_entries.get(key)
                need(entry is not None and entry.runtime_validated is True and key not in engine.graph_runner.disabled_entry_keys,'all four actual Native entries naturally validated')
                entries[key]=entry
    engine._hybrid383_permission=dict(issuer=_ISSUER,engine=engine,proof=proof,entries=entries,
        model_forward=model_forward,head_forward=head_forward,mc2=mc2)
    inherited=original.issue(engine,component_proof=component_proof)
    result=Scope(engine,inherited,_ISSUER);result.require(engine)
    return result


def report(scope,engine):
    scope.require(engine)
    previous=original.report(scope.original,engine)
    previous.update(schema='ROOT383.current693-original227-adaptive-hybrid-scope.v1',
        canonical_M=None,canonical_M_families=[80,120],physical_primary_plus_side_groups=None,
        physical_group_families=[16,24],fixed_KV_extent=None,maximum_KV_extent=2560,
        canonical_natural_entries=0 if engine.is_draft else 4,additional_current_model_cases=693,
        original227_FULL2560_preserved=True,old681_logical_cases_retained_in_current693=True,
        original_models_budget_allocator_arrivals_and_clocks=True,
        original_optimized_draft_preserved=True,whole_workload_AR_qualified=False,formal_Goodput=False,adopted=False)
    return previous
