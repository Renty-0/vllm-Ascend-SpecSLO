"""Retain all best257 and original209/173 gates; add same-engine681 math."""
from dataclasses import dataclass, field
import scope257 as original

_ISSUER = object()


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT423 scope: ' + message)


def current(engine):
    inherited = original.current(engine)
    from vllm_ascend.spec_decode.pearl import native_engine as native
    need(native.NativePearlEngine.draft_tree_forward is engine._best257_draft423,
         'same actual installed dispatcher and original RootSeed')
    from proof423 import verify_record
    from live423 import permission as e4_permission
    verify_record(engine, e4_permission(engine))
    permission = getattr(engine, '_uniform_KV_bucket423_permission', None)
    need(permission is not None and permission['issuer'] is _ISSUER
         and permission['engine'] is engine
         and permission['proof'] is getattr(engine, '_uniform_KV_bucket423_proof', None),
         'same current681 proof and actual model')
    if not engine.is_draft:
        state = engine._branch_tail_state
        need(state.uniform_KV_bucket423_enabled is True
             and state.prepare is permission['prepare']
             and engine.graph_runner.run_target_greedy is permission['greedy'],
             'same checked uniform-bucket dispatch callbacks')
    return inherited


@dataclass(frozen=True)
class Scope:
    engine: object = field(repr=False)
    original: object = field(repr=False)
    issuer: object = field(repr=False)

    def require(self, engine):
        need(self.issuer is _ISSUER and engine is self.engine, 'same issued423 scope')
        need(self.original.require(engine) == frozenset(), 'all original257/209/173 obligations')
        current(engine)
        return frozenset()


def issue(engine, *, component_proof):
    proof = getattr(engine, '_uniform_KV_bucket423_proof', None)
    need(proof is not None and proof['cases'] == 681
         and proof['same_model_instance'] is True
         and proof['original227_FULL_math_preserved'] is True
         and engine.cache_allocation is None and engine.cache_block_tables is None
         and not engine.graph_runner.graph_cache_sealed,
         'all fresh target math completed and released before unchanged warm/seal')
    need(not hasattr(engine, '_uniform_KV_bucket423_permission'), 'issue only once')
    engine._uniform_KV_bucket423_permission = dict(issuer=_ISSUER, engine=engine, proof=proof,
        prepare=engine._branch_tail_state.prepare, greedy=engine.graph_runner.run_target_greedy)
    # No device graph validation state is assigned here. Original issue checks
    # the unchanged227 proof and actual original Native graph entry objects.
    inherited = original.issue(engine, component_proof=component_proof)
    result = Scope(engine, inherited, _ISSUER)
    result.require(engine)
    return result


def report(scope, engine):
    scope.require(engine)
    result = original.report(scope.original, engine)
    result.update(schema='ROOT423.best257-plus-uniform-KV-bucket-scope.v1',
        original227_FULL2560_cases_preserved=True, additional_current_model_cases=681,
        original_best257_draft_and_branch_copy_unchanged=False,
        original_RootSeed_legacy_Fast_and_branch_copy_retained=True,
        selective_E4_same_engine_cases=75,
        fixed_KV_extent=None, maximum_KV_extent=2560, KV_buckets=[512,1024,1536,2048,2560],
        all_original_warm_twozero_seal_obligations=True,
        whole_workload_AR_qualified=False, formal_Goodput=False, adopted=False)
    return result
