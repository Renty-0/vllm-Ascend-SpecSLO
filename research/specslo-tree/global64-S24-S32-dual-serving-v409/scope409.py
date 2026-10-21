"""Retain all best257 and original209/173 gates; add same-engine681 math."""
from dataclasses import dataclass, field
import scope257 as original

_ISSUER = object()


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT409 scope: ' + message)


def current(engine):
    inherited = original.current(engine)
    from vllm_ascend.spec_decode.pearl import native_engine as native
    need(native.NativePearlEngine.draft_tree_forward is engine._best257_draft409,
         'actual installed optimized best257 draft preserved')
    permission = getattr(engine, '_uniform_KV_bucket409_permission', None)
    need(permission is not None and permission['issuer'] is _ISSUER
         and permission['engine'] is engine
         and permission['proof'] is getattr(engine, '_uniform_KV_bucket409_proof', None),
         'same current681 proof and actual model')
    from scope_wide409 import current as wide_current
    wide_current(engine, permission)
    if not engine.is_draft:
        state = engine._branch_tail_state
        need(state.uniform_KV_bucket409_enabled is True
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
        need(self.issuer is _ISSUER and engine is self.engine, 'same issued409 scope')
        need(self.original.require(engine) == frozenset(), 'all original257/209/173 obligations')
        current(engine)
        return frozenset()


def issue(engine, *, component_proof):
    proof = getattr(engine, '_uniform_KV_bucket409_proof', None)
    need(proof is not None and proof['cases'] == 681
         and proof['same_model_instance'] is True
         and proof['original227_FULL_math_preserved'] is True
         and engine.cache_allocation is None and engine.cache_block_tables is None
         and not engine.graph_runner.graph_cache_sealed,
         'all fresh target math completed and released before unchanged warm/seal')
    need(not hasattr(engine, '_uniform_KV_bucket409_permission'), 'issue only once')
    from scope_wide409 import local_proof
    wide_proof, wide_entries = local_proof(engine)
    engine._uniform_KV_bucket409_permission = dict(issuer=_ISSUER, engine=engine, proof=proof,
        prepare=engine._branch_tail_state.prepare, greedy=engine.graph_runner.run_target_greedy,
        wide_proof=wide_proof, wide_entries=wide_entries)
    # No device graph validation state is assigned here. Original issue checks
    # the unchanged227 proof and actual original Native graph entry objects.
    inherited = original.issue(engine, component_proof=component_proof)
    result = Scope(engine, inherited, _ISSUER)
    result.require(engine)
    return result


def report(scope, engine):
    scope.require(engine)
    result = original.report(scope.original, engine)
    result.update(schema='ROOT409.best257-plus-uniform-KV-bucket-scope.v1',
        original227_FULL2560_cases_preserved=True, additional_current_model_cases=681,
        original_best257_draft_and_branch_copy_unchanged=True,
        fixed_KV_extent=None, maximum_KV_extent=2560, KV_buckets=[512,1024,1536,2048,2560],
        all_original_warm_twozero_seal_obligations=True,
        effective_max_target_requests=32, effective_max_verification_candidates=96,
        physical_primary_plus_side_groups=32, narrow_N24_C96_S24=True,
        wide_N32_C88_S32_max24_B3=True, additional_current_S32_cases=200,
        whole_workload_AR_qualified=False, formal_Goodput=False, adopted=False)
    return result
