"""Add draft-tail obligations while delegating all original257 scope checks."""
from dataclasses import dataclass, field

import scope257 as original

_ISSUER = object()


def need(value, message):
    if value is not True:
        raise RuntimeError('ROOT372 scope: ' + message)


@dataclass(frozen=True)
class Scope:
    engine: object = field(repr=False)
    original: object = field(repr=False)
    qualification: object = field(repr=False)
    issuer: object = field(repr=False)

    def require(self, engine):
        need(self.issuer is _ISSUER and engine is self.engine, 'same issued372 engine')
        need(self.original.require(engine) == frozenset(), 'all original257/209/173 scope obligations')
        if engine.is_draft:
            q = self.qualification
            need(q is getattr(engine, '_ROOT372_qualified_draft_tail', None)
                 and q['proof']['cases'] == 306 and q['proof']['fresh_same_model_instance'] is True,
                 'same current306 neural proof')
            from vllm_ascend.spec_decode.pearl import native_engine as native
            need(native.NativePearlEngine.draft_tree_forward is q['forward']
                 and engine.graph_runner.run_draft_greedy.__func__ is q['run_draft_func']
                 and engine.graph_runner._execute_draft.__func__ is q['execute_draft_func'],
                 'same owner-checked draft and private graph callbacks')
            need(len(q['entries']) == 64, 'all logicalN1to32 normal/full private graph families')
            for key, entry in q['entries'].items():
                need(engine.graph_runner.draft_entries.get(key) is entry and entry.runtime_validated is True
                     and key not in engine.graph_runner.disabled_entry_keys,
                     'same naturally validated private entry objects')
        return frozenset()


def issue(engine, *, component_proof):
    inherited = original.issue(engine, component_proof=component_proof)
    qualification = getattr(engine, '_ROOT372_qualified_draft_tail', None)
    if engine.is_draft:
        need(qualification is not None and engine.cache_allocation is None
             and engine.cache_block_tables is None and not engine.graph_runner.graph_cache_sealed,
             'fresh draft math released before unchanged original warm and sealing')
    result = Scope(engine, inherited, qualification, _ISSUER)
    result.require(engine)
    return result


def report(scope, engine):
    scope.require(engine)
    value = original.report(scope.original, engine)
    value.update(schema='ROOT372.current-target257-plus-fresh-draft-tail3-scope.v1',
        original_target257_and_all_prior_obligations_preserved=True,
        fresh_same_instance_draft306_required=True,
        new_draft_allN1to32_normal_and_eager_families=True,
        Native_qualification_flag_assignment=False,
        original_warm_twozero_and_sealing_required=True,
        whole_workload_AR_qualified=False, formal_Goodput=False, adopted=False)
    return value
