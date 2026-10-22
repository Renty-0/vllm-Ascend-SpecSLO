"""Proposed extra scope; actual original425 and current427 NN proof required."""
from dataclasses import dataclass
import scope425 as original
import witness431

_ISSUER = object()


@dataclass(frozen=True)
class Scope:
    engine: object
    original: object
    permission: object
    issuer: object

    def require(self, engine):
        if not (self.issuer is _ISSUER and self.engine is engine
                and self.original.require(engine) == frozenset()
                and witness431.current(engine) is self.permission):
            raise RuntimeError('ROOT432 current same-engine original425 plus real427 scope required')
        return frozenset()


def issue(engine, *, component_proof):
    inherited = original.issue(engine, component_proof=component_proof)
    permission = witness431.issue(engine, inherited)
    result = Scope(engine, inherited, permission, _ISSUER)
    result.require(engine)
    return result


def report(scope, engine):
    scope.require(engine)
    value = original.report(scope.original, engine)
    value.update(schema='ROOT432.original425-plus-current427-nonempty-refill-scope.v1',
        current_multiincumbent427_cases=6, current_real_incumbents=32,
        all_allocated_incumbent_pages_and_draft_ledger_protected=True,
        global_active_limit=64, maximum_nonempty_refill=32, low_water=32, minimum_nonempty_refill=8,
        original_refill_clocks_and_incumbent_time_charge_preserved=True,
        permission_inherited_from_HOST_or_previous_model=False)
    return value
