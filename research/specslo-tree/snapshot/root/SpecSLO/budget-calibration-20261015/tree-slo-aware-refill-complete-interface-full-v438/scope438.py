"""The same actual432 NN scope; truthfully describe current scheduling."""
from scope432 import Scope,issue
from scope432 import report as original


def report(scope,engine):
    value=original(scope,engine)
    value.update(schema='ROOT438.current-SLO-refill-with-original432-NN-scope.v1',
        low_water_policy438=dict(strict_ms_threshold=40.0,strict_incumbent_limit=16,
            otherwise_incumbent_limit=32,global_active_limit=64,
            empty_admission_limit=64,minimum_nonempty8=True,maximum_nonempty32=True),
        original_NN_incumbent_domain_upper_bound32=True,future_arrivals_or_outcomes_used=False)
    return value
