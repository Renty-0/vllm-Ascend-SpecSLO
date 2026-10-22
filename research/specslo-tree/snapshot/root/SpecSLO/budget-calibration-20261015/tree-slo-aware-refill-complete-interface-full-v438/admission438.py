"""Current arrived incumbents determine the refill threshold; global cap64."""
STRICT_TPOT_MS=40.0
STRICT_LOW_WATER=16
OTHER_LOW_WATER=32


def low_water(request_params,active):
    return STRICT_LOW_WATER if any(request_params[i].slo_tpot_ms is not None
        and request_params[i].slo_tpot_ms<=STRICT_TPOT_MS for i in active)else OTHER_LOW_WATER


def audit_trace(trace):
    from admission_audit432 import audit_trace as inherited
    value=inherited(trace)
    for row in value['cohorts']:
        witness=trace['ROOT432_refill_witnesses'][row['ordinal']]
        old=witness['incumbents']
        strict=any(trace['sampling'][i]['slo_tpot_ms']is not None
            and trace['sampling'][i]['slo_tpot_ms']<=40.0 for i in old)
        if len(old)>(16 if strict else 32):
            raise RuntimeError('ROOT438 actual SLO-aware incumbent refill threshold')
    return dict(value,current_SLO_aware_refill_policy=True,strict_ms_threshold=40.0,
        strict_low_water=16,otherwise_low_water=32,global_active_limit=64,
        future_arrivals_or_outcomes_used=False)
