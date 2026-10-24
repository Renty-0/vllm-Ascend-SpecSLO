"""Select an arrived prefix using incumbent credit and a fixed cost model."""

from pathlib import Path
import hashlib
import json
import math

GLOBAL_LIMIT = 64
MAX_NEW = 32
MIN_NONEMPTY = 8
CREDIT_FRACTION = 0.95
MODEL_SHA = 'f0d54b1e692f02d5b7ead5aec9ca2973cc98827254f5830698124771519ad8fc'
INTERCEPT_MS = 31.1820130301752
SLOPE_MS = 0.13797625060353721
COST_MARGIN = 1.10


def validate_model():
    path = Path(__file__).resolve().parent.parent / 'budget-diagnostics/ROOT462_PROSPECTIVE_PREFILL_COST_MODEL_HOST.json'
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MODEL_SHA:
        raise RuntimeError('Exact predeclared prefill cost model required')
    value = json.loads(raw)
    if (value['intercept_ms'], value['slope_ms_per_prompt_query'], value['safety_multiplier']) != (INTERCEPT_MS, SLOPE_MS, COST_MARGIN):
        raise RuntimeError('Unchanged cost parameters required')
    return dict(path=str(path), SHA256=MODEL_SHA, NN_permission=False)


def predicted_ms(queries):
    if type(queries) is not int or queries < 0:
        raise ValueError('Nonnegative actual uncached prompt queries required')
    # There is no prefill helper invocation when every selected owner is
    # already prefetched. The regression formula applies to positive queries.
    return 0.0 if queries == 0 else COST_MARGIN * (max(0.0, INTERCEPT_MS) + SLOPE_MS * queries)


def credit(params, state):
    slo = params.slo_tpot_ms
    if slo is None:
        return None
    elapsed = float(state.decode_elapsed_ms) - float(state.decode_start_elapsed_ms)
    delivered = state.delivered_tokens
    if (not math.isfinite(float(slo)) or slo <= 0 or not math.isfinite(elapsed)
            or elapsed < -1e-8 or type(delivered) is not int or delivered < 1):
        raise ValueError('Finite original incumbent decode accounting required')
    return CREDIT_FRACTION * float(slo) * max(0, delivered - 1) - max(0.0, elapsed)


def choose(engine, admitted, *, active, request_params, states, local_states,
           prefetched, now, counters):
    candidates = list(admitted[:MAX_NEW])
    old = list(active)
    if (len(old) != len(set(old)) or len(candidates) != len(set(candidates))
            or set(old).intersection(candidates) or len(old) + len(candidates) > GLOBAL_LIMIT):
        raise ValueError('Original unique owners within global64 required')
    if any(request_params[i].arrival_ts is not None and request_params[i].arrival_ts > now for i in candidates):
        raise ValueError('Only already arrived candidates may be considered')
    generation = getattr(engine, '_dynamic_tree_generation', None)
    if type(generation) is not int or generation <= 0:
        raise RuntimeError('Original positive source generation required')
    incumbents = []
    for i in old:
        available = credit(request_params[i], states[i])
        incumbents.append(dict(owner=i, slo_ms=request_params[i].slo_tpot_ms,
            delivered_tokens=states[i].delivered_tokens,
            decode_elapsed_ms=states[i].decode_elapsed_ms,
            decode_start_elapsed_ms=states[i].decode_start_elapsed_ms,
            credit_ms=available))
    constrained = [r['credit_ms'] for r in incumbents if r['credit_ms'] is not None]
    available = max(0.0, min(constrained)) if constrained else None
    queries = [0 if i in prefetched else int(local_states[i].prompt_length) for i in candidates]
    if any(q < 0 for q in queries):
        raise ValueError('Original nonnegative prompt lengths required')
    selected_count = len(candidates) if not old else 0
    if old:
        for n in range(MIN_NONEMPTY, len(candidates) + 1):
            if available is None or predicted_ms(sum(queries[:n])) <= available:
                selected_count = n
    selected = candidates[:selected_count]
    counters['spec_rhythm_ROOT463_admission_polls'] = counters.get('spec_rhythm_ROOT463_admission_polls', 0) + 1
    if old and selected_count < len(candidates):
        counters['spec_rhythm_ROOT463_cost_limited_polls'] = counters.get('spec_rhythm_ROOT463_cost_limited_polls', 0) + 1
    engine._root463_pending_admission = None
    if selected:
        engine._root463_pending_admission = dict(generation=generation,
            candidates=candidates, candidate_queries=queries,
            candidate_arrivals=[request_params[i].arrival_ts for i in candidates],
            selected=selected, active=old, incumbents=incumbents,
            available_pause_ms=available, predicted_prefill_ms=predicted_ms(sum(queries[:selected_count])),
            arrival_clock=now, credit_fraction=CREDIT_FRACTION, maximum_new=MAX_NEW,
            global_active_limit=GLOBAL_LIMIT, fixed_cost_model_SHA256=MODEL_SHA)
    return selected
