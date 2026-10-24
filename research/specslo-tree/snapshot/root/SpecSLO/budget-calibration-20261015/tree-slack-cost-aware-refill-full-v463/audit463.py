"""Keep all numerical audits; independently replay each committed admission."""

from pathlib import Path
from types import SimpleNamespace
import gzip
import hashlib
import json
import math

import audit460 as original
from admission463 import (CREDIT_FRACTION, GLOBAL_LIMIT, MAX_NEW, MIN_NONEMPTY,
                          MODEL_SHA, credit, predicted_ms, validate_model)


def need(ok, message):
    if ok is not True:
        raise RuntimeError('ROOT463 admission audit: ' + message)


def audit_trace(trace):
    rows = trace['ROOT432_refill_witnesses']
    prefills = trace['standalone_prefill_calls']
    need(len(rows) == len(prefills) and bool(rows), 'every actual refill witness retained')
    all_old = 0
    underestimates = []
    for ordinal, (row, prefill) in enumerate(zip(rows, prefills, strict=True)):
        d = row['ROOT463_admission']
        candidates, selected, old = d['candidates'], d['selected'], d['active']
        queries = d['candidate_queries']
        need(d['generation'] == trace['source_generation'] and d['global_active_limit'] == GLOBAL_LIMIT,
             'current actual source generation and unchanged global64')
        need(d['maximum_new'] == MAX_NEW and d['credit_fraction'] == CREDIT_FRACTION
             and d['fixed_cost_model_SHA256'] == MODEL_SHA, 'declared exact policy')
        need(selected == row['admitted'] == prefill['sequence_ids']
             and old == row['incumbents'] and len(selected) <= MAX_NEW,
             'same original actual prefill and incumbent membership')
        need(bool(selected) and selected == candidates[:len(selected)]
             and len(candidates) <= MAX_NEW and len(candidates) == len(set(candidates))
             and len(old) + len(selected) <= GLOBAL_LIMIT and not set(old).intersection(candidates),
             'whole original priority prefix and bounded owners')
        need(len(queries) == len(candidates) == len(d['candidate_arrivals']), 'complete considered prefix')
        need(d['arrival_clock'] == row['original_arrival_clock'], 'original arrival broadcast clock retained')
        for owner, amount, arrival in zip(candidates, queries, d['candidate_arrivals'], strict=True):
            need(arrival == trace['absolute_arrivals'][owner] and arrival <= d['arrival_clock'],
                 'no future arrival or substituted clock')
            need(amount == len(trace['tokenized_inputs'][owner]), 'actual uncached ordinary prompt length')
        available = []
        need([r['owner'] for r in d['incumbents']] == old, 'same complete old-owner order')
        for state, owner, elapsed in zip(d['incumbents'], old, row['incumbent_decode_before_ms'], strict=True):
            need(state['slo_ms'] == trace['sampling'][owner]['slo_tpot_ms']
                 and state['decode_elapsed_ms'] == elapsed, 'actual original SLO and elapsed fields')
            need(type(state['delivered_tokens']) is int and 1 <= state['delivered_tokens'] <= 256,
                 'actual finite committed output count')
            value = credit(SimpleNamespace(slo_tpot_ms=state['slo_ms']), SimpleNamespace(**state))
            need(value == state['credit_ms'], 'credit independently recomputed')
            if value is not None:
                available.append(value)
        limit = max(0.0, min(available)) if available else None
        need(d['available_pause_ms'] == limit, 'all protected incumbent constraints retained')
        if old:
            valid = [n for n in range(MIN_NONEMPTY, len(candidates) + 1)
                     if limit is None or predicted_ms(sum(queries[:n])) <= limit]
            expected = max(valid, default=0)
        else:
            expected = len(candidates)
        need(len(selected) == expected, 'largest fitting original prefix, no undeclared extra admission')
        predicted = predicted_ms(sum(queries[:len(selected)]))
        need(predicted == d['predicted_prefill_ms'] and sum(queries[:len(selected)]) == prefill['uncached_queries'],
             'same actually executed ordinary prompt work')
        need(math.isfinite(row['prefill_ms']) and row['prefill_ms'] >= 0, 'actual duration retained')
        if row['prefill_ms'] > predicted:
            underestimates.append(dict(ordinal=ordinal,incumbents=len(old),
                                       predicted_ms=predicted,actual_ms=row['prefill_ms']))
        all_old += len(old)
    return dict(committed_refills=len(rows), incumbent_exposures=all_old,
                all_original_priority_prefixes_and_credits_replayed=True,
                fixed_model_SHA256=MODEL_SHA, all_prediction_underestimates=underestimates,
                global_active_limit=GLOBAL_LIMIT, maximum_new_cohort=MAX_NEW,
                zero_nonempty_refills_would_not_grant_or_remove_NN_permission=True)


def audit(directory):
    result = original.audit(directory)
    controls = []
    for row in result['ROOT460_actual_adaptive_geometry_histories']:
        path = Path(row['path'])
        data = path.read_bytes()
        need(hashlib.sha256(data).hexdigest() == row['SHA256'], 'same original complete trace bytes')
        trace = json.loads(gzip.decompress(data))
        controls.append(dict(path=str(path), SHA256=row['SHA256'], admission=audit_trace(trace)))
    result['ROOT463_actual_credit_cost_admission'] = controls
    result['ROOT463_fixed_prefill_model'] = validate_model()
    result['ROOT463_all_original_NN_AR_clocks_and_cleanup_obligations_retained'] = True
    return result
