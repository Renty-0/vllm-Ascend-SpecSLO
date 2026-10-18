"""Expose the causal294 predictor and its complete per-generation records."""
from risk294 import begin_generation, install, observe, observe_primary, prediction
from risk294 import snapshot as original_snapshot
import math


def snapshot(policy):
    value = original_snapshot(policy)
    history = value.pop('history')
    generation = value['generation']
    if generation is not None:
        history[str(generation)] = dict(classes=value['classes'], requests=value['requests'],
                                        terminal_skipped=value['terminal_skipped'])
    value['by_generation'] = history
    value['predictor'] = 'request_conditional_risk294_other_request_class_prior'
    return value


def verify_snapshot(value, generation):
    if (value['generation'] != generation or type(generation) is not int
            or set(value['by_generation']) != {str(i) for i in range(1, generation + 1)}):
        raise ValueError('ROOT299 complete actual original generation records required')
    if (value['prior_samples'], value['request_prior_samples'], value['decay']) != (12., 2., 31./32.):
        raise ValueError('ROOT299 predeclared causal predictor parameters changed')
    requests, observations = 0, 0
    for number, record in value['by_generation'].items():
        totals = {c: dict(samples=[0.]*8, hits=[0.]*8) for c in ('tight', 'normal', 'loose')}
        if set(record['classes']) != set(totals):
            raise ValueError('ROOT299 actual three-class risk records required')
        for owner, row in record['requests'].items():
            if str(int(owner)) != owner or not 0 <= int(owner) < 463:
                raise ValueError('ROOT299 actual workload owner identity required')
            cls = row['slo_class']
            if cls not in totals or type(row['observations']) is not int or row['observations'] < 0:
                raise ValueError('ROOT299 actual request risk observations required')
            for mode in ('raw', 'recent'):
                for samples, hits in zip(row[mode]['samples'], row[mode]['hits'], strict=True):
                    if not math.isfinite(samples) or not math.isfinite(hits) or not 0 <= hits <= samples:
                        raise ValueError('ROOT299 observed conditional successes must be exposed trials')
            for field in ('samples', 'hits'):
                if len(row['raw'][field]) != 8 or len(row['recent'][field]) != 8:
                    raise ValueError('ROOT299 four primary and four alternate risk sets')
                for index, count in enumerate(row['raw'][field]):
                    totals[cls][field][index] += count
            cache = row['cache']
            if cache is not None:
                curve = cache['g']
                if (len(curve) != 9 or curve[0] != 0
                        or not all(math.isfinite(g) and 0 <= g <= 5 for g in curve)
                        or not all(a <= b + 1e-12 for a, b in zip(curve, curve[1:]))):
                    raise ValueError('ROOT299 bounded monotone per-request gain curve required')
                for field in ('conditional_probabilities', 'other_request_class_probabilities'):
                    if len(cache[field]) != 8 or not all(math.isfinite(q) and 0 <= q <= 1 for q in cache[field]):
                        raise ValueError('ROOT299 valid conditional risk probabilities required')
            requests += 1
            observations += row['observations']
        if totals != record['classes']:
            raise ValueError('ROOT299 class counts must equal sum of own request raw observations')
    return dict(generations=generation, request_records=requests,
                observed_verdicts=observations, exact_class_own_data_conservation=True)
