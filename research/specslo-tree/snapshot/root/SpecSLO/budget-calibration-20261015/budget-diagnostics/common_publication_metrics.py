"""HOST arithmetic for the shared worker/engine publication TPOT definition."""
import hashlib
import json
import math


def need(value, message):
    if not value:
        raise ValueError('publication metrics: '+message)


def number(value, *, positive=False):
    need(type(value) in (int, float) and math.isfinite(value), 'finite numeric clock')
    need(value > 0 if positive else value >= 0, 'nonnegative/positive clock')
    return float(value)


def token_digest(tokens):
    return hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()


def percentile(values, probability):
    ordered = sorted(values)
    need(bool(ordered), 'nonempty percentile sample')
    position = (len(ordered)-1)*probability
    lo = int(position)
    hi = min(lo+1, len(ordered)-1)
    return ordered[lo]+(ordered[hi]-ordered[lo])*(position-lo)


def compute(requests, final_outputs, events, elapsed_seconds, *, clock_field,
            method, expected_requests=463, expected_tokens=256, clock_origin=None):
    """All final output tokens or original final-output hashes are mandatory.

    Event clocks remain in their original units/origin. Only the difference
    between a request's last and first nonempty token publication is scored.
    Finished events containing no tokens do not extend that time interval.
    """
    wall = number(elapsed_seconds, positive=True)
    need(len(requests) == expected_requests, 'complete request count')
    source = {r['request_id']:r for r in requests}
    need(len(source) == expected_requests and set(source) == set(final_outputs), 'unique exact request IDs and all final outputs')
    by_id = {key:[] for key in source}
    for event in events:
        key = event['request_id']
        need(key in by_id and type(event['token_ids']) is list, 'known event owner and token list')
        need(all(type(t) is int and t >= 0 for t in event['token_ids']), 'complete integer tokens')
        need(type(event['finished']) is bool, 'explicit final marker')
        stamp = number(event[clock_field])
        if clock_origin is not None:
            need(clock_origin <= stamp <= clock_origin+wall+1e-5, 'publication inside measured original generation clock')
        by_id[key].append((stamp, event['token_ids'], event['finished']))
    rows = []
    for key, request in source.items():
        delivered = []
        first = last = None
        previous = -math.inf
        finished = 0
        nonempty_events = 0
        for stamp, tokens, final in by_id[key]:
            need(stamp >= previous and finished == 0, 'ordered events and no publication after finish')
            previous = stamp
            if tokens:
                if first is None:
                    first = stamp
                last = stamp
                nonempty_events += 1
            delivered.extend(tokens)
            finished += int(final)
        need(finished == 1 and len(delivered) == expected_tokens and nonempty_events >= 2,
             'complete256 tokens with multiple actual publications and exactly one finish')
        wanted = final_outputs[key]
        if isinstance(wanted, list):
            need(delivered == wanted, 'events exactly reconstruct original final tokens')
        else:
            need(wanted['output_tokens'] == expected_tokens and wanted['sha256'] == token_digest(delivered),
                 'events exactly match original final token count/hash')
        threshold = number(request['slo_tpot_ms'], positive=True)
        duration = (last-first)*1000
        paper = duration/expected_tokens
        observed = duration/(expected_tokens-1)
        rows.append(dict(request_id=key, slo_class=request['slo_class'], slo_tpot_ms=threshold,
                         output_tokens=expected_tokens, paper_tpot_ms=paper, observed_tpot_ms=observed,
                         attained=paper <= threshold, first_publication=first, last_publication=last,
                         output_token_ids_sha256=token_digest(delivered)))

    def summarize(selected):
        total = sum(row['output_tokens'] for row in selected)
        good = sum(row['output_tokens'] for row in selected if row['attained'])
        return dict(requests=len(selected), attained=sum(row['attained'] for row in selected),
                    output_tokens=total, good_tokens=good, Raw=total/wall, Goodput=good/wall,
                    TPOT_paper_median_ms=percentile([r['paper_tpot_ms'] for r in selected], .5),
                    TPOT_paper_P90_ms=percentile([r['paper_tpot_ms'] for r in selected], .9),
                    TPOT_observed_median_ms=percentile([r['observed_tpot_ms'] for r in selected], .5),
                    TPOT_observed_P90_ms=percentile([r['observed_tpot_ms'] for r in selected], .9))

    return dict(schema='common-worker-engine-publication-metrics.v1', method=method,
                event_clock_field=clock_field, elapsed_seconds=wall,
                TPOT_definition='last minus first nonempty worker/engine publication; /256 paper, /255 observed',
                client_delivery_clock_claimed=False, original_elapsed_not_adjusted=True,
                all_final_outputs_reconstructed=True, all=summarize(rows),
                by_class={kind:summarize([r for r in rows if r['slo_class'] == kind])
                          for kind in sorted({r['slo_class'] for r in rows})}, rows=rows)
