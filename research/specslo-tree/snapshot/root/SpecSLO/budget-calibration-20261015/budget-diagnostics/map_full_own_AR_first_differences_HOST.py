"""Map retained first AR differences to actual publication and target geometry.

Stdlib only. This does not diagnose floating-point errors or reexecute tensors.
It verifies the complete recorded stream before locating any difference.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path


def need(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def desc(path):
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def map_publications(trace):
    requests = trace['requests']
    need(len(requests) == 463, 'original complete463 required')
    results = {r['request_id']: r['completion_token_ids'] for r in trace['generate_results']}
    need(len(results) == 463 and all(len(v) == 256 for v in results.values()), '463x256 actual completions')
    accumulated = [[] for _ in requests]
    locations = [[] for _ in requests]
    normal = []
    for event in trace['stream_events']:
        owner = event['request_index']
        need(type(owner) is int and 0 <= owner < 463, 'actual owner range')
        need(event['request_id'] == requests[owner]['request_id'], 'event request identity')
        tokens = event['token_ids']
        need(all(type(v) is int and v >= 0 for v in tokens), 'retained integer tokens')
        begin = len(accumulated[owner])
        if begin == 0:
            need(len(tokens) == 1 and event['finished'] is False, 'first publication is one actual prefill root')
            locations[owner].append(dict(kind='prefill-root', output_position=0))
        else:
            normal.append((owner, begin, event))
        accumulated[owner].extend(tokens)
        need(len(accumulated[owner]) <= 256, 'no duplicate or excessive publication')
    need(all(accumulated[i] == results[r['request_id']] for i, r in enumerate(requests)), 'all original stream outputs reconstruct actual completions')
    cursor = 0
    geometry = Counter()
    for ordinal, call in enumerate(trace['normal_target_calls']):
        n = call['N']
        need(0 < n <= 32 and len(call['budgets']) == len(call['contexts']) == n, 'actual target dimensions')
        owners = call['sequence_ids'][:n]
        need(len(set(owners)) == n, 'unique real target owners')
        subset = normal[cursor:cursor+n]
        need([e[0] for e in subset] == owners, 'every actual target owner matches complete publication order')
        for owner_row, (owner, begin, event) in enumerate(subset):
            need(call['contexts'][owner_row] == len(trace['tokenized_inputs'][owner]) + begin - 1, 'actual target context matches complete prior committed prefix')
            length = len(event['token_ids'])
            for offset, token in enumerate(event['token_ids']):
                locations[owner].append(dict(kind='target-publication', call_ordinal=ordinal,
                    owner_row=owner_row, output_position=begin+offset, delivered_chunk_offset=offset,
                    delivered_chunk_length=length, delivered_token=token, N=n,
                    budget=call['budgets'][owner_row], context=call['contexts'][owner_row],
                    family=call['family'], physical_M=call['physical_M'], physical_S=call['physical_S'],
                    real_query_count=call['real_query_count'], finished=event['finished'],
                    last_delivered_token=offset == length-1,
                    verdict_reason_not_retained=True))
        cursor += n
        geometry[(call['family'], call['physical_M'], call['physical_S'])] += 1
    need(cursor == len(normal) and all(len(v) == 256 for v in locations), 'all target calls and all256 token locations accounted once')
    return results, locations, [dict(family=k[0], M=k[1], S=k[2], calls=v) for k, v in sorted(geometry.items())]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate-only', action='store_true')
    args = parser.parse_args()
    worker_path = args.actual/'rank1-local-diagnostic.json'
    worker = load(worker_path)
    info = worker['windows']['traces'][-1]['original_trace']
    need(info['role'] == 'full463' and info['scope_count'] == 463, 'original final full463 descriptor')
    trace_path = Path(info['artifact']['path'])
    raw = trace_path.read_bytes()
    need(len(raw) == info['artifact']['bytes'] and hashlib.sha256(raw).hexdigest() == info['artifact']['sha256'], 'retained original trace byte identity')
    trace = load(trace_path)
    candidate, locations, geometry = map_publications(trace)
    output = dict(schema='ROOT.actual-full-own-AR-first-difference-location.HOST.v1',
        UTC=datetime.now(timezone.utc).isoformat(), worker=desc(worker_path), trace=desc(trace_path),
        source=desc(Path(__file__)), all463_streams_and_all_target_geometry_accounted=True,
        target_calls=len(trace['normal_target_calls']), geometry=geometry,
        candidate_only=args.candidate_only, NPU_used=False, tensor_or_root_cause_claim=False)
    if not args.candidate_only:
        folder = args.actual/'postfull-independent-FIA-AR'
        summary_path = folder/'rank1-summary.json'
        summary = load(summary_path)
        need(summary['completed'] is True and summary['reference_only_prompt_plus_own_outputs'] is True
             and summary['candidate_KV_or_tokens_seeded'] is False, 'completed independent own-history AR')
        reference = {}
        inputs = [desc(summary_path)]
        for cohort in summary['cohorts']:
            path = folder/f"rank1-cohort{cohort['ordinal']}.json.gz"
            row = load(path)
            need(row['reference_generated_on_this_rank'] is True and row['candidate_tokens_ever_used_as_inputs'] is False, 'actual own reference cohort')
            for rid, tokens in zip(row['request_ids'], row['reference_completions'], strict=True):
                need(rid not in reference and len(tokens) == 256, 'unique full independent reference')
                reference[rid] = tokens
            inputs.append(desc(path))
        need(reference.keys() == candidate.keys(), 'every original request reference present')
        rows = []
        histogram = Counter()
        for owner, request in enumerate(trace['requests']):
            rid = request['request_id']
            position = next((i for i, (c, r) in enumerate(zip(candidate[rid], reference[rid], strict=True)) if c != r), None)
            if position is not None:
                location = locations[owner][position]
                rows.append(dict(request_index=owner, request_id=rid, slo_class=request['slo_class'],
                    first_difference_zero_based=position, candidate_token=candidate[rid][position],
                    own_AR_token=reference[rid][position], location=location))
                histogram[(location['kind'], location.get('family'), location.get('physical_M'))] += 1
        need(len(rows) == summary['comparison']['mismatching_requests'], 'actual summary mismatch count independently recomputed')
        need(summary['comparison']['exact_all_tokens_equal'] == (not rows), 'truthful equality boolean')
        output.update(reference_inputs=inputs, equal_requests=463-len(rows), mismatching_requests=len(rows),
            first_differences=rows, by_first_difference_geometry=[dict(kind=k[0], family=k[1], M=k[2], requests=v) for k,v in sorted(histogram.items(), key=str)],
            limitation='Locations are retained integer evidence. They do not establish whether a difference is numerical, a KV error, a mask error, or a race; the last delivered token does not identify correction versus bonus versus clipping.')
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(output=desc(args.output), mismatching_requests=output.get('mismatching_requests'), target_calls=output['target_calls'])))


if __name__ == '__main__':
    main()
