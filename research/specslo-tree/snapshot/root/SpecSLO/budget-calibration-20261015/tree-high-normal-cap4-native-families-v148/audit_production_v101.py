"""Independent HOST audit of original arrivals, output, budgets and graph deltas.

This module does not grant numerical qualification or infer source provenance
from a successful workload. The supervisor binds those separate actual proofs.
It imports no model/device library and performs no runtime installation.
"""
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path


MANIFEST_SHA256 = 'dc58dc5af1c5b28c6944f8f3faadeaae2dae34298c07b3ac84745d367c67d662'
NATIVE_COUNTERS = (
    'aclgraph_captures', 'aclgraph_capture_attempts', 'aclgraph_runtime_validation_replays',
    'aclgraph_failed_captures', 'aclgraph_capacity_fallbacks', 'aclgraph_shape_fallbacks',
    *(f'aclgraph_{role}_{field}' for role in ('generic', 'draft', 'target')
      for field in ('eager_fallback_calls', 'disabled_entry_calls', 'unclassified_calls',
                    'runtime_validation_calls', 'runtime_validation_failures')),
)
SLOS = {'tight': 40., 'normal': 50., 'loose': 150.}
WORKER_SCHEMA = 'V101.S35-runtime-worker.v1'
TRACE_SCHEMA = 'V101.S35-actual-trace.v1'


class AuditRejected(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditRejected(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def token_digest(tokens):
    return digest(json.dumps(tokens, separators=(',', ':')).encode())


def manifest_rows(path, count):
    raw = Path(path).read_bytes()
    require(digest(raw) == MANIFEST_SHA256, 'Original full463 manifest SHA differs')
    lines = raw.splitlines(keepends=True)
    require(len(lines) == 463 and count in (65, 463), 'Manifest scope must be original65 or full463')
    rows = [json.loads(line) for line in lines[:count]]
    require(len({r['request_id'] for r in rows}) == count, 'Duplicate manifest owner')
    require(all(type(r['prompt']) is str and r['prompt'] and r['max_tokens'] == 256
                and r['per_request_gamma'] == 4 and r['slo_class'] in SLOS
                and r['slo_tpot_ms'] == SLOS[r['slo_class']]
                and finite(r['arrival_offset_sec']) and r['arrival_offset_sec'] >= 0 for r in rows),
            'Original prompt/output/gamma/SLO/arrival contract differs')
    require([r['arrival_offset_sec'] for r in rows] == sorted(r['arrival_offset_sec'] for r in rows),
            'Manifest Poisson trace order differs')
    return rows, digest(b''.join(lines[:count]))


def native_delta(trace, zero):
    require(type(trace.get('native')) is dict, 'Explicit native before/after/delta missing')
    raw = trace['native']
    require(set(raw) == {'before', 'after', 'delta'}, 'Native snapshots have unexpected fields')
    for name in ('before', 'after', 'delta'):
        require(type(raw[name]) is dict and set(raw[name]) == set(NATIVE_COUNTERS),
                'Native discovery counter missing; implicit zero forbidden')
        require(all(type(v) is int and v >= 0 for v in raw[name].values()),
                'Native counters must be nonnegative integers')
    delta = {key: raw['after'][key]-raw['before'][key] for key in NATIVE_COUNTERS}
    require(delta == raw['delta'] and all(v >= 0 for v in delta.values()), 'Native raw delta mismatch')
    require(trace.get('forbidden_native_counters') == list(NATIVE_COUNTERS),
            'Explicit complete Native zero-discovery contract differs')
    if zero:
        require(all(v == 0 for v in delta.values()), 'Sealed/zero trace captured, validated or fell back')
    else:
        failures = [key for key in NATIVE_COUNTERS if key not in (
            'aclgraph_captures', 'aclgraph_capture_attempts', 'aclgraph_runtime_validation_replays',
            *(f'aclgraph_{role}_runtime_validation_calls' for role in ('generic', 'draft', 'target')))]
        require(all(delta[key] == 0 for key in failures), 'Warmup had actual graph failure/fallback')
    return delta


def output_arithmetic(trace, rows):
    """Rebuild every client output and SLO sum; never substitute throughput for Goodput."""
    results, events = trace.get('generate_results'), trace.get('stream_events')
    count = len(rows)
    require(type(results) is list and len(results) == count, 'Leader client result count differs')
    require(type(events) is list and events, 'Actual client stream absent')
    delivered, finished = [[] for _ in rows], [0]*count
    first_event,last_event=[None]*count,[None]*count
    previous = -1.
    for event in events:
        i = event.get('request_index')
        require(type(i) is int and 0 <= i < count and event.get('request_id') == rows[i]['request_id'],
                'Stream owner/index differs from manifest')
        values, elapsed = event.get('token_ids'), event.get('elapsed_seconds')
        require(type(values) is list and values and all(type(t) is int and t >= 0 for t in values),
                'Stream contains missing/invalid token IDs')
        require(finite(elapsed) and elapsed >= previous and type(event.get('finished')) is bool,
                'Stream time/order/finish event invalid')
        require(not finished[i], 'Output appeared after terminal event')
        previous = elapsed
        if first_event[i] is None:first_event[i]=elapsed
        last_event[i]=elapsed
        delivered[i].extend(values)
        finished[i] += int(event['finished'])
    wall = trace.get('elapsed_seconds')
    require(finite(wall) and wall > 0 and wall+1.e-6 >= previous, 'Generate timer excludes client output')
    classes = {name: dict(requests=0, attained=0, output_tokens=0, good_tokens=0) for name in SLOS}
    for i, (row, result) in enumerate(zip(rows, results, strict=True)):
        require(type(result) is dict and result.get('request_id') == row['request_id']
                and result.get('slo_class') == row['slo_class']
                and result.get('slo_tpot_ms') == row['slo_tpot_ms']
                and result.get('max_tokens') == 256 and result.get('ignore_eos') is True,
                'Per-request output/SLO policy changed')
        require(all(result.get(k) == v for k,v in dict(temperature=0., draft_temperature=0.,
                top_p=1., top_k=0, draft_top_p=1., draft_top_k=0).items()),
                'Per-request original greedy sampling changed')
        require(len(delivered[i]) == 256 and finished[i] == 1
                and result.get('completion_token_ids') == delivered[i], 'Stream/client complete256 output mismatch')
        paper, observed = result.get('paper_tpot_ms'), result.get('observed_tpot_ms')
        require(finite(paper) and paper >= 0 and finite(observed) and observed >= 0
                and math.isclose(paper, observed*255/256, rel_tol=1.e-10, abs_tol=1.e-8),
                'Original paper/observed TPOT denominator differs')
        attained = paper <= row['slo_tpot_ms']
        require(result.get('paper_slo_attained') is attained
                and result.get('paper_slo_goodput_tokens') == (256 if attained else 0),
                'Reported attainment differs from original per-request paper TPOT')
        cls = classes[row['slo_class']]
        cls['requests'] += 1
        cls['attained'] += int(attained)
        cls['output_tokens'] += 256
        cls['good_tokens'] += 256 if attained else 0
    wall_classes={name:dict(requests=0,attained=0,output_tokens=0,good_tokens=0) for name in SLOS}
    wall_rows=[]
    for i,(row,result) in enumerate(zip(rows,results,strict=True)):
        require(finite(first_event[i]) and finite(last_event[i]) and last_event[i]>=first_event[i],
            'Actual first/last existing client event time missing')
        duration=1000*(last_event[i]-first_event[i]);paper=duration/256;observed=duration/255
        attained=paper<=row['slo_tpot_ms'];cls=wall_classes[row['slo_class']]
        cls['requests']+=1;cls['attained']+=int(attained);cls['output_tokens']+=256
        cls['good_tokens']+=256 if attained else 0
        wall_rows.append(dict(request_index=i,request_id=row['request_id'],slo_class=row['slo_class'],
            first_token_event_seconds=first_event[i],last_token_event_seconds=last_event[i],
            first_to_last_token_wall_ms=duration,paper_tpot_ms=paper,observed_tpot_ms=observed,
            paper_slo_attained=attained,native_clock_paper_tpot_ms=result['paper_tpot_ms'],
            event_wall_minus_native_clock_paper_ms=paper-result['paper_tpot_ms']))
    import statistics
    wall_good=sum(c['good_tokens'] for c in wall_classes.values())
    event_wall=dict(scope='existing client stream first-token event through last-token event per owner; no added timer/callback',
        paper_denominator=256,observed_denominator=255,attained=sum(c['attained'] for c in wall_classes.values()),
        goodput_tokens_per_second=wall_good/wall,goodput_wall_denominator_seconds=wall,
        by_class=wall_classes,requests=wall_rows,
        median_event_wall_minus_native_clock_paper_ms=statistics.median(r['event_wall_minus_native_clock_paper_ms'] for r in wall_rows),
        native_clock_is_exact_event_wall_claimed=False,formal_Goodput=False)
    good = sum(c['good_tokens'] for c in classes.values())
    return dict(requests=count, output_tokens=count*256, attained=sum(c['attained'] for c in classes.values()),
                elapsed_seconds=wall, raw_tokens_per_second=count*256/wall,
                goodput_tokens_per_second=good/wall, by_class=classes,
                native_main_tpot_scope='historical Native accumulated cycle clock; not exact client event wall',
                existing_stream_event_wall=event_wall,
                output_sha256=token_digest(delivered), formal_Goodput=False,
                whole_workload_AR_equality_qualified=False)


def audit_transactions(trace, count):
    """Replay cursor/epoch/first-token transitions without trusting reported totals."""
    current, pending, submitted, publication, last_abort = None, None, False, None, None
    fallback = False
    commits, published, aborted = [], [], []
    identities, cycles = set(), set()
    tokens = trace['tokenized_inputs']
    def clocks(values):
        require(type(values) is dict and set(values) == {str(i) for i in range(count)}
                and all(finite(v) and v >= 0 for v in values.values()), 'Complete original decode clocks missing')
    for event in trace['logical_events']:
        kind = event.get('kind')
        if kind == 'reservation':
            owner, identity = event.get('owner'), event.get('identity')
            require(current is None and last_abort is None and not fallback
                    and type(owner) is int and 0 <= owner < count
                    and type(identity) is list and len(identity) == 3 and identity[0] == owner
                    and type(identity[1]) is str and identity[1]
                    and type(identity[2]) is int and identity[2] > 0 and tuple(identity) not in identities
                    and event.get('cursor') == 0 and event.get('prompt_length') == len(tokens[owner])
                    and event.get('prompt_sha256') == token_digest(tokens[owner])
                    and event.get('initial_fixture_cursor') is None and event.get('initial_fixture_scope') is None,
                    'Reservation owner/epoch/source must be unique and start at real cursor0')
            identities.add(tuple(identity))
            current = dict(owner=owner, identity=identity, cursor=0, last_cycle=-1, final=False)
        elif kind == 'plan':
            require(current is not None and pending is None and not current['final'] and not fallback,
                    'Mixed plan lacks idle current reservation')
            ids, budgets = event.get('target_ids'), event.get('budgets')
            require(type(ids) is list and 1 <= len(ids) <= 32 and len(set(ids)) == len(ids)
                    and all(type(i) is int and 0 <= i < count and i != current['owner'] for i in ids)
                    and type(budgets) is list and len(budgets) == len(ids)
                    and all(type(b) is int and 1 <= b <= 5 for b in budgets)
                    and sum(budgets) <= 96 and sum(b+1 for b in budgets) <= 128,
                    'Mixed work escaped preserved N32/C96/Q128 family')
            n, q = len(ids), sum(b+1 for b in budgets)
            scratch_count = (18-n if n<=16 else 34-n)
            prefill = min(len(tokens[current['owner']])-current['cursor'], 128, 197-q-2*scratch_count)
            base, extra = divmod(197-q-prefill, scratch_count)
            partition = [b+1 for b in budgets]+[prefill]+[base+1]*extra+[base]*(scratch_count-extra)
            cycle = event.get('cycle_id')
            require(type(cycle) is int and cycle > current['last_cycle'] and cycle not in cycles
                    and event.get('start') == current['cursor'] and event.get('end') == current['cursor']+prefill
                    and event.get('query_partition') == partition
                    and event.get('final') is (event['end'] == len(tokens[current['owner']]))
                    and len(event.get('proposal_ids', [])) == n
                    and len(set(event['proposal_ids'])) == n
                    and all(type(i) is int and i >= 0 for i in event['proposal_ids']),
                    'Mixed plan cursor/partition/proposal identity differs')
            cycles.add(cycle)
            pending, submitted = dict(event, owner=current['owner'], identity=current['identity']), False
        elif kind == 'submitted':
            require(pending is not None and not submitted and event.get('cycle_id') == pending['cycle_id']
                    and event.get('phase') == 'submitted', 'Mixed submission missing or duplicated')
            submitted = True
        elif kind == 'commit':
            require(pending is not None and submitted and all(event.get(k) == pending[k]
                    for k in ('cycle_id', 'start', 'end', 'final'))
                    and event.get('cursor') == pending['end']
                    and event.get('phase') == ('committed-final-awaiting-publication' if pending['final'] else 'reserved'),
                    'Mixed commit lost original submitted cursor/phase')
            require((type(event.get('first_token')) is int and event['first_token'] >= 0)
                    if pending['final'] else event.get('first_token') is None,
                    'Mixed final/nonfinal first-token field invalid')
            clocks(event.get('clocks_before_cycle_charge'))
            commits.append(dict(pending, first_token=event.get('first_token'),
                                clocks_before_cycle_charge=event['clocks_before_cycle_charge']))
            current.update(cursor=pending['end'], last_cycle=pending['cycle_id'], final=pending['final'])
            pending, submitted = None, False
        elif kind == 'publication-entry':
            ids = event.get('active_before')
            require(current is not None and current['final'] and publication is None
                    and event.get('owner') == current['owner'] and type(ids) is list
                    and current['owner'] not in ids and len(ids) < 64 and len(set(ids)) == len(ids)
                    and all(type(i) is int and 0 <= i < count for i in ids),
                    'First-token publication owner/capacity invalid')
            after = event.get('clocks_after_cycle_charge')
            clocks(after)
            before = commits[-1]['clocks_before_cycle_charge']
            require(after[str(current['owner'])] == before[str(current['owner'])]
                    and all(after[k] >= v for k,v in before.items())
                    and all(after[str(i)] > before[str(i)] for i in ids),
                    'Publication preceded old-active charging or charged own prompt')
            publication = event
        elif kind == 'publication':
            require(current is not None and publication is not None and event.get('owner') == current['owner']
                    and event.get('first_token') == commits[-1]['first_token']
                    and event.get('reservation_remaining') is False
                    and event.get('decode_elapsed_ms') == event.get('decode_start_elapsed_ms')
                    == publication['clocks_after_cycle_charge'][str(current['owner'])],
                    'First-token publication/cursor state invalid')
            published.append(current['owner'])
            current, publication = None, None
        elif kind == 'fallback-request':
            require(current is not None and pending is None and not fallback, 'Fallback lacks owned reservation')
            fallback = True
        elif kind == 'abort':
            require(current is not None and pending is None and publication is None
                    and event.get('owner') == current['owner']
                    and event.get('physical_invalidation_required') is (current['cursor'] > 0),
                    'Abort must match actual partial write history')
            last_abort = current
            aborted.append(current['owner'])
            current = None
        elif kind == 'fallback-reset':
            require(current is None and last_abort is not None and fallback, 'Fallback reset lacks abort')
            last_abort, fallback = None, False
        else:
            raise AuditRejected('Unexpected transaction event: '+str(kind))
    require(current is None and pending is None and publication is None and not fallback,
            'Outstanding transaction/reservation at terminal boundary')
    require(trace.get('close_abort_identities') == ([] if last_abort is None else [last_abort['identity']]),
            'Terminal abort does not match actual close boundary')
    standalone = []
    for call in trace['standalone_prefill_calls']:
        ids = call.get('sequence_ids')
        require(type(ids) is list and ids and len(set(ids)) == len(ids)
                and all(type(i) is int and 0 <= i < count for i in ids)
                and call.get('cached_tokens') == [0]*len(ids)
                and call.get('prompt_lengths') == [len(tokens[i]) for i in ids]
                and call.get('prompt_sha256') == [token_digest(tokens[i]) for i in ids]
                and call.get('uncached_queries') == sum(len(tokens[i]) for i in ids)
                and call.get('precomputed_target_tokens') is False and call.get('draft_prefill_completed') is False,
                'Standalone fallback must recompute entire genuine prompt')
        standalone.extend(ids)
    require(sorted(standalone+published) == list(range(count)) and all(i in standalone for i in aborted),
            'Every owner must have exactly one genuine first-token admission path')
    return dict(mixed_commits=len(commits), mixed_publications=len(published), aborted_reservations=len(aborted),
                standalone_prefill_batches=len(trace['standalone_prefill_calls']),
                mixed_candidates=sum(sum(c['budgets']) for c in commits),
                mixed_verification_queries=sum(sum(c['budgets'])+len(c['budgets']) for c in commits))


def audit_trace_set(traces, rows, prefix_sha256, *, zero):
    require(type(traces) is list and len(traces) == 4, 'Four actual trace records required')
    require({t.get('rank') for t in traces} == set(range(4))
            and all(type(t.get('rank')) is int for t in traces), 'Trace ranks are not exactly0..3')
    traces = sorted(traces, key=lambda t: t['rank'])
    first = traces[0]
    common = ('requests', 'tokenized_inputs', 'sampling', 'absolute_arrivals', 'origin',
              'logical_events', 'begin_inputs', 'close_abort_identities', 'standalone_prefill_calls')
    for trace in traces:
        require(trace.get('schema') == TRACE_SCHEMA and trace.get('scope_count') == len(rows)
                and trace.get('manifest_sha256') == MANIFEST_SHA256
                and trace.get('prefix_sha256') == prefix_sha256, 'Actual trace source/workload scope differs')
        require(all(key in trace and trace[key] == first[key] for key in common),
                'Four-rank actual host input/transaction trace disagreement')
        require(all(trace.get(key) is True for key in ('wrappers_restored', 'class_delegate_restored', 'cache_released')),
                'Trace runtime binding/cache cleanup incomplete')
        native_delta(trace, zero)
        if zero:
            require(trace.get('sealed_before') is trace.get('sealed_after'), 'Zero trace changed sealed state')
        if len(rows) == 463:
            require(trace.get('sealed_before') is True and trace.get('sealed_after') is True,
                    'Full463 measurement did not start and end on sealed Native cache')
    origin = first['origin']
    require(finite(origin) and first['absolute_arrivals'] == [origin+r['arrival_offset_sec'] for r in rows],
            'Arrival offsets changed or origin not shared')
    require(type(first['tokenized_inputs']) is list and len(first['tokenized_inputs']) == len(rows)
            and all(type(ts) is list and len(ts) >= 2 and all(type(t) is int and t >= 0 for t in ts)
                    for ts in first['tokenized_inputs']), 'Complete actual tokenized prompts missing')
    require(type(first['requests']) is list and len(first['requests']) == len(rows), 'Manifest projection missing')
    for i, (actual, row) in enumerate(zip(first['requests'], rows, strict=True)):
        require(actual.get('request_index') == i and actual.get('request_id') == row['request_id']
                and actual.get('prompt_sha256') == digest(row['prompt'].encode())
                and actual.get('arrival_offset_sec') == row['arrival_offset_sec']
                and actual.get('slo_class') == row['slo_class']
                and actual.get('slo_tpot_ms') == row['slo_tpot_ms']
                and actual.get('max_tokens') == 256, 'Projection does not match actual original manifest')
        sampling = first['sampling'][i]
        expected_sampling = dict(temperature=0., draft_temperature=0., top_p=1., top_k=0,
            draft_top_p=1., draft_top_k=0, max_tokens=256, ignore_eos=True,
            slo_class=row['slo_class'], slo_tpot_ms=row['slo_tpot_ms'], request_id=row['request_id'],
            spec_rhythm_max_gamma=5, arrival_ts=None)
        require(sampling == expected_sampling, 'Original sampling/maxB5/output/SLO request specification changed')
    require(traces[0].get('normal_target_calls') == [] and traces[0].get('generate_results') is None,
            'Draft cannot own target or client output')
    calls = traces[1].get('normal_target_calls')
    require(type(calls) is list and all(t.get('normal_target_calls') == calls for t in traces[1:]),
            'Three-target actual call records disagree')
    for call in calls:
        n, budgets = call.get('N'), call.get('budgets')
        require(type(n) is int and 1 <= n <= 32 and type(budgets) is list and len(budgets) == n
                and all(type(b) is int and 1 <= b <= 5 for b in budgets)
                and sum(budgets) <= 96 and sum(b+1 for b in budgets) <= 128
                and call.get('real_query_count') == sum(b+1 for b in budgets),
                'Actual target escaped unchanged N32/C96/root-inclusive128 budget')
        q = call.get('physical_query_partition')
        require(type(q) is list and q[:n] == [b+1 for b in budgets]
                and all(type(v) is int and v > 0 for v in q)
                and call.get('physical_M') == sum(q) and call.get('physical_S') == len(q),
                'Target physical query partition inconsistent')
    for trace in traces:
        counts = trace.get('decode_counters', trace.get('actual_decode_counters'))
        require(type(counts) is dict and type(counts.get('spec_rhythm_peak_active_requests')) is int
                and 1 <= counts['spec_rhythm_peak_active_requests'] <= 64, 'Actual active peak absent/over64')
        if trace['rank'] != 1:
            require(trace.get('generate_results') is None and trace.get('stream_events') == [],
                    'Only targetleader rank1 owns client output')
    transactions = audit_transactions(first, len(rows))
    expected_queries = sum(c['real_query_count'] for c in calls)+transactions['mixed_verification_queries']
    expected_candidates = sum(sum(c['budgets']) for c in calls)+transactions['mixed_candidates']
    for trace in traces:
        before, after = trace.get('policy_before'), trace.get('policy_after')
        require(type(before) is dict and type(after) is dict, 'Original before/after policy accounting absent')
        for key, expected in (
                ('dynamic_tree_target_calls', len(calls)+transactions['mixed_commits']),
                ('dynamic_tree_target_actual_queries', expected_queries)):
            require(type(before.get(key)) is int and type(after.get(key)) is int
                    and before[key] >= 0 and after[key]-before[key] == expected,
                    'Original policy accounting must include each normal/mixed call exactly once')
        counts = trace.get('decode_counters', trace.get('actual_decode_counters'))
        require(counts.get('spec_rhythm_verified_tokens') == expected_candidates
                and counts.get('spec_rhythm_prefill_requests') == len(rows)
                and counts.get('spec_rhythm_prefill_batches') == transactions['standalone_prefill_batches']
                    +transactions['mixed_publications'], 'Decode candidate/prefill accounting mismatch')
        terminal = trace.get('terminal_evidence')
        require(type(terminal) is dict and terminal.get('reservation_remaining') is False
                and terminal.get('committed_cursor') is None and terminal.get('prepared_cursor') is None,
                'Original terminal reservation/cursor not closed')
        boundary_events = trace.get('boundary_events')
        require(type(boundary_events) is list, 'Actual boundary events missing')
        sequence = 0
        for event in boundary_events:
            require(type(event) is dict and type(event.get('stage')) is str, 'Boundary record malformed')
            if 'sequence' not in event:
                continue
            require(event.get('sequence') == sequence and event.get('boundary_kind') == 'host-status-agreement'
                    and type(event.get('ranks')) is list and len(event['ranks']) == 4
                    and sorted(r.get('rank') for r in event['ranks']) == [0,1,2,3]
                    and all(r.get('error') is None and r.get('sequence') == sequence
                            and r.get('stage') == event['stage'] for r in event['ranks']),
                    'Original four-rank boundary sequence/error agreement failed')
            sequence += 1
    return dict(output=output_arithmetic(traces[1], rows),
                transactions=transactions,
                actual_peak_active=max(t.get('decode_counters', t.get('actual_decode_counters'))['spec_rhythm_peak_active_requests'] for t in traces),
                normal_target_calls=len(calls), actual_target_N_gt16=sum(c['N'] > 16 for c in calls),
                actual_capacity64_scope_completed=any(t.get('decode_counters', t.get('actual_decode_counters'))['spec_rhythm_peak_active_requests'] > 32 for t in traces),
                numerical_qualification_inferred=False)


def load_artifact(artifact, private_directory):
    root, path = Path(private_directory).resolve(), Path(artifact['path']).resolve()
    require(path.is_relative_to(root) and path.is_file(), 'Trace artifact escaped private run directory')
    raw = path.read_bytes()
    require(len(raw) == artifact['bytes'] and len(raw) <= 64*1024*1024
            and digest(raw) == artifact['sha256'], 'Actual trace artifact bytes/SHA differ')
    with gzip.open(path, 'rb') as source:
        unpacked = source.read(256*1024*1024+1)
    require(len(unpacked) <= 256*1024*1024 and len(unpacked) == artifact['uncompressed_bytes'],
            'Actual trace uncompressed length differs/exceeds bound')
    return json.loads(unpacked)


def audit_worker_reports(reports, manifest_path, private_directory):
    require(type(reports) is list and len(reports) == 4, 'Four complete worker reports required')
    require({r.get('rank') for r in reports} == set(range(4))
            and all(type(r.get('rank')) is int for r in reports), 'Worker ranks are not exactly0..3')
    reports = sorted(reports, key=lambda r: r['rank'])
    action = reports[0].get('action')
    require(action in ('s35-qualify65', 's35-full463'), 'Unknown capacity64 action')
    for report in reports:
        rank = report['rank']
        require(report.get('schema') == WORKER_SCHEMA and report.get('action') == action
                and report.get('errors') == [], 'Worker schema/action/errors incomplete')
        require(report.get('online_qualified') is False and report.get('formal_Goodput') is False,
                'Workload receipt cannot assert independent numerical/online qualification')
        # This runtime preserves the original ordinary mixed-graph flag0.
        # Its custom S19 bridge uses tree scratch, not Native's five additional
        # ordinary mixed scratch rows. 532 was only the flag1 upper bound.
        expected = dict(maxseq=64, target=32, C=96, Q=128, KV=544, storage=527)
        require(report.get('capacity_contract') == expected, 'Actual capacity64 contract differs')
        require(report.get('ordinary_mixed_graph_enabled') is False
                and report.get('actual_mixed_scratch_sequence_ids') == []
                and report.get('actual_tree_scratch_sequence_ids') == list(range(463,527)),
                'Actual flag0 tree/mixed scratch ABI differs')
        require(report.get('startup_completed') is True, 'Actual startup completion missing')
        require(report.get('inventory_counts') == (dict(draft=86, generic=28, target=0) if rank == 0
                else dict(draft=0, generic=201, target=44)), 'Actual complete finite inventory differs')
        require(report.get('sealed') is True, 'Original Native graph cache not sealed')
    from bounded_report_audit_v78 import chain_record,finite_json
    signatures=[]
    for report in reports:
        scope=report['S35_structural_scope']
        require(scope['N']==list(range(17,33)) and scope['physical_M']==197 and scope['physical_S']==35
            and scope['mask']==[35,1,128,4096] and scope['C']==96 and scope['Q']==128
            and all(scope[key] is False for key in ('arbitrary_token_or_path_exact_equivalence_claimed',
                'exact_Btuple_device_enumeration_claimed','actual_workload_KV_reconstructed',
                'independent_own_AR_qualified','online_qualified','formal_Goodput')),'actual bounded structural representative scope')
        proof=report['mixed_same_M_qualification']
        require(proof['M']==197 and proof['physical_S']==[19,35]
            and proof['configured_capacity']==64 and proof['actual_original_actor_executed'] is True
            and proof['independent_epochs'] is True and proof['candidate_KV_used_as_reference'] is False,
            'new same-process original independent actor proof missing')
        chain=load_artifact(proof['artifact'],private_directory)
        finite_json(chain)
        from shape_scope_v101 import validate_inventory
        validate_inventory(chain['actual_inventory']['before'],chain['actual_inventory']['after'],is_draft=report['rank']==0)
        signatures.append(chain_record(chain,'draft' if report['rank']==0 else 'target',rank=report['rank']))
    require(all(value==signatures[0] for value in signatures[1:]),'four-rank local verdict/publication/compaction agreement')
    traces = [r.get('traces') for r in reports]
    require(all(type(ts) is list for ts in traces), 'Actual per-rank traces absent')
    roles = [[t.get('role') for t in ts] for ts in traces]
    expected_roles = ['warmup65', 'zero65-1', 'zero65-2'] + (['full463'] if action.endswith('full463') else [])
    require(all(rs == expected_roles for rs in roles), 'Complete warmup/two zero traces/full463 roles missing')
    audited = []
    for index, role in enumerate(expected_roles):
        records = [load_artifact(ts[index]['artifact'], private_directory) for ts in traces]
        count = 463 if role == 'full463' else 65
        rows, prefix = manifest_rows(manifest_path, count)
        for report, trace in zip(reports, records, strict=True):
            require(trace.get('rank') == report['rank'], 'Artifact rank differs from envelope')
        audited.append(dict(role=role, **audit_trace_set(records, rows, prefix, zero=role != 'warmup65')))
    full = audited[-1] if action.endswith('full463') else None
    return dict(schema='V101.independent-workload-and-budget-audit.v1', passed=True,
                action=action, traces=audited, full463=full,
                full463_completed=full is not None, formal_Goodput=False,
                numerical_qualification_inferred=False, whole_workload_AR_equality_qualified=False,
                scope='Actual workload/artifact/delta/output/arithmetic audit; separate numerical, source and cleanup proofs required')
