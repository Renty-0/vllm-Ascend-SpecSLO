#!/usr/bin/env python3
"""Independent HOST audit of terminal V85 candidate/partial-AR mismatch evidence.

Run only with the retained Python3.11 after the supervisor is terminal. Missing
terminal/raw evidence rejects without creating output. The original job status
is preserved separately from this independent audit. No candidate file, raw
record, archive or receipt is rewritten. Output is exclusive and mode0600.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
import re
from pathlib import Path
import sys
import tarfile


BASE = Path('/root/SpecSLO/budget-calibration-20261015')
CANDIDATE = BASE/'tree-live-prefix-page-protection-v85'
TAG = 'live-poisson48-page-prefix-v1'
ACTION = 'live-poisson-probe'
RECEIPT = CANDIDATE/('local-receipt-'+TAG+'.json')
RAW_DIRECTORY = CANDIDATE/'local-output'/TAG
FREEZE = CANDIDATE/'PREPARED_SOURCE_HASHES_V85.json'
CPU = CANDIDATE/'CPU_RESULT_V85.json'
PRIOR = CANDIDATE/'ORIGINAL_PRIOR_BINDING_V78.json'
OUTPUT = BASE/'budget-diagnostics/ROOT_V85_ACTUAL_CANDIDATE_AR_MISMATCH_AUDIT.json'
RETAINED_PYTHON = Path('/root/miniconda3/envs/vllm-hust-dev/bin/python3.11')
ROOT_PIN = BASE/'budget-diagnostics/ROOT_V85_FINAL_CPU_AND_SOURCE_PIN_AUDIT.json'



class AuditRejected(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise AuditRejected(message)


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_bytes())


def map_digest(mapping):
    return sha_bytes(json.dumps(mapping, sort_keys=True, separators=(',', ':')).encode())


def verify_files(mapping, label, *, local=False):
    require(type(mapping) is dict and bool(mapping), label+' hash map is absent')
    bad = []
    for name, expected in sorted(mapping.items()):
        require(type(name) is str and Path(name).is_absolute()
            and type(expected) is str and len(expected) == 64, label+' malformed path/hash')
        if local:
            require(Path(name).resolve().parent == CANDIDATE, label+' source outside candidate: '+name)
        try:
            if sha_file(name) != expected:
                bad.append(name)
        except OSError:
            bad.append(name)
    require(not bad, label+' changed/missing files: '+repr(bad[:12]))
    return dict(files_checked=len(mapping), map_sha256=map_digest(mapping), exact=True)


def verify_archive(path, expected_sha, sources):
    path = Path(path)
    require(sha_file(path) == expected_sha, 'Archive SHA differs: '+str(path))
    wanted = {Path(name).name: expected for name, expected in sources.items()}
    require(len(wanted) == len(sources), 'Archive input basenames alias')
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        require(len(members) == len(wanted)
            and len({member.name for member in members}) == len(members)
            and all(member.isfile() and member.name in wanted for member in members),
            'Archive exact regular-file membership differs: '+str(path))
        for member in members:
            digest = hashlib.sha256()
            with archive.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            require(digest.hexdigest() == wanted[member.name], 'Archive bytes differ: '+member.name)
    return dict(path=str(path), sha256=expected_sha, members=len(wanted), exact=True)



def load_root_pin():
    """The fixed root audit is the explicit trusted identity input, not a launch."""
    raw = ROOT_PIN.read_bytes()
    pin = json.loads(raw)
    require(pin.get('schema') == 'V85.actual-final-CPU-and-source-pin-audit.v1'
        and pin.get('all_checks_passed') is True and pin.get('formal_Goodput') is False
        and pin.get('NPU_initialized') is False
        and type(pin.get('checks')) is dict and bool(pin['checks']),
        'Trusted root final HOST/source audit is absent or rejected')
    identity = pin.get('identity')
    keys = {'freeze_sha256', 'CPU_sha256', 'original_prior_binding_sha256'}
    require(type(identity) is dict and set(identity) == keys
        and all(type(value) is str and len(value) == 64
            and all(char in '0123456789abcdef' for char in value) for value in identity.values()),
        'Trusted root final audit has no exact readiness identity')
    return raw, pin


def current_candidate_sources():
    # Match the exact frozen checker selection, without calling its launch or
    # readiness collectors. Runtime records reside below their own directories.
    excluded = {FREEZE.name, CPU.name, 'SOURCE_ONLY_FREEZE_V85.json',
                'SOURCE_PREPARATION_SNAPSHOT_V85.json'}
    return {str(path.resolve()):sha_file(path) for path in sorted(CANDIDATE.iterdir())
        if path.is_file() and path.suffix in ('.py', '.sh', '.json')
        and path.name not in excluded
        and not path.name.startswith(('local-receipt-', 'cost-receipt-'))}


def verify_integrity(receipt, pin):
    identity = pin['identity']
    require(receipt['readiness']['identity'] == identity, 'Supervisor/root source identity differs')
    for path, key in ((FREEZE, 'freeze_sha256'), (CPU, 'CPU_sha256'),
                      (PRIOR, 'original_prior_binding_sha256')):
        require(sha_file(path) == identity[key], 'Pinned identity bytes differ: '+str(path))
    freeze, cpu, prior = read_json(FREEZE), read_json(CPU), read_json(PRIOR)
    require(freeze.get('schema') == 'V85.physical-page-prefix-equivalence-local-prepared-source.v1'
        and freeze.get('CPU_sha256') == identity['CPU_sha256']
        and freeze.get('original_prior_binding_sha256') == identity['original_prior_binding_sha256']
        and freeze.get('online_qualified') is False
        and freeze.get('local_numeric_qualified') is False, 'Prepared freeze contract differs')
    require(cpu.get('phase') == 'final-local-source-before-NPU' and cpu.get('status') == 'passed'
        and type(cpu.get('tests_run')) is int and cpu['tests_run'] > 0
        and cpu.get('failures') == 0 and cpu.get('errors') == 0
        and cpu.get('NPU_initialized') is False and cpu.get('Python_dont_write_bytecode') is True
        and cpu.get('model_initialization_attempts') == []
        and cpu.get('NPU_initialization_attempts') == []
        and cpu.get('builder_and_compile_passed') is True
        and type(cpu.get('modules')) is list and len(cpu['modules']) == 29
        and len(set(cpu['modules'])) == 29 and all(type(name) is str for name in cpu['modules']),
        'Actual final HOST/29modules/no-model/no-NPU receipt differs')
    require(freeze['files'] == cpu['input_hashes'] == cpu['input_hashes_after']
        == current_candidate_sources(), 'Frozen/current/before/after local source maps differ')
    require(cpu['external_input_hashes'] == cpu['external_input_hashes_after'],
        'CPU external source closure changed')
    require(receipt['readiness'].get('CPU_tests_run') == cpu['tests_run'],
        'Supervisor did not use this actual final HOST test count')
    optional = {'CPU_path':str(CPU), 'CPU_sha256':identity['CPU_sha256'],
        'freeze_path':str(FREEZE), 'freeze_sha256':identity['freeze_sha256'],
        'local_source_count':len(freeze['files']),
        'external_input_count':len(cpu['external_input_hashes'])}
    for key, value in optional.items():
        require(key not in pin or pin[key] == value, 'Trusted root pin field differs: '+key)
    expected_sources = dict(freeze['files'])
    expected_sources.update({str(FREEZE):identity['freeze_sha256'],
        str(CPU):identity['CPU_sha256'], str(PRIOR):identity['original_prior_binding_sha256']})
    require(receipt['source_hashes'] == expected_sources, 'Supervisor archive source map differs')
    require(type(prior.get('observed')) is dict and bool(prior['observed']),
        'Original prior source closure is absent')
    shared = set(prior['observed']) & set(cpu['external_input_hashes'])
    require(all(prior['observed'][path] == cpu['external_input_hashes'][path] for path in shared),
        'Original prior and CPU closure disagree on shared inputs')
    expected_external = {**prior['observed'], **cpu['external_input_hashes']}
    require(receipt['external_input_hashes'] == expected_external, 'Supervisor external closure differs')
    checks = dict(local_sources=verify_files(expected_sources, 'Prepared local source', local=True),
        CPU_external_sources=verify_files(cpu['external_input_hashes'], 'CPU external source'),
        supervisor_external_sources=verify_files(expected_external, 'Supervisor external source'),
        actual_counts=dict(local_source_files=len(freeze['files']),
            supervisor_source_files=len(expected_sources), CPU_external_files=len(cpu['external_input_hashes']),
            supervisor_external_files=len(expected_external), CPU_modules=len(cpu['modules']),
            CPU_tests_run=cpu['tests_run']))
    snapshot = CANDIDATE/'source-snapshots'/('local-'+TAG+'.tar.gz')
    require(Path(receipt['source_snapshot']) == snapshot, 'Unexpected source snapshot path')
    checks['supervisor_source_archive'] = verify_archive(snapshot,
        receipt['source_snapshot_sha256'], expected_sources)
    cpu_archive = Path(cpu['actual_local_source_archive'])
    require(cpu_archive.is_relative_to(CANDIDATE/'host-checks'), 'CPU archive outside retained host attempt')
    checks['CPU_source_archive'] = verify_archive(cpu_archive,
        cpu['actual_local_source_archive_sha256'], freeze['files'])
    checks['CPU_receipt'] = dict(path=str(CPU), sha256=identity['CPU_sha256'],
        tests_run=cpu['tests_run'], modules=cpu['modules'], NPU_initialized=False, exact=True)
    return checks, expected_sources, expected_external, cpu, freeze


def host_firewall(frame, event, argument):
    if event == 'call':
        module = frame.f_globals.get('__name__', '')
        if (frame.f_code.co_name == '__init__' and module.endswith('.native_engine')
                and type(frame.f_locals.get('self')).__name__ == 'NativePearlEngine'):
            raise AuditRejected('HOST audit forbids model construction')
        if module.startswith('torch_npu.npu') and frame.f_code.co_name in ('init', '_lazy_init'):
            raise AuditRejected('HOST audit forbids device initialization')
    elif event == 'c_call' and getattr(argument, '__name__', '') == '_npu_init':
        raise AuditRejected('HOST audit forbids device initialization')


def write_exclusive(report):
    raw = (json.dumps(report, indent=2, allow_nan=False)+'\n').encode()
    descriptor = os.open(OUTPUT, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)




PROGRESS = CANDIDATE/'live-progress'
CANDIDATE_PATHS = [PROGRESS/f'rank{rank}-candidate-before-AR.json' for rank in range(4)]


def partial_preconditions():
    require(len(sys.argv) == 1, 'Fixed V85 partial-failure audit accepts no arguments')
    require(sys.version_info[:2] == (3, 11)
        and Path(sys.executable).resolve() == RETAINED_PYTHON.resolve() and sys.dont_write_bytecode,
        'Use retained Python3.11 with -B')
    require(not OUTPUT.exists(), 'Refusing to overwrite '+str(OUTPUT))
    receipt_raw = RECEIPT.read_bytes()
    receipt = json.loads(receipt_raw)
    require(receipt.get('schema') == 'V78.local-diagnostic-supervisor.v1'
        and receipt.get('action') == ACTION and receipt.get('status') in ('complete', 'failed'),
        'Actual supervisor is not terminal; no output created')
    require(all(path.is_file() for path in CANDIDATE_PATHS)
        and all((PROGRESS/f'rank{rank}-AR-request0.json').is_file() for rank in (1, 2, 3)),
        'Actual four candidate snapshots and target3 AR0 records are required')
    pin_raw, pin = load_root_pin()
    paths = CANDIDATE_PATHS+sorted(PROGRESS.glob('rank*-AR-request*.json'))
    require(len(paths) == len(set(paths)), 'Partial input paths alias')
    raw = {path:path.read_bytes() for path in paths}
    return receipt_raw, receipt, pin_raw, pin, raw


def cleanup_evidence(receipt):
    require(receipt['status'] == 'failed' and type(receipt.get('errors')) is list
        and any('KeyboardInterrupt' in value for value in receipt['errors'])
        and type(receipt.get('child', {}).get('exit_code')) is int
        and receipt['child']['exit_code'] != 0,
        'Preserved failed/KeyboardInterrupt/nonzero-child evidence differs')
    require(receipt.get('physical_cards') == [2, 5, 6, 7]
        and receipt.get('owned_remaining') == []
        and receipt.get('devices_after', {}).get('all_idle') is True
        and receipt.get('experimental_sources_unchanged') is True
        and receipt.get('external_sources_unchanged') is True
        and receipt.get('formal_Goodput') is False and receipt.get('online_qualified') is False,
        'Actual terminal scope/source/owned/idle evidence is incomplete')
    for name in ('pre_verifier', 'post_verifier'):
        code = receipt.get(name, {}).get('exit_code')
        require(type(code) is int and code == 0, 'Original '+name+' did not pass')
    devices = receipt['devices_after']
    require(devices.get('four_910B2') is True
        and all(devices.get('processes', {}).get(str(card), {}).get('idle') is True
            and devices['processes'][str(card)].get('exit_code') == 0 for card in (2, 5, 6, 7)),
        'Original per-card post-cleanup process receipts do not establish idle')
    worker_path = RAW_DIRECTORY/'worker-exit.json'
    worker = dict(path=str(worker_path), present=worker_path.is_file())
    if worker['present']:
        raw = worker_path.read_bytes()
        worker.update(sha256=sha_bytes(raw), original_record=json.loads(raw))
    else:
        worker['interpretation'] = 'No graceful worker-exit record exists; closed/all_exited are not inferred'
    return dict(owned_remaining=[], devices_after=devices,
        pre_verifier=receipt['pre_verifier'], post_verifier=receipt['post_verifier'],
        experimental_sources_unchanged=True, external_sources_unchanged=True,
        worker_exit_record=worker, cleanup_from_original_receipt=True,
        original_worker_graceful_shutdown_not_assumed=True)


def load_helpers(cpu):
    sys.path.insert(0, str(CANDIDATE))
    module = importlib.import_module('live_poisson_probe')
    local = importlib.import_module('local_mixed_qualification')
    startup = importlib.import_module('startup_receipts_v75')
    checker = importlib.import_module('check_v75')
    for value, name in ((module, 'live_poisson_probe.py'), (local, 'local_mixed_qualification.py'),
                        (startup, 'startup_receipts_v75.py'), (checker, 'check_v75.py')):
        require(Path(value.__file__).resolve() == CANDIDATE/name, 'Unexpected frozen helper import: '+name)
    require(list(checker.CPU_MODULES) == cpu['modules'], 'Actual CPU module membership differs')
    code = compile((CANDIDATE/'live_poisson_probe.py').read_bytes(),
        str(CANDIDATE/'live_poisson_probe.py'), 'exec', dont_inherit=True)
    for name in ('_audit_logical', 'require_live_poisson_source'):
        expected = [value for value in code.co_consts
            if type(value) is type(code) and value.co_name == name]
        require(len(expected) == 1 and getattr(module, name).__code__ == expected[0],
            'Frozen HOST helper code differs: '+name)
    return module, local, startup


def token_digest(tokens):
    return sha_bytes(json.dumps(list(tokens), separators=(',', ':')).encode())


def candidate_evidence(snapshots, module, local, startup):
    source = module.require_live_poisson_source()
    original = module.original
    rows = original._manifest_rows(original.MANIFEST.read_bytes())
    first = snapshots[0]
    common = ('source', 'allowed_partitions', 'tokenized_inputs', 'sampling', 'absolute_arrivals',
        'logical_events', 'begin_inputs', 'standalone_prefill_calls', 'close_abort_identities')
    logical = None
    native = {}
    stage_counts = None
    generic_first = None
    for rank, snapshot in enumerate(snapshots):
        require(snapshot.get('rank') == rank and type(snapshot.get('rank')) is int
            and snapshot.get('schema') == module.SCHEMA
            and snapshot.get('status') == 'actual-candidate-complete-reference-pending'
            and snapshot.get('source') == source
            and snapshot.get('allowed_partitions') == module._scope(module.PARTITIONS)
            and snapshot.get('decode_entry_count') == 1 and snapshot.get('observation_errors') == []
            and all(snapshot.get(name) is True for name in ('wrappers_restored',
                'class_delegate_restored', 'candidate_cache_released'))
            and all(snapshot.get(name) is False for name in ('numerical_qualification_inferred',
                'all65_qualified', 'full463_qualified', 'online_qualified', 'cost_or_Goodput'))
            and all(snapshot[key] == first[key] for key in common),
            'Actual candidate source/rank/all4 common/control/cleanup evidence differs: '+str(rank))
        origin = snapshot['arrival_origin']
        require(origin['entry_barriers'] == origin['origin_broadcasts'] == 1
            and all(original._finite(origin[key]) for key in
                ('leader_origin', 'local_before_broadcast', 'local_after_broadcast'))
            and origin['local_before_broadcast'] <= origin['local_after_broadcast']
            and origin['leader_origin'] == first['arrival_origin']['leader_origin']
            and snapshot['absolute_arrivals'] == [origin['leader_origin']+row['arrival_offset_sec'] for row in rows],
            'Actual decode-entry arrival origin differs: '+str(rank))
        if rank == 1:
            require(origin['local_before_broadcast'] <= origin['leader_origin'] <= origin['local_after_broadcast'],
                'Actual leader arrival origin lies outside its sampled entry')
        terminal = snapshot['terminal_evidence']
        require(terminal['reservation_remaining'] is False and terminal['committed_cursor'] is None
            and terminal['prepared_cursor'] is None
            and terminal['qualification_not_inferred_from_runtime_state'] is True
            and terminal['boundary_events'] == snapshot['boundary_events'], 'Actual transaction is not terminal')
        current = module._audit_logical(snapshot)
        if logical is None:
            logical = current
        require(current == logical, 'Frozen logical/clock audit differs across ranks')
        delta = local.native_delta(snapshot['native_before'], snapshot['native_after'])
        require(snapshot['native_delta'] == delta and all(delta[key] == 0 for key in startup.FAILURE_COUNTERS),
            'Actual original Native failure/fallback/validation delta is nonzero')
        native[str(rank)] = dict(before=snapshot['native_before'], after=snapshot['native_after'],
            delta=delta, failure_counter_delta={key:delta[key] for key in startup.FAILURE_COUNTERS})
        generic = []
        for event in snapshot['boundary_events']:
            if 'sequence' not in event:
                continue
            seq = len(generic)
            require(event['sequence'] == seq and type(event['ranks']) is list and len(event['ranks']) == 4
                and sorted(row['rank'] for row in event['ranks']) == list(range(4))
                and all(type(row['rank']) is int and row['sequence'] == seq and row['stage'] == event['stage']
                    and row['error'] is None for row in event['ranks'])
                and event['boundary_kind'] == 'host-status-agreement', 'Actual host boundary error/order differs')
            generic.append(event)
        if generic_first is None:
            generic_first = generic
            stage_counts = dict(Counter(event['stage'] for event in generic))
        require(generic == generic_first, 'Actual generic boundary agreement differs across ranks')
        footprints = [event for event in snapshot['boundary_events'] if event['stage'] == 'live-query-write-footprint-bound']
        require(len(footprints) == len(current['commits']), 'Actual committed write footprints are missing')
        for footprint, commit in zip(footprints, current['commits'], strict=True):
            require(footprint['cycle_id'] == commit['cycle_id']
                and footprint['query_write_count'] == (commit['end']-commit['start'] if rank == 0 else 197)
                and footprint['runtime_KV_bytes_compared'] is False,
                'Actual role-specific write footprint differs')
        counters = snapshot['actual_decode_counters']
        require(all(type(counters.get(key)) is int and counters[key] >= 0
            and counters[key] == first['actual_decode_counters'][key] for key in original.LOGICAL_COUNTERS)
            and counters['spec_rhythm_tree_rounds'] > 0
            and 1 <= counters['spec_rhythm_peak_active_requests'] <= 32
            and counters['spec_rhythm_live_admitted_requests'] == counters['spec_rhythm_aborted_requests'] == 0,
            'Actual all4 original logical counters differ')
        if rank == 0:
            require(snapshot['normal_target_calls'] == [], 'Draft claims target-only forwards')
        else:
            original._audit_target_calls(snapshot['normal_target_calls'])
        if rank != 1:
            require(snapshot['generate_results'] is None and snapshot['stream_events'] == [],
                'Nonleader claims client output')
    normal = snapshots[1]['normal_target_calls']
    require(all(snapshots[rank]['normal_target_calls'] == normal for rank in (2, 3)),
        'Actual normal target TP shape traces differ')
    commits, publications = logical['commits'], logical['publications']
    expected_q = sum(row['real_query_count'] for row in normal)+sum(sum(row['budgets'])+len(row['budgets']) for row in commits)
    expected_b = sum(sum(row['budgets']) for row in normal)+sum(sum(row['budgets']) for row in commits)
    for rank, snapshot in enumerate(snapshots):
        before, after = snapshot['policy_before'], snapshot['policy_after']
        require(set(before) == set(after) == set(original.POLICY_COUNTERS)
            and all(type(value) is int and value >= 0 for item in (before, after) for value in item.values())
            and after[original.POLICY_COUNTERS[0]]-before[original.POLICY_COUNTERS[0]] == len(normal)+len(commits)
            and after[original.POLICY_COUNTERS[1]]-before[original.POLICY_COUNTERS[1]] == expected_q
            and snapshot['actual_decode_counters']['spec_rhythm_verified_tokens'] == expected_b,
            'Actual mixed/normal policy accounting differs')
        if rank:
            counters = snapshot['actual_decode_counters']
            require(counters['spec_rhythm_target_query_tokens'] == sum(row['physical_M'] for row in normal)
                +sum(sum(row['budgets'])+len(row['budgets']) for row in commits)
                and counters['spec_rhythm_target_graph_padding_queries'] == sum(
                    row['physical_M']-row['real_query_count'] for row in normal),
                'Actual Native normal/mixed query accounting differs')
    tokenized = first['tokenized_inputs']
    require(len(tokenized) == 48 and first['sampling'] == original.sampling_specifications(rows),
        'Actual original48 tokenizer/control projection differs')
    standalone = []
    prompt_queries = 0
    for call in first['standalone_prefill_calls']:
        ids = call['sequence_ids']
        require(type(ids) is list and ids and all(type(index) is int and 0 <= index < 48 for index in ids)
            and call['cached_tokens'] == [0]*len(ids)
            and call['prompt_lengths'] == [tokenized[index]['token_count'] for index in ids]
            and call['prompt_sha256'] == [tokenized[index]['token_sha256'] for index in ids]
            and call['uncached_queries'] == sum(tokenized[index]['token_count'] for index in ids)
            and call['precomputed_target_tokens'] is False and call['draft_prefill_completed'] is False,
            'Actual standalone/fallback did not recompute its complete prompt')
        standalone.extend(ids)
        prompt_queries += call['uncached_queries']
    require(sorted(standalone+[row['owner'] for row in publications]) == list(range(48))
        and all(row['owner'] in standalone for row in logical['aborted'])
        and first['actual_decode_counters']['spec_rhythm_prefill_requests'] == 48
        and first['actual_decode_counters']['spec_rhythm_prefill_batches'] == len(first['standalone_prefill_calls'])+len(publications),
        'Actual first-token paths do not cover every request exactly once')
    owner = snapshots[1]
    results = owner['generate_results']
    require(type(results) is list and len(results) == 48, 'Candidate did not complete48 result rows')
    delivered, finished, previous = [[] for _ in rows], [0]*48, -1.
    for event in owner['stream_events']:
        index, output = event['request_index'], event['token_ids']
        require(type(index) is int and 0 <= index < 48 and event['request_id'] == rows[index]['request_id']
            and type(output) is list and output and all(type(token) is int and token >= 0 for token in output)
            and type(event['finished']) is bool and original._finite(event['elapsed_seconds'])
            and event['elapsed_seconds'] >= previous and finished[index] == 0,
            'Actual candidate stream is malformed/repeated/out of order')
        previous = event['elapsed_seconds']
        delivered[index].extend(output)
        finished[index] += int(event['finished'])
    output_rows = []
    for index, (result, manifest) in enumerate(zip(results, rows, strict=True)):
        output = result['completion_token_ids']
        require(type(output) is list and len(output) == 256
            and all(type(token) is int and token >= 0 for token in output)
            and delivered[index] == output and finished[index] == 1
            and result['request_id'] == manifest['request_id'] and result['finish_reason'] == 'length'
            and result['arrival_ts'] == first['absolute_arrivals'][index]
            and result['max_tokens'] == 256 and result['ignore_eos'] is True
            and result['temperature'] == result['draft_temperature'] == 0.
            and result['slo_class'] == manifest['slo_class'] and result['slo_tpot_ms'] == manifest['slo_tpot_ms']
            and result['tree_mode'] is True, 'Actual candidate48 result/stream/control completion differs')
        output_rows.append(dict(request_index=index, request_id=result['request_id'], output_tokens=256,
            completion_sha256=token_digest(output), stream_tokens=len(delivered[index]), finished_events=finished[index]))
    require(sum(len(row) for row in delivered) == 12288
        and all(delivered[row['owner']][0] == row['first_token'] for row in publications),
        'Candidate12288 tokens or mixed first-token publication differs')
    histogram = Counter((len(row['budgets']), tuple(row['budgets'])) for row in commits)
    controls = {kind:sum(event['kind'] == kind for event in first['logical_events'])
        for kind in ('fallback-request', 'fallback-reset', 'abort', 'submit-error')}
    require(controls['submit-error'] == 0, 'An actual mixed target submission failed')
    clock_trace = [{key:row[key] for key in ('cycle_id', 'owner', 'start', 'end', 'final',
        'clocks_before_cycle_charge')} for row in commits]
    return dict(candidate_execution_complete=True, request_count=48, client_output_tokens=12288,
        per_request_output=output_rows, stream_events=len(owner['stream_events']),
        original_logical_counters={key:first['actual_decode_counters'][key] for key in original.LOGICAL_COUNTERS},
        actual_mixed_commits=len(commits), actual_mixed_publications=len(publications),
        actual_N16_mixed_commits=sum(len(row['budgets']) == 16 for row in commits),
        actual_visited_partitions=[dict(N=n, budgets=list(b), chunks=count) for (n,b),count in sorted(histogram.items())],
        actual_mixed_prompt_queries=sum(row['end']-row['start'] for row in commits),
        actual_standalone_prefill_batches=len(first['standalone_prefill_calls']),
        actual_standalone_prompt_queries=prompt_queries, observed_control_counts=controls,
        actual_frozen_logical_helper='live_poisson_probe._audit_logical',
        from0_and_N_gt1_mixed_and_original_clock_order_validated=True,
        all4_common_fields_equal=list(common), all4_generic_boundary_events=len(generic_first),
        generic_boundary_stage_counts=stage_counts, commit_clock_trace=clock_trace,
        actual_publication_trace=logical['publications'], native_counters_by_rank=native,
        Native_failure_counter_deltas_all_zero=True), results, tokenized


def validate_reference(row, index, tokenized, result):
    require(type(row) is dict and row['request_index'] == index
        and row['request_id'] == result['request_id'] and row['prompt_sha256'] == tokenized['token_sha256']
        and row['max_tokens'] == 256
        and row['execution'] == 'direct Native eager PA; fresh allocation; own previous predictions',
        'Actual independent reference request/prompt/execution differs')
    output, calls = row['completion_token_ids'], row['calls']
    require(type(output) is list and len(output) == 256
        and all(type(token) is int and token >= 0 for token in output)
        and type(calls) is list and len(calls) == 256, 'Actual independent reference did not finish256 calls/tokens')
    length = tokenized['token_count']
    for step, call in enumerate(calls):
        require(call['step'] == step and call['query_count'] == (length if step == 0 else 1)
            and call['start'] == (0 if step == 0 else length+step-1)
            and call['end'] == (length if step == 0 else length+step)
            and call['input_sha256'] == (tokenized['token_sha256'] if step == 0 else token_digest([output[step-1]]))
            and call['hidden_logits_and_Native_vote_finite'] is True,
            'Independent AR input chain/query positions/finite evidence differs: '+str(step))
    candidate = result['completion_token_ids']
    differences = [dict(index_0based=position, candidate_token=a, independent_AR_token=b)
        for position,(a,b) in enumerate(zip(candidate, output, strict=True)) if a != b]
    first = None if not differences else differences[0]['index_0based']
    own_only = [step for step,call in enumerate(calls) if step > 0
        and call['input_sha256'] == token_digest([output[step-1]])
        and call['input_sha256'] != token_digest([candidate[step-1]])]
    return dict(request_index=index, request_id=row['request_id'], output_tokens=256,
        prompt_queries=length, reference_forward_calls=256,
        source_prompt_and_all_own_previous_prediction_inputs_validated=True,
        hidden_logits_and_original_Native_votes_finite=True,
        completion_sha256=token_digest(output), candidate_completion_sha256=token_digest(candidate),
        exact_equal=not differences, first_mismatch_index_0based=first,
        common_prefix_tokens=256 if first is None else first,
        mismatch_count=len(differences), mismatches=differences,
        actual_post_divergence_calls_using_AR_own_previous_token_instead_of_candidate=own_only,
        context_at_first_mismatch=None if first is None else dict(
            start=max(0,first-5), end=min(256,first+6),
            candidate=candidate[max(0,first-5):min(256,first+6)],
            independent_AR=output[max(0,first-5):min(256,first+6)]))


def partial_AR_evidence(raw, module, results, tokenized):
    groups, files = {}, []
    for path, data in raw.items():
        match = re.fullmatch(r'rank([0-3])-AR-request([0-9]+)\.json', path.name)
        if not match:
            continue
        rank,index = map(int,match.groups())
        record = json.loads(data)
        require(0 <= index < 48 and record.get('schema') == module.SCHEMA
            and record.get('status') == 'actual-reference-request-complete-unscored'
            and type(record.get('rank')) is int and record['rank'] == rank
            and record.get('request_index') == index and record.get('request_id') == results[index]['request_id']
            and record.get('completed_steps') == 256 and record.get('completed_requests') == index+1
            and record.get('online_qualified') is False and record.get('cost_or_Goodput') is False,
            'Actual completed AR checkpoint envelope differs: '+path.name)
        require((rank == 0 and record['reference_row'] is None)
            or (rank in (1,2,3) and type(record['reference_row']) is dict), 'AR checkpoint role differs')
        groups.setdefault(index,{})[rank] = record
        files.append(dict(path=str(path), rank=rank, request_index=index, completed_steps=256,
            sha256=sha_bytes(data), actual_target_reference=rank != 0))
    require(0 in groups and all(rank in groups[0] for rank in (1,2,3)), 'Actual target3 AR0 is incomplete')
    completed, unpaired = [], []
    for index, ranks in sorted(groups.items()):
        if not all(rank in ranks for rank in (1,2,3)):
            unpaired.append(dict(request_index=index, present_ranks=sorted(ranks)))
            continue
        reference = ranks[1]['reference_row']
        require(all(ranks[rank]['reference_row'] == reference for rank in (2,3)),
            'Actual independent target TP AR rows differ: '+str(index))
        result = validate_reference(reference,index,tokenized[index],results[index])
        result['agreeing_target_ranks'] = [1,2,3]
        completed.append(result)
    request0 = next(row for row in completed if row['request_index'] == 0)
    require(request0['first_mismatch_index_0based'] == 250 and request0['mismatch_count'] == 4,
        'Independently derived request0 mismatch differs from the reported actual observation')
    return dict(completed_checkpoint_files=files,
        requests_with_three_equal_completed_target_AR_records=completed,
        unpaired_completed_checkpoints=unpaired,
        completed_logical_reference_output_tokens=sum(row['output_tokens'] for row in completed),
        target_TP_copies_not_multiplied=True, request0=request0,
        fresh_independent_AR_inputs_validated_for_completed_records_only=True,
        full48_AR_completed=False, independent12288_qualified=False,
        candidate_and_independent_AR0_exact=False)


def main():
    try:
        receipt_raw, receipt, pin_raw, pin, raw = partial_preconditions()
    except (AuditRejected,OSError,ValueError) as error:
        print('NOT RUN: '+str(error),file=sys.stderr)
        return 2
    report = dict(schema='ROOT.V85.actual-candidate-partial-AR-mismatch-failure-audit.v1',
        audited_UTC=datetime.now(timezone.utc).isoformat(),
        analysis_script=dict(path=str(Path(__file__).resolve()),sha256=sha_file(__file__)),
        Python=dict(executable=sys.executable,version=sys.version,dont_write_bytecode=sys.dont_write_bytecode),
        trusted_root_final_pin=dict(path=str(ROOT_PIN),sha256=sha_bytes(pin_raw),identity=pin['identity']),
        original_job_status=receipt['status'],original_job_status_modified=False,
        original_supervisor_receipt=dict(path=str(RECEIPT),sha256=sha_bytes(receipt_raw),
            status=receipt['status'],errors=receipt['errors'],child=receipt['child']),
        actual_raw_inputs={str(path):sha_bytes(data) for path,data in raw.items()},
        full48_AR_auditor_called=False,full48_qualified=False,full463_qualified=False,
        independent12288_qualified=False,independent_AR_qualified=False,
        all65_qualified=False,new_mixed_qualified=False,formal_Goodput=False,
        online_qualified=False,latency_or_Goodput_inferred=False,integrity_checks={},errors=[])
    previous = sys.getprofile()
    try:
        checks,local,external,cpu,freeze = verify_integrity(receipt,pin)
        require(cpu['tests_run'] == 249 and len(cpu['modules']) == 29
            and len(freeze['files']) == 198 and len(cpu['external_input_hashes']) == 1610,
            'Actual final CPU249/29modules/198local/1610external counts differ')
        require(pin.get('tests_run') == 249 and pin.get('CPU_modules') == 29
            and pin.get('local_files') == 198 and pin.get('external_files') == 1610,
            'Trusted actual root final pin counts differ')
        report['integrity_checks'] = checks
        report['cleanup'] = cleanup_evidence(receipt)
        sys.setprofile(host_firewall)
        module,local_module,startup = load_helpers(cpu)
        snapshots = [json.loads(raw[path]) for path in CANDIDATE_PATHS]
        candidate,results,tokenized = candidate_evidence(snapshots,module,local_module,startup)
        report['actual_candidate'] = candidate
        report['partial_independent_AR'] = partial_AR_evidence(raw,module,results,tokenized)
        sys.setprofile(previous)
        verify_files(local,'Local source after HOST failure audit',local=True)
        verify_files(external,'External source after HOST failure audit')
        require(current_candidate_sources() == freeze['files'],'Current frozen source membership changed')
        require(ROOT_PIN.read_bytes() == pin_raw and RECEIPT.read_bytes() == receipt_raw,
            'Original root pin or terminal receipt changed during failure audit')
        require(all(path.read_bytes() == data for path,data in raw.items()),
            'Original partial raw bytes changed during failure audit')
        current_inputs = set(CANDIDATE_PATHS+sorted(PROGRESS.glob('rank*-AR-request*.json')))
        require(current_inputs == set(raw),'Actual partial input file membership changed')
        for name in ('CPU_source_archive','supervisor_source_archive'):
            archive = checks[name]
            require(sha_file(archive['path']) == archive['sha256'],'Original archive changed during failure audit')
        worker = report['cleanup']['worker_exit_record']
        if worker['present']:
            require(sha_file(worker['path']) == worker['sha256'],'Original worker-exit record changed')
        else:
            require(not Path(worker['path']).exists(),'Worker-exit record appeared during failure audit')
        report['all_original_evidence_unchanged_after_HOST_analysis'] = True
        report['actual_mismatch_independently_verified'] = True
    except Exception as error:
        report['errors'].append(type(error).__name__+': '+str(error))
    finally:
        sys.setprofile(previous)
    report['all_checks_passed'] = not report['errors']
    report['status'] = 'failure-evidence-verified' if report['all_checks_passed'] else 'audit-rejected'
    write_exclusive(report)
    print('V85 partial failure audit '+report['status']+'; original supervisor remains '+receipt['status']+'.')
    if report['all_checks_passed']:
        candidate,reference = report['actual_candidate'],report['partial_independent_AR']
        first = reference['request0']
        print('Candidate48 outputs='+str(candidate['client_output_tokens'])+'; mixed commits/publications='
            +str(candidate['actual_mixed_commits'])+'/'+str(candidate['actual_mixed_publications'])
            +'; N16 mixed commits='+str(candidate['actual_N16_mixed_commits'])+'.')
        print('Target3 independent AR0 agree; first mismatch0based='+str(first['first_mismatch_index_0based'])
            +'; differing tokens='+str(first['mismatch_count'])+'/256; completed AR requests='
            +str(len(reference['requests_with_three_equal_completed_target_AR_records']))+'.')
    else:
        print('Errors: '+repr(report['errors']))
    print('Private exclusive output: '+str(OUTPUT)+'. Full48/12288 independent AR, all65, full463, GP and online qualification remain false.')
    return 0 if report['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
