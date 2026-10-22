#!/usr/bin/env python3
"""Explicit HOST audit of actual V85 live48 plus independent eager AR; no launch.

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
OUTPUT = BASE/'budget-diagnostics/ROOT_V85_ACTUAL_LIVE_POISSON48_AUDIT.json'
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


def preconditions():
    require(len(sys.argv) == 1, 'This audit has fixed V85 inputs and accepts no arguments')
    require(sys.version_info[:2] == (3, 11)
        and Path(sys.executable).resolve() == RETAINED_PYTHON.resolve(),
        'Use retained /root/miniconda3/envs/vllm-hust-dev/bin/python3.11')
    require(sys.dont_write_bytecode, 'Invoke retained Python with -B')
    require(not OUTPUT.exists(), 'Refusing to overwrite '+str(OUTPUT))
    raw = RECEIPT.read_bytes()
    receipt = json.loads(raw)
    require(receipt.get('status') in ('complete', 'failed'),
        'Supervisor is not terminal; no output was created')
    require(receipt.get('schema') == 'V78.local-diagnostic-supervisor.v1'
        and receipt.get('action') == ACTION, 'Unexpected supervisor receipt/action')
    paths = [RAW_DIRECTORY/f'rank{rank}-local-diagnostic.json' for rank in range(4)]
    require(all(path.is_file() for path in paths),
        'Four completed actual raw worker reports are not present; no output was created')
    pin_raw, pin = load_root_pin()
    return raw, receipt, paths, pin_raw, pin


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


def verify_cleanup(receipt):
    require(receipt.get('physical_cards') == [2, 5, 6, 7]
        and receipt.get('formal_Goodput') is False and receipt.get('online_qualified') is False,
        'Supervisor device/scope contract differs')
    require(receipt.get('owned_remaining') == []
        and receipt.get('devices_after', {}).get('all_idle') is True
        and receipt.get('experimental_sources_unchanged') is True
        and receipt.get('external_sources_unchanged') is True,
        'Terminal owned/idle/source cleanup is incomplete')
    for label in ('pre_verifier', 'post_verifier'):
        code = receipt.get(label, {}).get('exit_code')
        require(type(code) is int and code == 0, 'Original '+label+' did not pass')
    require(type(receipt.get('errors')) is list
        and type(receipt.get('child', {}).get('exit_code')) is int,
        'Terminal error/child-exit evidence is absent')
    if receipt['status'] == 'complete':
        require(receipt['errors'] == [] and receipt['child']['exit_code'] == 0,
            'Complete supervisor has child/errors inconsistency')
    else:
        require(bool(receipt['errors']), 'Failed supervisor has no preserved errors')
    exited = read_json(RAW_DIRECTORY/'worker-exit.json')
    require(exited.get('closed') is True and exited.get('all_exited') is True,
        'Original worker-exit record is incomplete')
    return dict(owned_remaining=[], all_four_devices_idle=True,
        experimental_sources_unchanged=True, external_sources_unchanged=True,
        pre_verifier_exit_code=0, post_verifier_exit_code=0,
        worker_exit=exited, worker_exit_sha256=sha_file(RAW_DIRECTORY/'worker-exit.json'))



def exact_frozen_auditor(cpu):
    # No checker entry point is called: its readiness collectors can spawn a
    # verifier. These imports and the selected raw auditor are pure HOST paths.
    sys.path.insert(0, str(CANDIDATE))
    module = importlib.import_module('local_mixed_qualification')
    require(Path(module.__file__).resolve() == CANDIDATE/'local_mixed_qualification.py',
        'Audit imported a different local worker module')
    function = module.audit_completed_live_poisson_worker_reports
    code = compile((CANDIDATE/'local_mixed_qualification.py').read_bytes(),
        str(CANDIDATE/'local_mixed_qualification.py'), 'exec', dont_inherit=True)
    expected = [value for value in code.co_consts
        if type(value) is type(code) and value.co_name == function.__name__]
    require(len(expected) == 1 and function.__code__ == expected[0],
        'Loaded outer audit code differs from the exact frozen function')
    checker = importlib.import_module('check_v75')
    require(Path(checker.__file__).resolve() == CANDIDATE/'check_v75.py'
        and list(checker.CPU_MODULES) == cpu['modules'], 'CPU module membership differs')
    return function



def token_digest(tokens):
    return sha_bytes(json.dumps(list(tokens), separators=(',', ':')).encode())


def derive_live_scope(reports, frozen_audit):
    """Describe already accepted actual work, counting one common logical trace."""
    records = sorted((row['live_poisson_probe'] for row in reports), key=lambda row:row['rank'])
    first = records[0]
    targets, leader = first['target_ranks'], first['target_leader_rank']
    module = importlib.import_module('live_poisson_probe')
    require(Path(module.__file__).resolve() == CANDIDATE/'live_poisson_probe.py',
        'Live scope helper imported outside the exact frozen candidate')
    logical = module._audit_logical(first)
    commits, publications = logical['commits'], logical['publications']
    reservations, aborted = logical['reservations'], logical['aborted']
    tokenized, normal = first['tokenized_inputs'], records[targets[0]]['normal_target_calls']
    mixed_rows, geometries = [], {}
    n_hist, b_hist, partition_hist = Counter(), Counter(), Counter()
    for commit in commits:
        budgets = tuple(commit['budgets'])
        partition = tuple(commit['query_partition'])
        n, prompt_q = len(budgets), commit['end']-commit['start']
        m, s = sum(partition), len(partition)
        normal_q = n+sum(budgets)
        require(m == 197 and s == 19 and normal_q+prompt_q <= m,
            'Accepted mixed record unexpectedly left its actual physical extent')
        begin = logical['begin_by_cycle'][commit['cycle_id']]
        contexts = begin['contexts']
        mixed_rows.append(dict(cycle_id=commit['cycle_id'], owner=commit['owner'],
            identity=commit['identity'], start=commit['start'], end=commit['end'], final=commit['final'],
            N=n, budgets=list(budgets), target_ids=commit['target_ids'],
            proposal_ids=commit['proposal_ids'], normal_prefix_lengths=contexts,
            physical_M=m, physical_S=s, physical_query_partition=list(partition),
            normal_verification_queries=normal_q, real_prompt_queries=prompt_q,
            scratch_queries=m-normal_q-prompt_q))
        n_hist[n] += 1
        b_hist.update(budgets)
        partition_hist[(n, budgets)] += 1
        key = (n, budgets, prompt_q, commit['final'], partition)
        group = geometries.setdefault(key, dict(chunks=0, normal_prefix_min=min(contexts),
            normal_prefix_max=max(contexts), prompt_cursor_min=commit['start'], prompt_cursor_max=commit['end']))
        group['chunks'] += 1
        group['normal_prefix_min'] = min(group['normal_prefix_min'], min(contexts))
        group['normal_prefix_max'] = max(group['normal_prefix_max'], max(contexts))
        group['prompt_cursor_min'] = min(group['prompt_cursor_min'], commit['start'])
        group['prompt_cursor_max'] = max(group['prompt_cursor_max'], commit['end'])
    visited = [dict(N=n, budgets=list(budgets), chunks=count)
        for (n, budgets), count in sorted(partition_hist.items())]
    require(frozen_audit.get('request_count') == 48 and frozen_audit.get('client_output_tokens') == 12288
        and frozen_audit.get('independent_eager_AR_exact') is True
        and frozen_audit.get('mixed_committed_chunks') == len(commits)
        and frozen_audit.get('mixed_publications') == len(publications)
        and frozen_audit.get('actual_visited_partitions') == visited
        and all(frozen_audit.get(key) is False for key in
            ('numerical_qualification_inferred', 'all65_qualified', 'full463_qualified',
             'online_qualified', 'cost_or_Goodput')),
        'Frozen audit result does not establish the requested exact actual48 scope')
    shape_rows = []
    for (n, budgets, prompt_q, final, partition), counts in geometries.items():
        shape_rows.append(dict(N=n, budgets=list(budgets), real_prompt_queries=prompt_q,
            final=final, physical_M=sum(partition), physical_S=len(partition),
            physical_query_partition=list(partition), **counts))
    shape_rows.sort(key=lambda row:(-row['chunks'], row['N'], row['budgets'],
        row['real_prompt_queries'], row['final'], row['physical_query_partition']))
    standalone = first['standalone_prefill_calls']
    batch_hist = Counter(len(row['sequence_ids']) for row in standalone)
    full_prompt_q = sum(row['uncached_queries'] for row in standalone)
    mixed_prompt_q = sum(row['real_prompt_queries'] for row in mixed_rows)
    aborted_owners = {row['owner'] for row in aborted}
    recomputed_q = sum(tokenized[owner]['token_count']
        for call in standalone for owner in call['sequence_ids'] if owner in aborted_owners)
    references = records[targets[0]]['eager_AR_reference']
    results = records[leader]['generate_results']
    reference_rows = []
    for index, (reference, result, prompt) in enumerate(zip(references, results, tokenized, strict=True)):
        require(reference['completion_token_ids'] == result['completion_token_ids']
            and len(reference['completion_token_ids']) == 256, 'Accepted AR/client tokens unexpectedly differ')
        reference_rows.append(dict(request_index=index, request_id=reference['request_id'],
            prompt_queries=prompt['token_count'], prompt_sha256=prompt['token_sha256'],
            output_tokens=len(reference['completion_token_ids']),
            exact_completion_sha256=token_digest(reference['completion_token_ids']),
            actual_forward_calls=len(reference['calls']),
            actual_reference_query_tokens=sum(call['query_count'] for call in reference['calls'])))
    require(len(reference_rows) == 48 and sum(row['output_tokens'] for row in reference_rows) == 12288,
        'Actual independent AR token total differs')
    base = first['source']['original_input_and_decode']
    controls = first['observed_control_counts']
    return dict(counting_scope='one common logical trace; target TP reports are not multiplied',
        request_count=48, client_output_tokens=12288,
        actual_visited_partitions=visited, distinct_visited_partitions=len(visited),
        allowed_input_partitions=len(first['allowed_partitions']), all65_qualified=False,
        mixed=dict(committed_chunks=len(commits), publications=len(publications),
            visited_request_indices=sorted({row['owner'] for row in mixed_rows}),
            published_request_indices=[row['owner'] for row in publications],
            N_histogram=[dict(N=n, chunks=count) for n, count in sorted(n_hist.items())],
            visited_N16=n_hist[16] > 0,
            per_B_proposal_occurrences=[dict(B=b, occurrences=b_hist[b]) for b in range(1, 6)],
            actual_geometry_histogram=shape_rows, chunk_records=mixed_rows,
            committed_real_prompt_queries=mixed_prompt_q,
            normal_verification_queries=sum(row['normal_verification_queries'] for row in mixed_rows),
            physical_queries=sum(row['physical_M'] for row in mixed_rows),
            scratch_queries=sum(row['scratch_queries'] for row in mixed_rows),
            reservation_records=reservations, publication_records=publications),
        fallback=dict(observed_control_counts=controls, abort_records=aborted,
            explicit_terminal_close_abort_identities=first['close_abort_identities'],
            prior_mixed_queries_for_aborted_reservations=sum(row['cursor'] for row in aborted),
            full_standalone_prompt_queries_for_aborted_owners=recomputed_q,
            standalone_batches=len(standalone), standalone_requests=sum(batch_hist.elements()),
            standalone_batch_size_histogram=[dict(requests=n, batches=count)
                for n, count in sorted(batch_hist.items())],
            standalone_full_prompt_queries=full_prompt_q,
            total_mixed_plus_full_standalone_prompt_query_work=mixed_prompt_q+full_prompt_q,
            distinct_original_prompt_queries=sum(row['token_count'] for row in tokenized),
            interpretation='recomputed full prompts and prior mixed chunks remain counted; no saved-time inference'),
        normal_target=dict(calls=len(normal),
            real_query_tokens=sum(row['real_query_count'] for row in normal),
            physical_query_tokens=sum(row['physical_M'] for row in normal)),
        independent_eager_AR=dict(scope='only the actual same48 original prompts and256 outputs each',
            output_tokens_exactly_compared=12288, agreeing_target_ranks=targets,
            per_request=reference_rows,
            source='original full prompt, then only own previous predictions; separate fresh allocation per request',
            full463_qualified=False),
        manifest_class_counts=dict(sorted(Counter(row['slo_class'] for row in first['sampling']).items())),
        manifest_identity={key:base[key] for key in
            ('manifest_path', 'manifest_sha256', 'prefix_sha256', 'prefix_count')},
        formal_Goodput=False, latency_inferred=False, full463_qualified=False, online_qualified=False)


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



def main():
    try:
        receipt_bytes, receipt, paths, pin_bytes, pin = preconditions()
        raw_bytes = {path:path.read_bytes() for path in paths}
    except (AuditRejected, OSError, ValueError) as error:
        print('NOT RUN: '+str(error), file=sys.stderr)
        return 2
    report = dict(schema='ROOT.V85.actual-live48-independent-eager-AR-host-audit.v1',
        audited_UTC=datetime.now(timezone.utc).isoformat(),
        analysis_script=dict(path=str(Path(__file__).resolve()), sha256=sha_file(__file__)),
        Python=dict(executable=sys.executable, version=sys.version, dont_write_bytecode=sys.dont_write_bytecode),
        trusted_root_final_pin=dict(path=str(ROOT_PIN), sha256=sha_bytes(pin_bytes), identity=pin['identity']),
        original_job_status=receipt['status'], original_job_status_modified=False,
        supervisor_receipt=dict(path=str(RECEIPT), sha256=sha_bytes(receipt_bytes),
            status=receipt['status'], action=receipt['action'], errors=receipt.get('errors'),
            child_exit_code=receipt.get('child', {}).get('exit_code')),
        raw_reports={str(path):sha_bytes(raw_bytes[path]) for path in paths},
        exact_frozen_auditor='local_mixed_qualification.audit_completed_live_poisson_worker_reports',
        exact_frozen_auditor_executed=False, actual_live_records_accepted=False,
        integrity_checks={}, errors=[], formal_Goodput=False, latency_inferred=False,
        numerical_qualification_inferred=False, all65_qualified=False,
        full463_qualified=False, online_qualified=False, cost_or_Goodput=False)
    previous_profile = sys.getprofile()
    try:
        checks, local, external, cpu, freeze = verify_integrity(receipt, pin)
        report['integrity_checks'] = checks
        report['cleanup'] = verify_cleanup(receipt)
        raw_reports = [json.loads(raw_bytes[path]) for path in paths]
        sys.setprofile(host_firewall)
        audit = exact_frozen_auditor(cpu)
        report['exact_frozen_auditor_executed'] = True
        report['frozen_audit_result'] = audit(raw_reports)
        report['frozen_auditor_passed'] = True
        report['actual_live_scope'] = derive_live_scope(raw_reports, report['frozen_audit_result'])
        sys.setprofile(previous_profile)
        verify_files(local, 'Local source after HOST audit', local=True)
        verify_files(external, 'External source after HOST audit')
        require(current_candidate_sources() == freeze['files'], 'Current local source membership changed during analysis')
        require(ROOT_PIN.read_bytes() == pin_bytes, 'Trusted root final audit changed during analysis')
        require(RECEIPT.read_bytes() == receipt_bytes, 'Original supervisor receipt changed during analysis')
        require(all(sha_file(path) == report['raw_reports'][str(path)] for path in paths),
            'Original raw worker bytes changed during analysis')
        require(sha_file(RAW_DIRECTORY/'worker-exit.json') == report['cleanup']['worker_exit_sha256'],
            'Original worker-exit bytes changed during analysis')
        for key in ('supervisor_source_archive', 'CPU_source_archive'):
            archived = report['integrity_checks'][key]
            require(sha_file(archived['path']) == archived['sha256'], 'Retained archive changed during analysis')
        report['evidence_unchanged_after_analysis'] = True
        report['actual_live_records_accepted'] = True
    except Exception as error:
        report['errors'].append(type(error).__name__+': '+str(error))
    finally:
        sys.setprofile(previous_profile)
    report['all_checks_passed'] = not report['errors']
    report['status'] = 'passed' if report['all_checks_passed'] else 'rejected'
    try:
        write_exclusive(report)
    except OSError as error:
        print('AUDIT NOT SAVED: '+str(error), file=sys.stderr)
        return 2
    print('V85 HOST audit '+report['status']+'; original supervisor job remains '+receipt['status']+'.')
    if report['all_checks_passed']:
        scope = report['actual_live_scope']
        print('Mixed chunks/publications: '+str(scope['mixed']['committed_chunks'])+'/'
            +str(scope['mixed']['publications'])+'; visited partitions: '
            +str(scope['distinct_visited_partitions'])+'; N16 visited: '+str(scope['mixed']['visited_N16'])+'.')
        print('Actual same48 client/independent AR tokens exactly compared: '
            +str(scope['independent_eager_AR']['output_tokens_exactly_compared'])
            +'; fallback controls: '+repr(scope['fallback']['observed_control_counts'])+'.')
    else:
        print('Errors: '+repr(report['errors']))
    print('Private exclusive output: '+str(OUTPUT)+'. No GP, latency, all65, full463 or online qualification inferred.')
    return 0 if report['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
