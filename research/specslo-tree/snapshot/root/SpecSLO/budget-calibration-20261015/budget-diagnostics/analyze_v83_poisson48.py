#!/usr/bin/env python3
"""Explicit HOST audit of preserved V83 original48 records; never a launch.

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
CANDIDATE = BASE/'tree-live-poisson-coverage-v83'
TAG = 'original-poisson48-v1'
ACTION = 'poisson-coverage-probe'
RECEIPT = CANDIDATE/('local-receipt-'+TAG+'.json')
RAW_DIRECTORY = CANDIDATE/'local-output'/TAG
FREEZE = CANDIDATE/'PREPARED_SOURCE_HASHES_V83.json'
CPU = CANDIDATE/'CPU_RESULT_V83.json'
PRIOR = CANDIDATE/'ORIGINAL_PRIOR_BINDING_V78.json'
OUTPUT = BASE/'budget-diagnostics/ROOT_V83_ACTUAL_POISSON48_AUDIT.json'
RETAINED_PYTHON = Path('/root/miniconda3/envs/vllm-hust-dev/bin/python3.11')
IDENTITY = {
    'freeze_sha256': '7791aab7293a88aee1dcce7830ec9025f7c2c577ccfd73b5b6b4c31183004b7b',
    'CPU_sha256': 'a9a3a51ad8323a1c656483081ff9bf7bcf18d9efd07a40d63f145b0233965a5e',
    'original_prior_binding_sha256': '5c185e86b427a63d1b31cadb1fbde581af58cb39034ab525ade2d5a5d802686c',
}


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


def preconditions():
    require(len(sys.argv) == 1, 'This audit has fixed V83 inputs and accepts no arguments')
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
        'Four actual raw worker reports are not present; no output was created')
    return raw, receipt, paths


def verify_integrity(receipt):
    require(receipt['readiness']['identity'] == IDENTITY, 'Supervisor source identity differs')
    for path, key in ((FREEZE, 'freeze_sha256'), (CPU, 'CPU_sha256'),
                      (PRIOR, 'original_prior_binding_sha256')):
        require(sha_file(path) == IDENTITY[key], 'Pinned identity bytes differ: '+str(path))
    freeze, cpu, prior = read_json(FREEZE), read_json(CPU), read_json(PRIOR)
    require(freeze.get('schema') == 'V83.original-Poisson-coverage-local-prepared-source.v1'
        and freeze.get('CPU_sha256') == IDENTITY['CPU_sha256']
        and freeze.get('original_prior_binding_sha256') == IDENTITY['original_prior_binding_sha256']
        and freeze.get('online_qualified') is False
        and freeze.get('local_numeric_qualified') is False, 'Prepared freeze contract differs')
    require(cpu.get('phase') == 'final-local-source-before-NPU' and cpu.get('status') == 'passed'
        and type(cpu.get('tests_run')) is int and cpu['tests_run'] == 232
        and cpu.get('failures') == 0 and cpu.get('errors') == 0
        and cpu.get('NPU_initialized') is False and cpu.get('Python_dont_write_bytecode') is True
        and cpu.get('model_initialization_attempts') == []
        and cpu.get('NPU_initialization_attempts') == []
        and cpu.get('builder_and_compile_passed') is True,
        'Actual final HOST232/no-model/no-NPU receipt differs')
    require(freeze['files'] == cpu['input_hashes'] == cpu['input_hashes_after']
        and len(freeze['files']) == 192, 'Frozen/current CPU192 local source maps differ')
    require(cpu['external_input_hashes'] == cpu['external_input_hashes_after']
        and len(cpu['external_input_hashes']) == 1208, 'CPU1208 external source closure differs')
    expected_sources = dict(freeze['files'])
    expected_sources.update({str(FREEZE): IDENTITY['freeze_sha256'],
        str(CPU): IDENTITY['CPU_sha256'], str(PRIOR): IDENTITY['original_prior_binding_sha256']})
    require(receipt['source_hashes'] == expected_sources, 'Supervisor archive source map differs')
    expected_external = {**prior['observed'], **cpu['external_input_hashes']}
    require(receipt['external_input_hashes'] == expected_external, 'Supervisor external closure differs')
    checks = dict(
        local_sources=verify_files(expected_sources, 'Prepared local source', local=True),
        CPU_external_sources=verify_files(cpu['external_input_hashes'], 'CPU external source'),
        supervisor_external_sources=verify_files(expected_external, 'Supervisor external source'))
    snapshot = CANDIDATE/'source-snapshots'/('local-'+TAG+'.tar.gz')
    require(Path(receipt['source_snapshot']) == snapshot, 'Unexpected source snapshot path')
    checks['supervisor_source_archive'] = verify_archive(snapshot,
        receipt['source_snapshot_sha256'], expected_sources)
    cpu_archive = Path(cpu['actual_local_source_archive'])
    require(cpu_archive.is_relative_to(CANDIDATE/'host-checks'), 'CPU archive outside retained host attempt')
    checks['CPU_source_archive'] = verify_archive(cpu_archive,
        cpu['actual_local_source_archive_sha256'], freeze['files'])
    checks['CPU_receipt'] = dict(path=str(CPU), sha256=IDENTITY['CPU_sha256'],
        tests_run=232, NPU_initialized=False, exact=True)
    return checks, expected_sources, expected_external, cpu


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
    # Import no checker entry point: require_local_launch/parent collectors can
    # spawn verifier subprocesses. Only the already verified pure HOST audit
    # function is called here, with no diagnostic permit or model creation.
    sys.path.insert(0, str(CANDIDATE))
    module = importlib.import_module('local_mixed_qualification')
    require(Path(module.__file__).resolve() == CANDIDATE/'local_mixed_qualification.py',
        'Audit imported a different local worker module')
    function = module.audit_completed_poisson_worker_reports
    code = compile((CANDIDATE/'local_mixed_qualification.py').read_bytes(),
        str(CANDIDATE/'local_mixed_qualification.py'), 'exec', dont_inherit=True)
    expected = [value for value in code.co_consts
        if type(value) is type(code) and value.co_name == function.__name__]
    require(len(expected) == 1 and function.__code__ == expected[0],
        'Loaded audit code differs from the exact frozen function')
    checker = importlib.import_module('check_v75')
    require(Path(checker.__file__).resolve() == CANDIDATE/'check_v75.py'
        and list(checker.CPU_MODULES) == cpu['modules'], 'CPU module membership differs')
    return function


def derive_coverage(reports):
    records = sorted((row['poisson_coverage_probe'] for row in reports), key=lambda row: row['rank'])
    targets = records[0]['target_ranks']
    representative = records[targets[0]]
    trace = representative['target_calls']
    requests = records[0]['source']['requests']
    classes = {row['request_index']: row['slo_class'] for row in requests}
    n_hist, b_hist, family_hist, page_hist = Counter(), Counter(), Counter(), Counter()
    geometry, partitions, per_class = {}, {}, {}
    all_contexts, real_total, physical_total, owners_total = [], 0, 0, 0
    def add_group(groups, key, row):
        item = groups.setdefault(key, dict(calls=0, prefix_min=min(row['contexts']),
            prefix_max=max(row['contexts'])))
        item['calls'] += 1
        item['prefix_min'] = min(item['prefix_min'], min(row['contexts']))
        item['prefix_max'] = max(item['prefix_max'], max(row['contexts']))
    for row in trace:
        n_hist[row['N']] += 1
        b_hist.update(row['budgets'])
        family_hist[row['family']] += 1
        real_total += row['real_query_count']
        physical_total += row['physical_M']
        owners_total += row['N']
        all_contexts.extend(row['contexts'])
        key = (row['N'], tuple(row['budgets']), row['family'], row['physical_M'], row['physical_S'])
        add_group(geometry, key, row)
        add_group(partitions, key+(tuple(row['physical_query_partition']),), row)
        for owner, prefix in zip(row['sequence_ids'][:row['N']], row['contexts'], strict=True):
            page_hist[prefix//128] += 1
            label = classes[owner]
            entry = per_class.setdefault(label, dict(owner_appearances=0,
                prefix_min=prefix, prefix_max=prefix))
            entry['owner_appearances'] += 1
            entry['prefix_min'], entry['prefix_max'] = min(entry['prefix_min'], prefix), max(entry['prefix_max'], prefix)
    require(trace and physical_total > 0 and real_total > 0, 'Audited target trace unexpectedly empty')
    def group_rows(groups, with_partition=False):
        values = []
        for key, count in groups.items():
            n, budgets, family, m, s = key[:5]
            value = dict(N=n, budgets=list(budgets), family=family, physical_M=m, physical_S=s,
                calls=count['calls'], call_fraction=count['calls']/len(trace),
                prefix_min=count['prefix_min'], prefix_max=count['prefix_max'],
                real_query_count_per_call=n+sum(budgets),
                physical_queries=count['calls']*m,
                physical_query_fraction=count['calls']*m/physical_total)
            if with_partition:
                value['physical_query_partition'] = list(key[5])
            values.append(value)
        return sorted(values, key=lambda value: (-value['calls'], value['N'], value['budgets'],
            value['family'], value['physical_M'], value['physical_S']))
    partition_rows = group_rows(partitions, True)
    prompt_calls = records[0]['prefill_calls']
    batch_hist = Counter(len(row['sequence_ids']) for row in prompt_calls)
    tokens = records[0]['tokenized_inputs']
    prompt_lengths = [row['token_count'] for row in tokens]
    n_gt16 = sum(row['N'] > 16 for row in trace)
    return dict(
        counting_scope='one logical target trace; matching TP ranks are not multiplied',
        representative_target_rank=representative['rank'], agreeing_target_ranks=targets,
        target_calls=len(trace), normal_owner_appearances=owners_total,
        N_histogram=[dict(N=n, calls=count) for n, count in sorted(n_hist.items())],
        N_gt16=dict(calls=n_gt16, call_fraction=n_gt16/len(trace),
            normal_owner_appearances=sum(row['N'] for row in trace if row['N'] > 16)),
        per_B_proposal_occurrences=[dict(B=b, occurrences=b_hist[b]) for b in range(1, 6)],
        family_histogram=[dict(family=family, calls=count) for family, count in sorted(family_hist.items())],
        exact_geometry_histogram=group_rows(geometry),
        physical_partition_histogram=partition_rows,
        hottest_partitions_by_calls=partition_rows[:20],
        hottest_partitions_by_physical_queries=sorted(partition_rows,
            key=lambda value: (-value['physical_queries'], -value['calls'], value['N'], value['budgets']))[:20],
        query_totals=dict(real_queries=real_total, physical_queries=physical_total,
            padding_queries=physical_total-real_total, real_to_physical_query_weighted_ratio=real_total/physical_total,
            physical_to_real_query_weighted_ratio=physical_total/real_total,
            interpretation='ratios of summed queries; no kernel-cost, latency or throughput inference'),
        context_span=dict(field='actual plan.prefix_len, not full prompt or KV byte contents',
            owner_appearances=len(all_contexts), prefix_min=min(all_contexts), prefix_max=max(all_contexts),
            per_class=per_class, page128_histogram=[dict(prefix_min=page*128,
                prefix_max=page*128+127, owner_appearances=count) for page, count in sorted(page_hist.items())]),
        prefill=dict(counting_scope='one logical all-rank prompt trace', batches=len(prompt_calls),
            batch_size_histogram=[dict(requests=size, batches=count) for size, count in sorted(batch_hist.items())],
            singleton_batches=batch_hist[1], requests=sum(size*count for size, count in batch_hist.items()),
            total_prompt_queries=sum(row['uncached_queries'] for row in prompt_calls),
            tokenized_prompt_queries=sum(prompt_lengths), prompt_length_min=min(prompt_lengths),
            prompt_length_max=max(prompt_lengths)),
        manifest_class_counts=dict(sorted(Counter(row['slo_class'] for row in requests).items())),
        manifest_identity={key:records[0]['source'][key] for key in
            ('manifest_path', 'manifest_sha256', 'prefix_sha256', 'prefix_count')})


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
        receipt_bytes, receipt, paths = preconditions()
    except (AuditRejected, OSError, ValueError) as error:
        print('NOT RUN: '+str(error), file=sys.stderr)
        return 2
    report = dict(schema='ROOT.V83.actual-original48-host-audit.v1',
        audited_UTC=datetime.now(timezone.utc).isoformat(),
        analysis_script=dict(path=str(Path(__file__).resolve()), sha256=sha_file(__file__)),
        Python=dict(executable=sys.executable, version=sys.version, dont_write_bytecode=sys.dont_write_bytecode),
        original_job_status=receipt['status'], original_job_status_modified=False,
        supervisor_receipt=dict(path=str(RECEIPT), sha256=sha_bytes(receipt_bytes),
            status=receipt['status'], action=receipt['action'], errors=receipt.get('errors'),
            child_exit_code=receipt.get('child', {}).get('exit_code')),
        raw_reports={str(path):sha_file(path) for path in paths},
        exact_frozen_auditor='local_mixed_qualification.audit_completed_poisson_worker_reports',
        exact_frozen_auditor_executed=False, actual_coverage_records_accepted=False,
        integrity_checks={}, errors=[], formal_Goodput=False, latency_inferred=False,
        new_mixed_qualified=False, numerical_qualification_inferred=False,
        independent_AR_qualified=False, full463_qualified=False, online_qualified=False)
    previous_profile = sys.getprofile()
    try:
        checks, local, external, cpu = verify_integrity(receipt)
        report['integrity_checks'] = checks
        report['cleanup'] = verify_cleanup(receipt)
        raw_reports = [read_json(path) for path in paths]
        sys.setprofile(host_firewall)
        audit = exact_frozen_auditor(cpu)
        report['exact_frozen_auditor_executed'] = True
        report['frozen_audit_result'] = audit(raw_reports)
        report['frozen_auditor_passed'] = True
        report['coverage'] = derive_coverage(raw_reports)
        sys.setprofile(previous_profile)
        verify_files(local, 'Local source after HOST audit', local=True)
        verify_files(external, 'External source after HOST audit')
        require(RECEIPT.read_bytes() == receipt_bytes, 'Original supervisor receipt changed during analysis')
        require(all(sha_file(path) == report['raw_reports'][str(path)] for path in paths),
            'Original raw worker bytes changed during analysis')
        report['evidence_unchanged_after_analysis'] = True
        report['actual_coverage_records_accepted'] = True
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
    print('V83 HOST audit '+report['status']+'; original supervisor job remains '+receipt['status']+'.')
    if report['all_checks_passed']:
        coverage = report['coverage']
        print('Target calls: '+str(coverage['target_calls'])+'; N>16: '
            +str(coverage['N_gt16']['calls'])+'; prefill batches/singletons: '
            +str(coverage['prefill']['batches'])+'/'+str(coverage['prefill']['singleton_batches'])+'.')
        print('Real/physical queries: '+str(coverage['query_totals']['real_queries'])+'/'
            +str(coverage['query_totals']['physical_queries'])+'; prefix span: '
            +str(coverage['context_span']['prefix_min'])+'..'+str(coverage['context_span']['prefix_max'])+'.')
    else:
        print('Errors: '+repr(report['errors']))
    print('Private exclusive output: '+str(OUTPUT)+'. No GP, latency, new mixed or AR qualification inferred.')
    return 0 if report['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
