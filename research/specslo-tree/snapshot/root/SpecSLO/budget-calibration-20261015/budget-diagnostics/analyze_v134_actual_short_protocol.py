"""Read a terminal V134 receipt and its retained HOST evidence; no model imports.

Run only after ROOT confirms the supervisor has closed.  The original frozen
strict auditor is reused once.  No saved device tensors exist in this scope,
so this program never claims to recompute hidden/KV numerics on HOST.
"""
import argparse
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tarfile
import traceback

B = Path('/root/SpecSLO/budget-calibration-20261015')
D = B/'budget-diagnostics'
P = B/'tree-short-parent-eager-v134'
PRE = D/'ROOT_V134_PRE_NPU_SOURCE_AND_HOST_AUDIT.json'
PRE_SHA = '09ce7c8e41080090145cd22c8da2754d0826288c7273f5b974038b50480ec90c'
CPU = P/'CPU_RESULT_V134.json'
FREEZE = P/'PREPARED_SOURCE_HASHES_V134.json'
ACTION = 'parent3-child3-repair01-local-protocol'
OUTCOMES = ('observed_target', 'valid_promotion', 'parent_reject')


def require(value, message):
    if not value:
        raise ValueError('V134 external actual audit: '+message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def read_json(path):
    path = Path(path)
    raw = path.read_bytes()
    if path.suffix == '.gz':
        raw = gzip.decompress(raw)
    return json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError('Nonfinite JSON constant '+value)))


def archive_check(path, expected):
    """Check every member byte, without extraction or tensor deserialization."""
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        require(len(members) == len(expected)
                and {m.name for m in members} == set(expected), 'archive exact member coverage')
        for member in members:
            require(member.isfile() and not member.issym() and not member.islnk(),
                    'archive only regular source members')
            source = archive.extractfile(member)
            require(source is not None, 'readable archive member')
            digest = hashlib.sha256()
            for block in iter(lambda: source.read(1024**2), b''):
                digest.update(block)
            require(digest.hexdigest() == expected[member.name],
                    'archive bytes differ: '+member.name)
    return dict(**artifact(path), all_members_bytes_checked=True, members=len(expected))


def source_gate(receipt):
    require(sha(PRE) == PRE_SHA, 'original ROOT preaudit SHA')
    pre, cpu, frozen = read_json(PRE), read_json(CPU), read_json(FREEZE)
    require(pre['status'] == 'passed' and pre['readiness_exit_code'] == 0
            and pre['CPU_sha256'] == sha(CPU) and pre['freeze_sha256'] == sha(FREEZE),
            'ROOT actual preaudit identity')
    sys.path.insert(0, str(P))
    import check_v134
    from source_snapshot_v134 import collect, runtime_outputs
    # Original exact source/readiness checker; no tests or device queries.
    ready = check_v134.require_ready(receipt['readiness']['identity'])
    require(collect(P) == frozen['files'] == cpu['input_hashes'] == cpu['input_hashes_after']
            and frozen['external_files'] == cpu['external_input_hashes']
                == cpu['external_input_hashes_after']
            and cpu['status'] == 'passed' and cpu['tests_run'] == 22
            and cpu['failures'] == cpu['errors'] == 0,
            'actual source/CPU before-after closure')
    require(len(frozen['files']) == 34 and len(frozen['external_files']) == 112,
            'frozen local/external scope')
    all_sources = {**frozen['files'], **frozen['external_files'],
                   str(CPU): sha(CPU), str(FREEZE): sha(FREEZE)}
    require(all(sha(path) == wanted for path, wanted in all_sources.items()),
            'all current pinned source/input bytes')
    prepared = Path(pre['archive'])
    require(sha(prepared) == pre['archive_sha256'], 'prepared full archive SHA')
    prepared_audit = archive_check(prepared,
        {'inputs/'+name.lstrip('/'): wanted for name, wanted in all_sources.items()})
    local = {**frozen['files'], str(CPU): sha(CPU), str(FREEZE): sha(FREEZE)}
    require(receipt['source_hashes'] == local
            and receipt['external_input_hashes'] == frozen['external_files'],
            'actual supervisor source/input maps')
    runtime_archive = Path(receipt['source_snapshot'])
    require(runtime_archive.is_relative_to(P/'source-snapshots')
            and sha(runtime_archive) == receipt['source_snapshot_sha256'],
            'actual runtime local source archive path/SHA')
    runtime_audit = archive_check(runtime_archive,
        {Path(name).name: wanted for name, wanted in local.items()})
    outputs = runtime_outputs(P)  # Strict known-fusion exception; never deleted.
    require(outputs == receipt['runtime_outputs'], 'strict actual CANN runtime output records')
    return dict(passed=True, ROOT_preaudit=artifact(PRE), CPU=artifact(CPU),
                freeze=artifact(FREEZE), local_sources=34, external_inputs=112,
                original_readiness_identity=ready['identity'], prepared_archive=prepared_audit,
                actual_runtime_local_archive=runtime_audit, runtime_outputs=outputs,
                archive_scope='Prepared archive covers all local/external inputs; runtime archive covers local+CPU+freeze.')


def cleanup_gate(receipt, directory):
    require(receipt['schema'] == 'V134.actual-upstream-supervisor.v1'
            and receipt['action'] == ACTION and receipt['physical_cards'] == [2, 5, 6, 7],
            'actual supervisor scope')
    require(receipt['status'] == 'complete' and receipt['errors'] == []
            and receipt['child']['exit_code'] == 0 and receipt['owned_remaining'] == []
            and receipt['devices_before']['all_idle'] is True
            and receipt['devices_after']['all_idle'] is True
            and receipt['experimental_sources_unchanged'] is True
            and receipt['external_sources_unchanged'] is True,
            'terminal original success and owned-process/source cleanup')
    logs = []
    for side in ('pre', 'post'):
        record = receipt[side+'_verifier']
        require(record['exit_code'] == 0 and sha(record['log']) == record['log_sha256']
                and Path(record['log']).stat().st_size == record['log_bytes'],
                'retained original verifier '+side)
        logs.append(artifact(record['log']))
    runtime = B/'serial-v1/runtime.sh'
    sources = re.findall(r'^source (/[^ \n]+)', runtime.read_text(), re.MULTILINE)
    observed = [dict(path=name, sha256=sha(name)) for name in sources]
    require(len(observed) == 2 and observed == receipt['actual_sourced_runtime_shells_before']
            == receipt['actual_sourced_runtime_shells_after'], 'actual sourced CANN/ATB shells before/after')
    require(receipt['runtime_aliases_before'] == receipt['runtime_aliases_after'],
            'actual runtime alias records before/after')
    for row in receipt['runtime_aliases_after']:
        path = Path(row['path'])
        require(path.is_symlink() and os.readlink(path) == row['link']
                and str(path.resolve()) == row['resolved'] and path.exists() is row['exists'],
                'actual final runtime alias identity')
    exited = read_json(directory/'worker-exit.json')
    require(exited['closed'] is exited['all_exited'] is True, 'all original workers exited')
    current_bytes = sum(p.stat().st_size for p in P.rglob('*') if p.is_file())
    require(current_bytes < 32*1024**2 and receipt['private_bytes'] < 32*1024**2
            and not any(p.stat().st_size > 16*1024**2 for p in P.rglob('*') if p.is_file()),
            'actual private/file byte limits')
    return dict(passed=True, worker_exit=artifact(directory/'worker-exit.json'),
                verifier_logs=logs, actual_runtime_shells=observed,
                runtime_aliases=receipt['runtime_aliases_after'], current_component_private_bytes=current_bytes,
                original_supervisor_elapsed_seconds=receipt['elapsed_seconds'], hardware_queried_by_this_auditor=False)


def lease_check(row, case):
    lease = row['actual_source_lease']
    count = case['normal']+case['eager']
    require(type(lease['source_epoch']) is int and lease['source_epoch'] > 0
            and lease['real_work_owners'] == list(range(count))
            and lease['scratch_owners'] == list(range(64, 102))
            and lease['actual_storage_rows'] == 128 and lease['actual_page_roof'] == 544
            and lease['work_and_dummy_disjoint'] is lease['actual_page_leases_unique'] is True,
            'actual component owner/scratch/epoch lease')
    tables = lease['actual_block_tables']
    require(set(tables) == {str(i) for i in [*range(count), *range(64, 102)]}, 'actual page table owners')
    pages = [value for values in tables.values() for value in values if value != -1]
    require(all(type(v) is int and 0 <= v < 544 for v in pages)
            and len(pages) == len(set(pages)) == lease['allocated_pages'], 'actual unique page lease arithmetic')


def records_gate(reports, completed, directory):
    from audit_short_v126 import audit_reports
    original = audit_reports(reports, private_directory=directory)
    require(completed['action'] == ACTION and completed['actual_component_audit'] == original
            and completed['replies'] == [['calibrated', r, {'rows': 0, 'timings': 0}] for r in range(4)],
            'original strict audit/completion replies')
    draft = reports[0]['result']
    require(len(draft['warmup']) == 4, 'four actual FTF warmups')
    for warm in draft['warmup']:
        first, changed, restored = warm['outputs']
        require(first == restored and changed['root_token_ids'][0]
                == (first['root_token_ids'][0]+1) % 151936, 'actual cold/changed/restored roots/output')
    for phase_index, phase in enumerate([draft['protocol'], *draft['zero_protocol_traces']]):
        for index, row in enumerate(phase):
            require(row['outcome'] == OUTCOMES[index % 3], 'actual protocol outcome order')
            for report in reports[1:]:
                other = ([report['result']['protocol'], *report['result']['zero_protocol_traces']]
                         [phase_index][index])
                for field in ('observed_parent_output', 'observed_parent_accepted',
                              'exercised_parent_output', 'exercised_parent_accepted',
                              'promoted_owners', 'invalidated_owners'):
                    require(row[field] == other[field], 'all4 actual parent/commit metadata '+field)
                require([(c['consumed_owners'], c['observed_output'], c['observed_accepted'])
                         for c in row['real_child_consume']]
                        == [(c['consumed_owners'], c['observed_output'], c['observed_accepted'])
                            for c in other['real_child_consume']], 'all4 actual followup verdicts')
    cases = []
    for index, warm in enumerate(draft['warmup']):
        case = warm['case']; numerical = draft['numerical'][3*index:3*index+3]
        for row in numerical:
            lease_check(row, case)
        first = numerical[0]['generation']
        outcomes = []
        for row in draft['protocol'][3*index:3*index+3]:
            outcomes.append(dict(outcome=row['outcome'], parent_counterfactual=row['parent_counterfactual_protocol_only'],
                actual_parent_target_rows=len(row['parent_target_owners']), promoted_owners=row['promoted_owners'],
                invalidated_owners=row['invalidated_owners'], actual_followup=[dict(
                    consumed_owners=c['consumed_owners'], promoted_child_owners=c['promoted_child_owners'],
                    deferred_normal_owners=c['deferred_normal_owners'], actual_observed_only=c['actual_observed_only'])
                    for c in row['real_child_consume']]))
        cases.append(dict(case=case, actual_work_rows=case['normal']+case['eager'],
            actual_selected=first['selected'], actual_generated=first['actual_counter_delta']['generated_candidates'],
            actual_main_M=first['main_graph_M'], actual_observed_steps=first['return_tensor_or_list_steps_observed'],
            original_source_restored=True, cold_changed_restored_independent_records=3, outcomes=outcomes))
    canonical = json.dumps(original, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return dict(passed=True, original_strict_audit=original,
                original_strict_audit_canonical_sha256=hashlib.sha256(canonical).hexdigest(), cases=cases,
                per_rank_boundary_counts=[len(r['boundary_events']) for r in reports],
                per_rank_natural_resident_key_counts=[{k: len(v) for k, v in r['result']['actual_graph_states'].items()}
                                                    for r in reports],
                zero_sweeps=2, protocol_records_per_sweep=12, original_unified_seal=True,
                device_numerical_gates='Actual frozen producer executed own new PA/draft allphysical hidden/encoded/all28 KV and target independent same-M/all64 queryKV gates; strict raw auditor verifies their retained records.',
                HOST_tensor_numerics_recomputed=False, tensor_artifacts_available=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output', type=Path,
                        default=D/'ROOT_V134_ACTUAL_SHORT_PARENT_PROTOCOL_AUDIT.json')
    args = parser.parse_args()
    require(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*', args.tag), 'private actual tag')
    receipt_path = P/('receipt-'+args.tag+'.json')
    receipt = read_json(receipt_path)
    require(receipt.get('status') in ('complete', 'failed'), 'receipt must be terminal; do not poll a running job')
    require(not args.output.exists(), 'do not overwrite a prior actual audit attempt')
    directory = P/'actual'/args.tag
    result = dict(schema='ROOT.V134.actual-short-parent-protocol-audit.v1',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), status='failed', errors=[],
        tag=args.tag, original_job_status=receipt['status'], receipt=artifact(receipt_path),
        actual_HOST_python=sys.executable, actual_HOST_argv=sys.orig_argv,
        actual_HOST_environment={name: os.environ.get(name) for name in
            ('TORCH_DEVICE_BACKEND_AUTOLOAD', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
        analysis_script=artifact(Path(__file__).resolve()), NPU_initialized=False, model_initialized=False,
        hardware_queries=False, observed_only_outcome='observed_target',
        controlled_parent_outcomes=['valid_promotion', 'parent_reject'],
        actual_online_eager_admission_qualified=False, simultaneous_live_owner_alignment_qualified=False,
        live_overlap_qualified=False, online_storage527_qualified=False,
        full_workload_qualified=False, pure_draft_or_target_cost_qualified=False, formal_Goodput=False)
    # Independent checks retain usable numerical records even when an outer
    # cleanup/receipt check fails.  Their qualification scopes stay separate.
    for name, callback in (('source_closure', lambda: source_gate(receipt)),
                           ('cleanup', lambda: cleanup_gate(receipt, directory))):
        try:
            result[name] = callback()
        except BaseException:
            result[name] = dict(passed=False)
            result['errors'].append(name+':\n'+traceback.format_exc())
    try:
        paths = [directory/f'rank{r}-short-protocol.json.gz' for r in range(4)]
        result['reports'] = [artifact(path) for path in paths]
        reports = [read_json(path) for path in paths]
        result['completed'] = artifact(directory/'completed.json')
        require(result['source_closure']['passed'], 'refuse execution of an unbound strict auditor')
        result['actual_records'] = records_gate(reports, read_json(directory/'completed.json'), directory)
        require(not any(name == 'torch' or name.startswith(('torch.', 'torch_npu', 'vllm_ascend'))
                        for name in sys.modules), 'HOST-only import boundary')
    except BaseException:
        result['errors'].append('actual_records:\n'+traceback.format_exc())
        result['available_retained_actual_files'] = [artifact(path) for path in sorted(directory.glob('*'))
            if path.is_file() and path.suffix in ('.json', '.gz')]
        result['partial_records_are_full_qualification'] = False
    result['status'] = 'failed' if result['errors'] else 'passed'
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(dict(status=result['status'], original_job_status=receipt['status'],
        path=str(args.output), sha256=sha(args.output), error_count=len(result['errors']),
        actual_four_cases=(len(result.get('actual_records', {}).get('cases', []))),
        selected192_qualified=result.get('actual_records', {}).get('passed', False),
        full_workload_qualified=False, formal_Goodput=False)))
    return int(result['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
